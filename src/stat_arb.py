from pathlib import Path
import logging
from itertools import combinations

import pandas as pd
import numpy as np
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
import matplotlib.pyplot as plt

from logging_config import setup_logging


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures"

FIG_DIR.mkdir(exist_ok=True)

P_VALUE_THRESHOLD = 0.05
ROLLING_Z_WINDOW = 60
ENTRY_Z = 2.0
EXIT_Z = 0.5
ADF_ALPHA = 0.5
MIN_TRADES_FOR_SHARPE = 5
TRADING_DAYS_PER_YEAR = 252
TRADE_WINDOW = 63


def load_data():
    """
    Load raw parquet price data and pivot into wide format.

    The returned DataFrame is indexed by timestamp and contains one column
    per symbol with corresponding close prices.

    Returns
    -------
    pd.DataFrame
        Wide-format price table: index = timestamp, columns = symbols.
    """

    df = pd.read_parquet(DATA_DIR / "bars.parquet").reset_index()

    stock_data = df.pivot(
        index="timestamp",
        columns="symbol",
        values="close",
    )

    return stock_data


def run_adf_tests(stock_data):
    """
    Run Augmented Dickey-Fuller tests on each individual price series.

    Each column in the input DataFrame is treated as a separate time series.
    NaN values are dropped before testing.

    Parameters
    ----------
    stock_data : pd.DataFrame
        Wide-format price data indexed by timestamp.

    Returns
    -------
    None
        Results are logged (ADF statistic and p-value for each symbol).
    """
    adf_results = []

    cols = list(stock_data.columns)
    for i, symbol in enumerate(cols):
        series = stock_data[symbol].dropna()

        level_stat, level_pvalue, *_ = ts.adfuller(series)

        diff_series = series.diff().dropna()
        diff_stat, diff_pvalue, *_ = ts.adfuller(diff_series)

        is_i1 = (
            level_pvalue > ADF_ALPHA
            and diff_pvalue <= ADF_ALPHA
        )

        logger.debug(
            "%s | ADF Statistic: %.4f | p-value: %.4f",
            symbol,
            level_stat,
            level_pvalue,
        )

        adf_results.append(
            {
                "symbol": symbol,
                "level_stat": level_stat,
                "level_pvalue": level_pvalue,
                "diff_stat": diff_stat,
                "diff_pvalue": diff_pvalue,
                "is_i1": is_i1,
            }
        )

    return pd.DataFrame(adf_results)


def run_cointegration_tests(stock_data):
    """
    Test every unique ticker pair for cointegration.

    OLS Regression is run to compute the hedge ratio (beta)

    Parameters
    ----------
        stock_data : pandas.DataFrame
            Series of timestamped, close prices for all stocks
            extracted from the Alpaca API, pivoted to wide format.

    Returns
    -------
        results_df : pandas.DataFrame
            Cointegration test results for all candidate pairs.
    """

    pair_results = []

    for X_name, Y_name in combinations(stock_data.columns, 2):
        pair_prices = stock_data[[X_name, Y_name]].dropna()

        if len(pair_prices) < TRADING_DAYS_PER_YEAR:
            continue

        X = pair_prices[X_name]
        Y = pair_prices[Y_name]

        X_const = sm.add_constant(X, has_constant="add")

        model = sm.OLS(Y, X_const)
        results = model.fit()

        intercept = results.params["const"]
        hedge_ratio = results.params[X_name]

        coint_t, pvalue, crit_values = ts.coint(Y, X)

        pair_results.append(
            {
                "symbol_1": X_name,
                "symbol_2": Y_name,
                "intercept": intercept,
                "hedge_ratio": hedge_ratio,
                "t_stat": coint_t,
                "pvalue": pvalue,
                "critical_1%": crit_values[0],
                "critical_5%": crit_values[1],
                "critical_10%": crit_values[2],
            }
        )

    if not pair_results:
        return pd.DataFrame(
            columns=[
                "symbol_1",
                "symbol_2",
                "intercept",
                "hedge_ratio",
                "t_stat",
                "pvalue",
                "critical_1%",
                "critical_5%",
                "critical_10%",
            ]
        )

    results_df = pd.DataFrame(pair_results)
    results_df = results_df.sort_values("pvalue").reset_index(drop=True)

    return results_df


def filter_pairs(results_df):
    """
    Filter statistically significant pairs with approved economic relationships.

    Removes pairs whose cointegration p-value exceeds the specified
    threshold
    Parameters
    ----------
    results_df : pandas.DataFrame
        Cointegration test results for all candidate pairs.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame containing trading pairs whom exceed p-value threshold.
    """

    valid_pairs = results_df[results_df["pvalue"] <= P_VALUE_THRESHOLD]
    valid_pairs = valid_pairs.copy()

    return valid_pairs


def filter_by_economic_link(valid_pairs):
    """
    Excludes valid pairs without a plausible economic link.

    Parameters
    ---------
        valid_pairs : pd.DataFrame
            Trading pairs whom exceed p-value threshold.

    Returns
    -------
        approved_pairs : pd.DataFrame
            Filtered DataFrame of valid_pairs that have an approved economic link.
    """
    approved_pairs = {
        # Integrated energy majors
        tuple(sorted(("BP", "SHEL"))),
        tuple(sorted(("XOM", "CVX"))),
        tuple(sorted(("BP", "EQNR"))),
        # different business models, but shared-commodity exposure CAUTION
        tuple(sorted(("CVX", "DVN"))),
        tuple(sorted(("DVN", "XOM"))),
        tuple(sorted(("CVX", "EQNR"))),
        # Consumer staples: manufacturer <-> retailer
        tuple(sorted(("KHC", "WMT"))),
        tuple(sorted(("GIS", "WMT"))),
        tuple(sorted(("BG", "HSY"))),
        tuple(sorted(("PG", "WMT"))),
        tuple(sorted(("GIS", "PG"))),
    }

    valid_pairs["pair"] = [
        tuple(sorted(pair))
        for pair in zip(valid_pairs["symbol_1"], valid_pairs["symbol_2"])
    ]

    valid_pairs = valid_pairs[valid_pairs["pair"].isin(approved_pairs)]

    return valid_pairs.drop(columns="pair").reset_index(drop=True)


def compute_spread(X, Y, beta, intercept=0.0):
    """
    Compute the hedge-adjusted spread.
    spread_t = Y_t - (intercept + beta * X_t)
    """
    return Y - (intercept + beta * X)


def calc_zscore_signal(stock_data, valid_pairs):
    """
    Compute rolling z-score signals for approved trading pairs.

    For each pair, calculates the spread using the estimated hedge ratio
    and standardizes the spread using a rolling mean and standard
    deviation over a fixed window.

    Parameters
    ----------
    stock_data : pandas.DataFrame
        Wide-format price data with timestamps as rows and ticker
        symbols as columns.

    valid_pairs : pandas.DataFrame
        Approved trading pairs with estimated hedge ratios.

    Returns
    -------
    dict[str, dict[str, pandas.Series]]
        Dictionary containing the spread and z-score time series for
        each approved trading pair.
    """

    pair_objects = {}

    for _, row in valid_pairs.iterrows():
        X_name = row["symbol_1"]
        Y_name = row["symbol_2"]
        beta = row["hedge_ratio"]
        intercept = row.get("intercept", 0.0)

        pair_prices = stock_data[[X_name, Y_name]].dropna()

        if len(pair_prices) < ROLLING_Z_WINDOW + 2:
            continue

        X = pair_prices[X_name]
        Y = pair_prices[Y_name]

        spread = compute_spread(X, Y, beta, intercept)

        rolling_mean = spread.rolling(window=ROLLING_Z_WINDOW).mean()
        rolling_std = spread.rolling(window=ROLLING_Z_WINDOW).std()

        zscore = (spread - rolling_mean) / rolling_std.replace(0, np.nan)

        pair_objects[f"{X_name}:{Y_name}"] = {
            X_name: X,
            Y_name: Y,
            "intercept": intercept,
            "hedge_ratio": beta,
            "spread": spread,
            "zscore": zscore,
        }

    return pair_objects


def backtest(pair_objects):
    """
    Backtest the static-beta pairs strategy using gross-notional-normalized
    daily returns.

    The position carried into each iteration earns the price movement from
    the previous close to the current close. The current closing z-score is
    then used to enter or exit for the following trading interval.

    This ensures:
    - an exiting trade receives the return through its exit-day close;
    - a newly entered trade does not receive returns that occurred before
      its entry;
    - the hedge ratio and gross entry notional remain fixed during a trade.
    """
    backtest_results = {}

    for pair_name, pair_data in pair_objects.items():
        x_name, y_name = pair_name.split(":")

        beta = float(pair_data["hedge_ratio"])

        pair_df = pd.concat(
            {
                "X": pair_data[x_name],
                "Y": pair_data[y_name],
                "zscore": pair_data["zscore"],
            },
            axis=1,
        ).dropna()

        if len(pair_df) < 2:
            continue

        trade_count = 0
        current_pos = 0

        entry_x = None
        entry_y = None
        entry_beta = None
        gross_entry_notional = None

        return_values = []
        timestamps = []

        for i in range(1, len(pair_df)):
            previous_x = pair_df["X"].iloc[i - 1]
            previous_y = pair_df["Y"].iloc[i - 1]

            current_x = pair_df["X"].iloc[i]
            current_y = pair_df["Y"].iloc[i]

            delta_x = current_x - previous_x
            delta_y = current_y - previous_y

            if current_pos != 0:
                daily_return = (
                    current_pos
                    * (delta_y - entry_beta * delta_x)
                    / gross_entry_notional
                )
            else:
                daily_return = 0.0

            return_values.append(daily_return)
            timestamps.append(pair_df.index[i])

            z_current = pair_df["zscore"].iloc[i]

            if current_pos == 0:
                if z_current > ENTRY_Z:
                    current_pos = -1
                    trade_count += 1

                    entry_x = current_x
                    entry_y = current_y
                    entry_beta = beta

                    gross_entry_notional = abs(entry_y) + abs(entry_beta * entry_x)

                elif z_current < -ENTRY_Z:
                    current_pos = 1
                    trade_count += 1

                    entry_x = current_x
                    entry_y = current_y
                    entry_beta = beta

                    gross_entry_notional = abs(entry_y) + abs(entry_beta * entry_x)

            elif current_pos == -1 and z_current <= EXIT_Z:
                current_pos = 0

                entry_x = None
                entry_y = None
                entry_beta = None
                gross_entry_notional = None

            elif current_pos == 1 and z_current >= -EXIT_Z:
                current_pos = 0

                entry_x = None
                entry_y = None
                entry_beta = None
                gross_entry_notional = None

        return_series = pd.Series(
            data=return_values,
            index=timestamps,
            name=pair_name,
            dtype=float,
        )

        total_return = (1.0 + return_series).prod() - 1.0

        backtest_results[pair_name] = {
            "sim_days": len(return_series),
            "trade_count": trade_count,
            "total_return": total_return,
            "return_series": return_series,
        }

    return backtest_results


def run_walk_forward_backtest(
    stock_data,
    lookback_window=TRADING_DAYS_PER_YEAR,
    trade_window=TRADE_WINDOW,
):
    """
    Run a walk-forward out-of-sample backtest.

    For each fold:
    1. Use the lookback window to select cointegrated pairs and estimate
       static OLS hedge ratios.
    2. Trade the next trade window using only those previously estimated
       parameters.
    3. Store the out-of-sample P&L segment.
    4. Roll forward by trade_window days and repeat.
    """

    fold_results = []
    adf_fold_results = []
    combined_returns = {}
    combined_trade_counts = {}

    n_rows = len(stock_data)
    fold = 0

    for train_start in range(
        0,
        n_rows - lookback_window - trade_window + 1,
        trade_window,
    ):
        train_end = train_start + lookback_window
        test_end = train_end + trade_window

        train_data = stock_data.iloc[train_start:train_end]
        test_data = stock_data.iloc[train_end:test_end]

        fold += 1

        logger.info(
            "Walk-forward fold %d | Train: %s to %s | Test: %s to %s",
            fold,
            train_data.index.min(),
            train_data.index.max(),
            test_data.index.min(),
            test_data.index.max(),
        )

        adf_results_df = run_adf_tests(train_data)

        adf_results_df["fold"] = fold
        adf_results_df["train_start"] = train_data.index.min()
        adf_results_df["train_end"] = train_data.index.max()

        adf_fold_results.append(adf_results_df)

        eligible_symbols = adf_results_df.loc[
            adf_results_df["is_i1"],
            "symbol",
        ].tolist()

        if len(eligible_symbols) < 2:
            logger.info(
                "Fold %d skipped: fewer than two I(1) symbols.",
                fold,
            )
            continue

        eligible_train_data = train_data[eligible_symbols]

        results_df = run_cointegration_tests(
            eligible_train_data
            )

        valid_pairs = filter_pairs(results_df)
        approved_pairs = filter_by_economic_link(valid_pairs)

        if approved_pairs.empty:
            logger.info("Fold %d skipped: no approved pairs.", fold)
            continue

        logger.info(
            "Fold %d valid pairs: %s",
            fold,
            list(zip(valid_pairs["symbol_1"], valid_pairs["symbol_2"])),
        )

        logger.info(
            "Fold %d approved pairs: %s",
            fold,
            list(zip(approved_pairs["symbol_1"], approved_pairs["symbol_2"])),
        )

        buffer_start = max(0, train_end - ROLLING_Z_WINDOW)
        zscore_input = stock_data.iloc[buffer_start:test_end]

        pair_objects = calc_zscore_signal(zscore_input, approved_pairs)

        for pair_data in pair_objects.values():
            for key, value in pair_data.items():
                if isinstance(value, pd.Series):
                    pair_data[key] = value.loc[value.index.isin(test_data.index)]

        pair_objects = {
            pair_name: pair_data
            for pair_name, pair_data in pair_objects.items()
            if len(pair_data["zscore"].dropna()) >= 2
        }

        if not pair_objects:
            logger.info("Fold %d skipped: no usable z-score signals.", fold)
            continue

        fold_backtest_results = backtest(pair_objects)

        for pair_name, result in fold_backtest_results.items():
            return_series = result["return_series"]

            if pair_name not in combined_returns:
                combined_returns[pair_name] = []
                combined_trade_counts[pair_name] = 0

            combined_returns[pair_name].append(return_series)
            combined_trade_counts[pair_name] += result["trade_count"]

            fold_results.append(
                {
                    "fold": fold,
                    "pair_name": pair_name,
                    "train_start": train_data.index.min(),
                    "train_end": train_data.index.max(),
                    "test_start": test_data.index.min(),
                    "test_end": test_data.index.max(),
                    "trade_count": result["trade_count"],
                    "total_return": result["total_return"],
                }
            )

    combined_results = {}

    for pair_name, return_segments in combined_returns.items():
        return_series = pd.concat(return_segments).sort_index()

        combined_results[pair_name] = {
            "sim_days": len(return_series),
            "trade_count": combined_trade_counts[pair_name],
            "total_return": (1.0 + return_series).prod() - 1.0,
            "return_series": return_series,
        }

    fold_results_df = pd.DataFrame(
        fold_results,
        columns=[
            "fold",
            "pair_name",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "trade_count",
            "total_return",
        ],
    )

    if adf_fold_results:
        adf_fold_results_df = pd.concat(
            adf_fold_results,
            ignore_index=True,
        )
    else:
        adf_fold_results_df = pd.DataFrame()  

    adf_results_path = DATA_DIR / "static_walk_forward_adf_results.csv"

    adf_fold_results_df.to_csv(
        adf_results_path,
        index=False,
    )

    return combined_results, fold_results_df, adf_fold_results_df


def calc_cumulative_return(backtest_results):
    """
    Calculate compounded cumulative return for each trading pair.

    Parameters
    ----------
    backtest_results : dict[str, dict]
        Backtest results containing a daily normalized return series
        for each pair.

    Returns
    -------
    dict[str, dict]
        Updated results containing a compounded cumulative return series
        and final cumulative return.
    """
    for pair_name, data in backtest_results.items():
        return_series = pd.Series(
            data["return_series"],
            dtype=float,
        )
        cumulative_return_series = (1.0 + return_series).cumprod() - 1.0

        data["cumulative_return_series"] = cumulative_return_series
        data["final_cumulative_return"] = (
            cumulative_return_series.iloc[-1]
            if not cumulative_return_series.empty
            else 0.0
        )

    return backtest_results


def calc_return_sharpe_ratio(backtest_results):
    """
    Calculate annualized Sharpe ratios from normalized daily returns.
    """
    for data in backtest_results.values():
        trade_count = data["trade_count"]

        return_series = pd.Series(data["return_series"], dtype=float)

        if trade_count < MIN_TRADES_FOR_SHARPE:
            data["sharpe_ratio"] = np.nan
            continue

        return_std = return_series.std()

        if pd.isna(return_std) or return_std == 0:
            data["sharpe_ratio"] = np.nan
            continue

        data["sharpe_ratio"] = (return_series.mean() / return_std) * np.sqrt(
            TRADING_DAYS_PER_YEAR
        )

    return backtest_results


def calc_return_max_drawdown(backtest_results):
    """
    Calculate maximum drawdown for cumulative raw P&L results.
    """
    for pair_name, data in backtest_results.items():
        return_series = pd.Series(
            data["return_series"],
            dtype=float
        )

        equity_curve = (1.0 + return_series).cumprod()
        running_peak = equity_curve.cummax()

        drawdown_series = equity_curve / running_peak - 1.0

        data["equity_curve"] = equity_curve
        data["drawdown_series"] = drawdown_series

        data["max_drawdown"] = (
            abs(drawdown_series.min()) if not drawdown_series.empty else 0.0
        )

    return backtest_results


def log_return_backtest_results(backtest_results):
    """
    Log summary metrics for normalized-return backtest results.
    """
    for pair_name, result in backtest_results.items():
        logger.info(
            (
                "%s | Sim Days: %d | Trades: %d | "
                "Total Return: %.2f%% | "
                "Final Cum Return: %.2f%% | "
                "Sharpe: %.2f | "
                "Max Drawdown: %.2f%%"
            ),
            pair_name,
            result["sim_days"],
            result["trade_count"],
            result["total_return"] * 100,
            result["final_cumulative_return"] * 100,
            result["sharpe_ratio"],
            result["max_drawdown"] * 100,
        )


def plot_return_curves(backtest_results, fileName, plotTitle):
    """
    Plot compounded cumulative return curves for each approved pair.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for pair_name, data in backtest_results.items():
        cumulative_returns_series = data["cumulative_return_series"]
        sharpe = data["sharpe_ratio"]
        max_drawdown = data["max_drawdown"]

        ax.plot(
            cumulative_returns_series.index,
            cumulative_returns_series.values * 100,
            label=(
                f"{pair_name} |Sharpe: {sharpe:.2f} | MDD: {max_drawdown * 100:.2f}%"
            ),
        )

    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_title(plotTitle)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / fileName)
    plt.close(fig)


def main():
    setup_logging()

    stock_data = load_data()

    logger.info("Running walk-forward out-of-sample backtest.")
    backtest_results, fold_results_df, _ = run_walk_forward_backtest(stock_data)

    if not backtest_results:
        logger.warning("No out-of-sample return results were generated.")
        return


    fold_results_path = DATA_DIR / "static_walk_forward_folds.csv"

    fold_results_df.to_csv(
        fold_results_path,
        index=False,
    )

    logger.info("Saved walk-forward fold results to %s.", fold_results_path)

    backtest_results = calc_cumulative_return(backtest_results)
    backtest_results = calc_return_sharpe_ratio(backtest_results)
    backtest_results = calc_return_max_drawdown(backtest_results)

    log_return_backtest_results(backtest_results)

    logger.info("Plotting cumulative return curves")

    plot_return_curves(
        backtest_results,
        fileName="static_return_curves.png",
        plotTitle="Static Cumulative Return Curves",
    )


if __name__ == "__main__":
    main()


from pathlib import Path
import logging
from itertools import combinations

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import matplotlib.pyplot as plt

from logging_config import setup_logging


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures"

FIG_DIR.mkdir(exist_ok=True)


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
    `   NaN values are dropped before testing.

        Parameters
        ----------
        stock_data : pd.DataFrame
            Wide-format price data indexed by timestamp.

        Returns
        -------
        None
            Results are logged (ADF statistic and p-value for each symbol).
    """

    for symbol in stock_data.columns:
        series = stock_data[symbol].dropna()

        stat, pvalue, *_ = ts.adfuller(series)

        logger.info(
            "%s | ADF Statistic: %.4f | p-value: %.4f",
            symbol,
            stat,
            pvalue,
        )


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

        X = pair_prices[X_name]
        Y = pair_prices[Y_name]

        X_const = sm.add_constant(X)

        model = sm.OLS(Y, X_const)
        results = model.fit()

        hedge_ratio = results.params[X_name]

        coint_t, pvalue, crit_values = ts.coint(Y, X)

        pair_results.append(
            {
                "symbol_1": X_name,
                "symbol_2": Y_name,
                "hedge_ratio": hedge_ratio,
                "t_stat": coint_t,
                "pvalue": pvalue,
                "critical_1%": crit_values[0],
                "critical_5%": crit_values[1],
                "critical_10%": crit_values[2],
            }
        )

    results_df = pd.DataFrame(pair_results).sort_values("pvalue").reset_index(drop=True)

    results_df.to_csv(DATA_DIR / "pairs_testing.csv")

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

    valid_pairs = results_df[results_df["pvalue"] <= 0.05]
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
        tuple(sorted(("TTE", "XOM"))),
        # Consumer staples: manufacturer <-> retailer
        tuple(sorted(("KHC", "WMT"))),
        tuple(sorted(("GIS", "WMT"))),
        tuple(sorted(("BG", "HSY"))),
    }

    valid_pairs["pair"] = [
        tuple(sorted(pair))
        for pair in zip(valid_pairs["symbol_1"], valid_pairs["symbol_2"])
    ]

    valid_pairs = valid_pairs[valid_pairs["pair"].isin(approved_pairs)]

    return valid_pairs.drop(columns="pair").reset_index(drop=True)


def compute_spread(X, Y, beta):
    """
    Compute the hedge-adjusted spread between two price series.

    Returns
    -------
     pandas.Series
        Spread defined as:
            spread = Y - beta * X
    """
    return Y - beta * X


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

        pair_prices = stock_data[[X_name, Y_name]].dropna()

        X = pair_prices[X_name]

        Y = pair_prices[Y_name]

        spread = compute_spread(X, Y, beta)

        rolling_mean = spread.rolling(window=60).mean()
        rolling_std = spread.rolling(window=60).std()

        zscore = (spread - rolling_mean) / rolling_std
        pair_objects[f"{X_name}:{Y_name}"] = {
            X_name: X,
            Y_name: Y,
            "hedge_ratio": beta,
            "spread": spread,
            "zscore": zscore,
        }

    return pair_objects


def backtest(pair_objects):
    """
    Simulate a simple mean-reversion strategy on each approved pair.

    For each pair, walks the z-score series day by day and maintains a
    single-unit position (long spread, short spread, or flat) based on
    entry/exit z-score thresholds. Position decisions at each step use
    only the prior day's z-score to avoid look-ahead bias. Daily P&L is
    computed as the beta-adjusted spread return, scaled by the position
    held during that day.

    Parameters
    ----------
    pair_objects : dict[str, dict]
        Output of `calc_zscore_signal`. Each entry must contain the two
        price series, the hedge ratio, and the z-score series for one
        approved pair.

    Returns
    -------
    dict[str, list[float]]
        Daily P&L series for each pair, keyed by pair name.
    """
    pnl_results = {}

    for pair_name, pair_data in pair_objects.items():
        x_name, y_name = pair_name.split(":")

        trade_count = 0
        current_pos = 0
        pnl_values = []
        timestamps = []

        X = pair_data[x_name]
        Y = pair_data[y_name]
        zscores = pair_data["zscore"]
        beta = pair_data["hedge_ratio"]

        for i in range(1, len(zscores)):
            z_prev = zscores.iloc[i - 1]

            X_ret = X.iloc[i] - X.iloc[i - 1]
            Y_ret = Y.iloc[i] - Y.iloc[i - 1]

            spread_ret = Y_ret - beta * X_ret

            if current_pos == 0:
                if z_prev > 2:
                    current_pos = -1
                    trade_count += 1
                elif z_prev < -2:
                    current_pos = 1
                    trade_count += 1
            elif current_pos == -1:
                if z_prev <= 0.5:
                    current_pos = 0
            elif current_pos == 1:
                if z_prev >= -0.5:
                    current_pos = 0

            pnl = current_pos * spread_ret
            pnl_values.append(pnl)
            timestamps.append(zscores.index[i])

        pnl_series = pd.Series(data=pnl_values, index=timestamps)

        pnl_results[pair_name] = {
            "sim_days": len(zscores),
            "trade_count": trade_count,
            "total_pnl": sum(pnl_series),
            "pnl_series": pnl_series,
        }

    return pnl_results


def calc_cumulative_pnl(pnl_results):
    """
    Calculate cumulative P&L for each trading pair.

    Parameters
    ----------
    pnl_results : dict[str, dict]
        Backtest results containing daily P&L series for each pair.

    Returns
    -------
    dict[str, dict]
        Updated backtest results with cumulative P&L series and final
        cumulative P&L added.
    """
    for pair_name, data in pnl_results.items():
        pnl_series = pd.Series(data["pnl_series"])
        cum_pnl_series = pnl_series.cumsum()
        final_cum_pnl = cum_pnl_series.iloc[-1]

        data["cum_pnl_series"] = cum_pnl_series
        data["final_cum_pnl"] = final_cum_pnl

    return pnl_results


def calc_sharpe_ratio(pnl_results):
    """
    Calculate annualized Sharpe ratio for each trading pair.

    The Sharpe ratio is calculated as:
        mean(daily P&L) / std(daily P&L) * sqrt(252)

    Pairs with insufficient trading activity or zero volatility are
    assigned NaN Sharpe ratios.

    Parameters
    ----------
    pnl_results : dict[str, dict]
        Backtest results containing trade count and daily P&L series.

    Returns
    -------
    dict[str, dict]
        Updated backtest results with Sharpe ratio added.
    """

    for pair_name, data in pnl_results.items():
        trade_count = data["trade_count"]
        pnl_series = pd.Series(data["pnl_series"])

        if trade_count < 5:
            data["sharpe_ratio"] = np.nan
            continue

        series_mean = pnl_series.mean()
        series_std = pnl_series.std()

        if series_std == 0:
            data["sharpe_ratio"] = np.nan
            continue

        data["sharpe_ratio"] = (series_mean / series_std) * np.sqrt(252)

    return pnl_results


def calc_max_drawdown(pnl_results):
    """
    Calculate maximum drawdown for each trading pair.

    Maximum drawdown measures the largest decline from a historical
    cumulative P&L peak.

    Parameters
    ----------
    pnl_results : dict[str, dict]
        Backtest results containing cumulative P&L series.

    Returns
    -------
    dict[str, dict]
        Updated backtest results with maximum drawdown added.
    """
    for pair_name, data in pnl_results.items():
        cum_pnl_series = pd.Series(data["cum_pnl_series"])

        running_peak = cum_pnl_series.iloc[0]
        max_drawdown = 0

        for pnl in cum_pnl_series:
            if pnl > running_peak:
                running_peak = pnl

            drawdown = running_peak - pnl

            if drawdown > max_drawdown:
                max_drawdown = drawdown

        data["max_drawdown"] = max_drawdown

    return pnl_results


def log_backtest_results(pnl_results):
    """
    Log summary performance metrics for each trading pair.

    Parameters
    ----------
    pnl_results : dict[str, dict]
        Completed backtest results containing performance metrics.
    """
    for pair_name, result in pnl_results.items():
        logger.info(
            (
                "%s | Sim Days: %d | Trades: %d | "
                "Total PnL: %.2f |Final Cum Pnl: %.2f |Sharpe: %.2f | Max Drawdown: %.2f"
            ),
            pair_name,
            result["sim_days"],
            result["trade_count"],
            result["total_pnl"],
            result["final_cum_pnl"],
            result["sharpe_ratio"],
            result["max_drawdown"],
        )


def plot_equity_curves(pnl_results):
    """
    Plot cumulative equity curves for each approved trading pair.

    Parameters
    ----------
    pnl_results : dict[str, dict]
        Backtest results containing cumulative P&L series,
        Sharpe ratio, and maximum drawdown.

    Returns
    -------
    None
        Saves the figure to the figures directory.
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    for pair_name, data in pnl_results.items():
        cum_series = data["cum_pnl_series"]
        sharpe = data["sharpe_ratio"]
        max_draw = data["max_drawdown"]

        ax.plot(
            cum_series.index,
            cum_series.values,
            label=(f"{pair_name} | Sharpe: {sharpe:.2f} | MDD: {max_draw:.2f}"),
        )

    ax.axhline(
        y=0,
        color="black",
        linestyle="--",
        linewidth=1,
    )

    ax.set_title("Equity Curves")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative P&L")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "equity_curves.png")
    plt.close(fig)


def main():
    setup_logging()

    stock_data = load_data()

    logger.info("Running ADF tests.")
    run_adf_tests(stock_data)

    logger.info("Running pairwise cointegration tests.")
    results_df = run_cointegration_tests(stock_data)

    logger.info("Top approved pairs:")
    valid_pairs = filter_pairs(results_df)
    approved_pairs = filter_by_economic_link(valid_pairs)
    logger.info("\n%s", approved_pairs)

    logger.info("Calculating Z-score Signal for valid pairs.")
    pair_objects = calc_zscore_signal(stock_data, approved_pairs)
    logger.info("\n%s", pair_objects)

    logger.info("Backtesting")
    pnl_results = backtest(pair_objects)
    pnl_results = calc_cumulative_pnl(pnl_results)
    pnl_results = calc_sharpe_ratio(pnl_results)
    pnl_results = calc_max_drawdown(pnl_results)

    log_backtest_results(pnl_results)

    logger.info("Plotting Equity Curves")
    plot_equity_curves(pnl_results)


if __name__ == "__main__":
    main()

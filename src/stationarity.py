from pathlib import Path
import logging
from itertools import combinations

import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts

from logging_config import setup_logging


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"


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

    zscores = {}

    for _, row in valid_pairs.iterrows():
        X_name = row["symbol_1"]
        Y_name = row["symbol_2"]
        beta = row["hedge_ratio"]

        X = stock_data[X_name]
        Y = stock_data[Y_name]

        spread = compute_spread(X, Y, beta)

        rolling_mean = spread.rolling(window=60).mean()
        rolling_std = spread.rolling(window=60).std()

        zscore = (spread - rolling_mean) / rolling_std
        zscores[f"{X_name}:{Y_name}"] = {
            "spread": spread,
            "zscore": zscore,
        }

    return zscores


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
    zscores = calc_zscore_signal(stock_data, approved_pairs)
    logger.info("\n%s", zscores)


if __name__ == "__main__":
    main()

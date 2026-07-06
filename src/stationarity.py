from pathlib import Path
import logging
from logging_config import setup_logging
from itertools import combinations

import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"

def load_data():
    """
    Load price data and reshape into a wide dataframe:
        timestamp | AAPL | CVX | XOM | ...
    """

    df = pd.read_parquet(DATA_DIR / "bars.parquet").reset_index()

    stock_data = (
        df.pivot(
            index="timestamp",
            columns="symbol",
            values="close",
        )
        .dropna()
    )

    return stock_data

def run_adf_tests(stock_data):
    """
    Run an ADF test on every individual price series.
    """
    for symbol in stock_data.columns:
        series = stock_data[symbol]

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
    """

    pair_results = []

    for X_name, Y_name in combinations(stock_data.columns, 2):

        X = stock_data[X_name]
        Y = stock_data[Y_name]

        X_const = sm.add_constant(X)

        model = sm.OLS(Y, X_const)
        results = model.fit()

        hedge_ratio = results.params[X_name]

        coint_t, pvalue, crit_values = ts.coint(X, Y)

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

    results_df = (
        pd.DataFrame(pair_results)
        .sort_values("pvalue")
        .reset_index(drop=True)
    )

    return results_df

def main():
    setup_logging()

    stock_data = load_data()

    logger.info("Running ADF tests.")
    run_adf_tests(stock_data)

    logger.info("Running pairwise cointegration tests.")
    results_df = run_cointegration_tests(stock_data)

    logger.info("Top candidate pairs:")
    logger.info("\n%s", results_df.head(10))

    return results_df

if __name__ == "__main__":
    main()

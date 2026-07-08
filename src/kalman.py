from pathlib import Path
import logging

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pykalman import KalmanFilter

from logging_config import setup_logging
from stat_arb import (
    load_data,
    run_cointegration_tests,
    filter_pairs,
    filter_by_economic_link,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures"

FIG_DIR.mkdir(exist_ok=True)


def calc_kalman(
    X,
    Y,
    delta=1e-4,
    observation_covariance=None,
):
    """
    Estimate a dynamic hedge ratio and intercept using a Kalman filter.

    The spread returned here is the one-step-ahead prediction error
    / innovation:
        innovation_t = Y_t - E[Y_t | information through t-1]

    This avoids using the posterior state that has already incorporated
    Y_t.
    """
    pair_prices = pd.concat([X, Y], axis=1).dropna()

    X = pair_prices.iloc[:, 0].astype(float)
    Y = pair_prices.iloc[:, 1].astype(float)

    X_const = sm.add_constant(X, has_constant="add")
    ols_results = sm.OLS(Y, X_const).fit()

    initial_intercept = ols_results.params["const"]
    initial_beta = ols_results.params[X.name]

    if observation_covariance is None:
        observation_covariance = float(np.var(ols_results.resid, ddof=1))

    observation_covariance = max(observation_covariance, 1e-8)

    transition_matrix = np.eye(2)
    transition_covariance = delta / (1 - delta) * np.eye(2)

    state_mean = np.array([initial_intercept, initial_beta])
    state_covariance = np.eye(2)

    kf = KalmanFilter(
        transition_matrices=transition_matrix,
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance,
        initial_state_mean=state_mean,
        initial_state_covariance=state_covariance,
    )

    beta_values = []
    intercept_values = []
    innovation_values = []
    predicted_values = []

    for x_t, y_t in zip(X.values, Y.values):
        observation_matrix = np.array([[1.0, x_t]])

        # One-step-ahead prior prediction before observing y_t.
        predicted_state_mean = transition_matrix @ state_mean
        predicted_y = float(observation_matrix @ predicted_state_mean)

        innovation = y_t - predicted_y

        # Posterior update after observing y_t.
        state_mean, state_covariance = kf.filter_update(
            filtered_state_mean=state_mean,
            filtered_state_covariance=state_covariance,
            observation=y_t,
            observation_matrix=observation_matrix,
            transition_matrix=transition_matrix,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
        )

        intercept_values.append(state_mean[0])
        beta_values.append(state_mean[1])
        innovation_values.append(innovation)
        predicted_values.append(predicted_y)

    return (
        pd.Series(beta_values, index=X.index, name="kalman_beta"),
        pd.Series(intercept_values, index=X.index, name="kalman_intercept"),
        pd.Series(innovation_values, index=X.index, name="kalman_innovation"),
        pd.Series(predicted_values, index=X.index, name="kalman_predicted_y"),
    )


def build_static_pairs(stock_data, valid_pairs):

    pair_objects = {}

    window = 30

    for _, row in valid_pairs.iterrows():
        X_name = row["symbol_1"]
        Y_name = row["symbol_2"]

        pair_prices = stock_data[[X_name, Y_name]].dropna()

        X = pair_prices[X_name]
        Y = pair_prices[Y_name]

        X_const = sm.add_constant(X)

        model = sm.OLS(Y, X_const)
        results = model.fit()

        beta = results.params[X_name]
        intercept = results.params["const"]

        spread_series = Y - (intercept + beta * X)

        rolling_mean = spread_series.rolling(window).mean()
        rolling_std = spread_series.rolling(window).std()

        zscore = (spread_series - rolling_mean) / rolling_std

        pair_objects[(X_name, Y_name)] = {
            "spread": spread_series,
            "beta": beta,
            "intercept": intercept,
            "zscore": zscore,
        }

    return pair_objects


def build_kalman_pairs(stock_data, valid_pairs):

    pair_objects = {}

    window = 30

    for _, row in valid_pairs.iterrows():
        X_name = row["symbol_1"]
        Y_name = row["symbol_2"]

        pair_prices = stock_data[[X_name, Y_name]].dropna()

        X = pair_prices[X_name]
        Y = pair_prices[Y_name]

        beta_series, intercept_series, spread_series, predicted_y = calc_kalman(X, Y)

        rolling_mean = spread_series.rolling(window).mean()
        rolling_std = spread_series.rolling(window).std()

        zscore = (spread_series - rolling_mean) / rolling_std

        pair_objects[(X_name, Y_name)] = {
            "spread": spread_series,
            "beta": beta_series,
            "intercept": intercept_series,
            "zscore": zscore,
        }

    return pair_objects


def summarize_pair(pair_objects, pair_name, model_name):

    pair = pair_objects[pair_name]

    print(f"\n{'=' * 50}")
    print(f"{model_name}: {pair_name[0]} / {pair_name[1]}")
    print(f"{'=' * 50}")

    if np.isscalar(pair["beta"]):
        print("\nStatic Hedge Ratio:")
        print(f"Beta: {pair['beta']:.4f}")
        print(f"Intercept: {pair['intercept']:.4f}")

    else:
        print("\nDynamic Hedge Ratio:")
        print(f"Latest beta: {pair['beta'].iloc[-1]:.4f}")
        print(f"Latest intercept: {pair['intercept'].iloc[-1]:.4f}")

    logger.info("\nSpread:")
    logger.info(pair["spread"].tail())

    logger.info("\nZ-score:")
    logger.info(pair["zscore"].tail())


def main():
    setup_logging()

    logger.info("Loading stock data.")
    stock_data = load_data()

    results_df = run_cointegration_tests(stock_data)

    valid_pairs = filter_pairs(results_df)

    approved_pairs = filter_by_economic_link(valid_pairs)

    if approved_pairs.empty:
        logger.warning("No approved pairs found.")
        return

    logger.info("Comparing static and Kalman pairs.")

    static_pairs = build_static_pairs(stock_data, approved_pairs)

    kalman_pairs = build_kalman_pairs(stock_data, approved_pairs)

    pair_name = tuple(approved_pairs.loc[0, ["symbol_1", "symbol_2"]])

    summarize_pair(static_pairs, pair_name, "OLS Static")
    summarize_pair(kalman_pairs, pair_name, "Kalman Dynamic")


if __name__ == "__main__":
    main()

from pathlib import Path
import logging

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pykalman import KalmanFilter

from logging_config import setup_logging
from stat_arb import (
    ENTRY_Z,
    EXIT_Z,
    ROLLING_Z_WINDOW,
    TRADE_WINDOW,
    TRADING_DAYS_PER_YEAR,
    calc_cumulative_return,
    calc_return_max_drawdown,
    calc_return_sharpe_ratio,
    filter_by_economic_link,
    filter_pairs,
    load_data,
    log_return_backtest_results,
    plot_return_curves,
    run_cointegration_tests,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures"

FIG_DIR.mkdir(exist_ok=True)

# False: fresh yearly OLS anchor with light adaptation over train-tail + test.
# True: seed from yearly OLS, adapt across the full train window, then continue
# into test while retaining only the train tail for rolling z-score warm-up.
ADAPT_FULL_TRAIN = False


def _clean_pair_prices(X, Y):
    """Align two price series and return them as floats with missing rows removed."""
    pair_prices = pd.concat([X, Y], axis=1).dropna()

    if pair_prices.empty:
        return pair_prices.iloc[:, 0], pair_prices.iloc[:, 1]

    X_clean = pair_prices.iloc[:, 0].astype(float)
    Y_clean = pair_prices.iloc[:, 1].astype(float)

    return X_clean, Y_clean


def init_kalman_state(X_train, Y_train, delta=1e-4, observation_covariance=None):
    """
    Initialize a Kalman filter using only the training window.

    The initial intercept and hedge ratio come from an OLS regression fit on
    train data only. This avoids leaking test data into the initial state.
    """
    X_train, Y_train = _clean_pair_prices(X_train, Y_train)

    if len(X_train) < 2:
        raise ValueError(
            "Need at least two aligned observations to initialize Kalman filter."
        )

    X_const = sm.add_constant(X_train, has_constant="add")
    ols_results = sm.OLS(Y_train, X_const).fit()

    initial_intercept = ols_results.params["const"]
    initial_beta = ols_results.params[X_train.name]

    if observation_covariance is None:
        observation_covariance = float(np.var(ols_results.resid, ddof=1))

    observation_covariance = np.array([[max(observation_covariance, 1e-8)]])

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

    return kf, state_mean, state_covariance, observation_covariance


def run_kalman_forward(
    X_new,
    Y_new,
    kf,
    state_mean,
    state_covariance,
    observation_covariance,
):
    """
    Step an already-initialized Kalman filter through new observations.

    Returns posterior beta/intercept series plus the one-step-ahead innovation:
        innovation_t = Y_t - E[Y_t | information through t-1]
    """
    X_new, Y_new = _clean_pair_prices(X_new, Y_new)

    transition_matrix = kf.transition_matrices
    transition_covariance = kf.transition_covariance

    beta_values = []
    intercept_values = []
    innovation_values = []
    predicted_values = []

    for x_t, y_t in zip(X_new.values, Y_new.values):
        observation_matrix = np.array([[1.0, x_t]])

        # Prior prediction before observing y_t. This is the tradable residual.
        predicted_state_mean = transition_matrix @ np.asarray(state_mean).reshape(-1)
        predicted_y = (observation_matrix @ predicted_state_mean).item()
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

        state_mean = np.asarray(state_mean).reshape(-1)

        intercept_values.append(state_mean[0])
        beta_values.append(state_mean[1])
        innovation_values.append(innovation)
        predicted_values.append(predicted_y)

    return (
        pd.Series(beta_values, index=X_new.index, name="kalman_beta"),
        pd.Series(intercept_values, index=X_new.index, name="kalman_intercept"),
        pd.Series(innovation_values, index=X_new.index, name="kalman_innovation"),
        pd.Series(predicted_values, index=X_new.index, name="kalman_predicted_y"),
        state_mean,
        state_covariance,
    )


def build_kalman_pairs_for_fold(train_data, forward_data, test_index, approved_pairs):
    """
    Build Kalman pair objects for one walk-forward fold.

    `train_data` initializes the filter from a fresh OLS anchor. `forward_data`
    should contain either:
    - train tail + test when ADAPT_FULL_TRAIN is False, giving light adaptation
      from the yearly OLS anchor before trading;
    - test only when ADAPT_FULL_TRAIN is True, because the filter is already
      adapted across the full training window before test.

    When ADAPT_FULL_TRAIN is True, the filter first walks through the full
    training window so the state has time to adapt before the test period.
    The returned series are trimmed to the test window, but the lagged hedge
    ratio is created before trimming so the first test day can use the final
    pre-test beta.
    """
    pair_objects = {}

    for _, row in approved_pairs.iterrows():
        X_name = row["symbol_1"]
        Y_name = row["symbol_2"]

        train_pair_prices = train_data[[X_name, Y_name]].dropna()
        forward_pair_prices = forward_data[[X_name, Y_name]].dropna()

        if len(train_pair_prices) < TRADING_DAYS_PER_YEAR:
            logger.debug(
                "Skipping %s:%s: insufficient train observations.", X_name, Y_name
            )
            continue

        if len(forward_pair_prices) < 2:
            logger.debug(
                "Skipping %s:%s: insufficient forward observations.", X_name, Y_name
            )
            continue

        X_train = train_pair_prices[X_name]
        Y_train = train_pair_prices[Y_name]

        kf, state_mean, state_covariance, observation_covariance = init_kalman_state(
            X_train,
            Y_train,
        )

        if ADAPT_FULL_TRAIN:
            (
                train_beta,
                train_intercept,
                train_innovation,
                train_predicted_y,
                state_mean,
                state_covariance,
            ) = run_kalman_forward(
                X_train,
                Y_train,
                kf,
                state_mean,
                state_covariance,
                observation_covariance,
            )

            # In full-train adaptation mode, only process the test window after
            # the train state has converged. The train tail is retained only for
            # rolling z-score warm-up.
            test_pair_prices = forward_pair_prices.loc[
                forward_pair_prices.index.intersection(test_index)
            ]

            if len(test_pair_prices) < 2:
                logger.debug(
                    "Skipping %s:%s: insufficient test observations.", X_name, Y_name
                )
                continue

            X_forward = test_pair_prices[X_name]
            Y_forward = test_pair_prices[Y_name]

            (
                test_beta,
                test_intercept,
                test_innovation,
                test_predicted_y,
                _,
                _,
            ) = run_kalman_forward(
                X_forward,
                Y_forward,
                kf,
                state_mean,
                state_covariance,
                observation_covariance,
            )

            beta = pd.concat([train_beta.tail(ROLLING_Z_WINDOW), test_beta])
            intercept = pd.concat(
                [train_intercept.tail(ROLLING_Z_WINDOW), test_intercept]
            )
            innovation = pd.concat(
                [train_innovation.tail(ROLLING_Z_WINDOW), test_innovation]
            )
            predicted_y = pd.concat(
                [train_predicted_y.tail(ROLLING_Z_WINDOW), test_predicted_y]
            )

            X_prices = pd.concat([X_train.tail(ROLLING_Z_WINDOW), X_forward])
            Y_prices = pd.concat([Y_train.tail(ROLLING_Z_WINDOW), Y_forward])

        else:
            if len(forward_pair_prices) < ROLLING_Z_WINDOW + 2:
                logger.debug(
                    "Skipping %s:%s: insufficient forward observations.", X_name, Y_name
                )
                continue

            X_forward = forward_pair_prices[X_name]
            Y_forward = forward_pair_prices[Y_name]

            (
                beta,
                intercept,
                innovation,
                predicted_y,
                _,
                _,
            ) = run_kalman_forward(
                X_forward,
                Y_forward,
                kf,
                state_mean,
                state_covariance,
                observation_covariance,
            )

            X_prices = X_forward
            Y_prices = Y_forward

        rolling_mean = innovation.rolling(window=ROLLING_Z_WINDOW).mean()
        rolling_std = innovation.rolling(window=ROLLING_Z_WINDOW).std()
        zscore = (innovation - rolling_mean) / rolling_std.replace(0, np.nan)

        # Created before trimming. This preserves the final
        # pre-test beta as the lagged beta for the first test day.
        hedge_ratio_lagged = beta.shift(1)

        pair_key = f"{X_name}:{Y_name}"
        pair_objects[pair_key] = {
            X_name: X_prices,
            Y_name: Y_prices,
            "intercept": intercept,
            "hedge_ratio": beta,
            "hedge_ratio_lagged": hedge_ratio_lagged,
            "spread": innovation,
            "zscore": zscore,
            "predicted_y": predicted_y,
        }

    trimmed_pair_objects = {}

    for pair_name, pair_data in pair_objects.items():
        trimmed_pair_data = {}

        for key, value in pair_data.items():
            if isinstance(value, pd.Series):
                trimmed_pair_data[key] = value.loc[value.index.intersection(test_index)]
            else:
                trimmed_pair_data[key] = value

        if len(trimmed_pair_data["zscore"].dropna()) >= 2:
            trimmed_pair_objects[pair_name] = trimmed_pair_data

    return trimmed_pair_objects


def backtest_dynamic_beta(pair_objects):
    """
    Backtest Kalman pair objects using gross-notional-normalized returns.

    Entry and exit decisions use the previous day's z-score. When a trade
    opens, the hedge ratio and gross notional are fixed for the life of
    the trade.

    Daily return:
        position * (delta_y - entry_beta * delta_x)
        ------------------------------------------------
        abs(entry_y) + abs(entry_beta * entry_x)
    """
    backtest_results = {}

    for pair_name, pair_data in pair_objects.items():
        x_name, y_name = pair_name.split(":")

        pair_df = pd.concat(
            {
                "X": pair_data[x_name],
                "Y": pair_data[y_name],
                "zscore": pair_data["zscore"],
                "hedge_ratio_lagged": pair_data["hedge_ratio_lagged"],
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
            z_prev = pair_df["zscore"].iloc[i - 1]
            beta_for_entry = pair_df["hedge_ratio_lagged"].iloc[i]

            previous_x = pair_df["X"].iloc[i - 1]
            previous_y = pair_df["Y"].iloc[i - 1]

            current_x = pair_df["X"].iloc[i]
            current_y = pair_df["Y"].iloc[i]

            delta_x = current_x - previous_x
            delta_y = current_y - previous_y

            if current_pos == 0:
                if z_prev > ENTRY_Z:
                    current_pos = -1
                    trade_count += 1

                    entry_x = previous_x
                    entry_y = previous_y
                    entry_beta = beta_for_entry

                    gross_entry_notional = abs(entry_y) + abs(entry_beta * entry_x)

                elif z_prev < -ENTRY_Z:
                    current_pos = 1
                    trade_count += 1

                    entry_x = previous_x
                    entry_y = previous_y
                    entry_beta = beta_for_entry

                    gross_entry_notional = abs(entry_y) + abs(entry_beta * entry_x)

            elif current_pos == -1 and z_prev <= EXIT_Z:
                current_pos = 0

                entry_x = None
                entry_y = None
                entry_beta = None
                gross_entry_notional = None

            elif current_pos == 1 and z_prev >= -EXIT_Z:
                current_pos = 0

                entry_x = None
                entry_y = None
                entry_beta = None
                gross_entry_notional = None

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

        return_series = pd.Series(
            data=return_values,
            index=timestamps,
            name=pair_name,
            dtype=float,
        )

        total_return = (1.0 + return_series).prod() - 1.0

        backtest_results[pair_name] = {
            "sim_days": len(pair_df),
            "trade_count": trade_count,
            "total_return": total_return,
            "return_series": return_series,
        }

    return backtest_results


def run_kalman_walk_forward_backtest(
    stock_data,
    lookback_window=TRADING_DAYS_PER_YEAR,
    trade_window=TRADE_WINDOW,
):
    """
    Run a walk-forward out-of-sample Kalman backtest.

    For each fold:
    1. Select pairs using cointegration tests on train data only.
    2. Initialize each Kalman filter using train data only.
    3. Run the filter through train-tail + test for z-score warm-up.
    4. Trim to test only and backtest with a lagged dynamic beta.
    """
    fold_results = []
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
            "Kalman fold %d | Train: %s to %s | Test: %s to %s",
            fold,
            train_data.index.min(),
            train_data.index.max(),
            test_data.index.min(),
            test_data.index.max(),
        )

        results_df = run_cointegration_tests(train_data)
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

        if ADAPT_FULL_TRAIN:
            forward_data = test_data
        else:
            buffer_start = max(0, train_end - ROLLING_Z_WINDOW)
            forward_data = stock_data.iloc[buffer_start:test_end]

        pair_objects = build_kalman_pairs_for_fold(
            train_data=train_data,
            forward_data=forward_data,
            test_index=test_data.index,
            approved_pairs=approved_pairs,
        )

        if not pair_objects:
            logger.info("Fold %d skipped: no usable Kalman signals.", fold)
            continue

        fold_backtest_results = backtest_dynamic_beta(pair_objects)

        if not fold_backtest_results:
            logger.info("Fold %d skipped: no Kalman returns generated.", fold)
            continue

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

    return combined_results, fold_results_df


def main():
    setup_logging()

    logger.info("Loading stock data.")
    stock_data = load_data()

    logger.info("Running Kalman walk-forward out-of-sample backtest.")
    backtest_results, fold_results_df = run_kalman_walk_forward_backtest(stock_data)

    if not backtest_results:
        logger.warning("No Kalman out-of-sample return results were generated.")
        return

    fold_results_path = DATA_DIR / "kalman_walk_forward_folds.csv"
    fold_results_df.to_csv(fold_results_path, index=False)
    logger.info("Saved Kalman fold results to %s.", fold_results_path)

    backtest_results = calc_cumulative_return(backtest_results)
    backtest_results = calc_return_sharpe_ratio(backtest_results)
    backtest_results = calc_return_max_drawdown(backtest_results)

    log_return_backtest_results(backtest_results)

    logger.info("Plotting Kalman cumulative return curves.")
    plot_return_curves(
        backtest_results,
        fileName="kalman_return_curves.png",
        plotTitle="Kalman Cumulative Return Curves",
    )


if __name__ == "__main__":
    main()

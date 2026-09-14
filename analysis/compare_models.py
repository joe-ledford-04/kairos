import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from kalman import run_kalman_walk_forward_backtest
from logging_config import setup_logging
from stat_arb import (
    calc_cumulative_return,
    calc_return_max_drawdown,
    calc_return_sharpe_ratio,
    load_data,
    run_walk_forward_backtest,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures"

DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


def build_summary_df(backtest_results):
    rows = []

    for pair_name, result in backtest_results.items():
        rows.append(
            {
                "pair_name": pair_name,
                "sim_days": result["sim_days"],
                "trade_count": result["trade_count"],
                "total_return": result["total_return"],
                "final_cumulative_return": result["final_cumulative_return"],
                "sharpe_ratio": result["sharpe_ratio"],
                "max_drawdown": result["max_drawdown"],
            }
        )

    return pd.DataFrame(rows)


def plot_pair_comparison(
    pair_name,
    static_result,
    kalman_result,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    static_curve = static_result["cumulative_return_series"]
    kalman_curve = kalman_result["cumulative_return_series"]

    ax.plot(
        static_curve.index,
        static_curve.values * 100,
        label=(
            f"Static | Sharpe: {static_result['sharpe_ratio']:.2f} | "
            f"MDD: {static_result['max_drawdown'] * 100:.2f}%"
        ),
    )

    ax.plot(
        kalman_curve.index,
        kalman_curve.values * 100,
        label=(
            f"Kalman | Sharpe: {kalman_result['sharpe_ratio']:.2f} | "
            f"MDD: {kalman_result['max_drawdown'] * 100:.2f}%"
        ),
    )

    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)

    ax.set_title(f"{pair_name} — Static vs. Kalman")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend()

    fig.tight_layout()

    safe_pair_name = pair_name.replace(":", "_")

    fig.savefig(
        FIG_DIR / f"{safe_pair_name}_static_vs_kalman.png"
    )

    plt.close(fig)


def main():
    setup_logging()

    # Silence INFO chatter from the imported backtest modules.
    logging.getLogger("stat_arb").setLevel(logging.WARNING)
    logging.getLogger("kalman").setLevel(logging.WARNING)

    stock_data = load_data()

    logger.info("Running static walk-forward backtest.")

    (
        static_backtest_results,
        static_fold_results_df,
        _,
    ) = run_walk_forward_backtest(stock_data)

    if not static_backtest_results:
        logger.warning("No static out-of-sample results were generated.")
        return

    logger.info("Running Kalman walk-forward backtest.")

    (
        kalman_backtest_results,
        kalman_fold_results_df,
    ) = run_kalman_walk_forward_backtest(stock_data)

    if not kalman_backtest_results:
        logger.warning("No Kalman out-of-sample results were generated.")
        return

    static_backtest_results = calc_cumulative_return(
        static_backtest_results
    )
    static_backtest_results = calc_return_sharpe_ratio(
        static_backtest_results
    )
    static_backtest_results = calc_return_max_drawdown(
        static_backtest_results
    )

    kalman_backtest_results = calc_cumulative_return(
        kalman_backtest_results
    )
    kalman_backtest_results = calc_return_sharpe_ratio(
        kalman_backtest_results
    )
    kalman_backtest_results = calc_return_max_drawdown(
        kalman_backtest_results
    )

    static_summary = build_summary_df(static_backtest_results)
    kalman_summary = build_summary_df(kalman_backtest_results)

    comparison_df = pd.merge(
        static_summary,
        kalman_summary,
        on="pair_name",
        how="inner",
        suffixes=("_static", "_kalman"),
    )

    comparison_df["sharpe_delta"] = (
        comparison_df["sharpe_ratio_kalman"]
        - comparison_df["sharpe_ratio_static"]
    )

    comparison_df["return_delta"] = (
        comparison_df["final_cumulative_return_kalman"]
        - comparison_df["final_cumulative_return_static"]
    )

    comparison_df["drawdown_delta"] = (
        comparison_df["max_drawdown_kalman"]
        - comparison_df["max_drawdown_static"]
    )

    comparison_df = comparison_df.sort_values(
        "sharpe_delta",
        ascending=False,
    ).reset_index(drop=True)

    comparison_path = DATA_DIR / "static_vs_kalman_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    logger.info(
        "Saved static vs. Kalman comparison to %s.",
        comparison_path,
    )

    logger.info("\n%s", comparison_df)

    static_pairs = set(static_backtest_results)
    kalman_pairs = set(kalman_backtest_results)

    static_only = static_pairs - kalman_pairs
    kalman_only = kalman_pairs - static_pairs

    if static_only:
        logger.warning(
            "Pairs present only in static results: %s",
            sorted(static_only),
        )

    if kalman_only:
        logger.warning(
            "Pairs present only in Kalman results: %s",
            sorted(kalman_only),
        )

    matching_pairs = static_pairs & kalman_pairs

    for pair_name in sorted(matching_pairs):
        plot_pair_comparison(
            pair_name,
            static_backtest_results[pair_name],
            kalman_backtest_results[pair_name],
        )


if __name__ == "__main__":
    main()

# Kairos

A statistical arbitrage trading system that trades US equities via Alpaca's paper trading API. Kairos identifies pairs of stocks whose prices historically move together, detects when they've drifted apart in a statistically abnormal way, and generates signals to bet on that relationship reverting — with an AI agent layer planned to catch the cases where the relationship breaks for good.

**Status: actively in development.** This README describes both what's built today and where the project is headed.

## The idea

Two companies with similar underlying businesses (say, two integrated oil majors, or a snack manufacturer and its retailer) often have prices that move together over time — not because one causes the other, but because they're exposed to the same economic forces. When their prices temporarily diverge from that historical relationship, there's a statistical case for betting they'll converge again.

The catch: sometimes they don't converge, because something changed the underlying relationship — a merger, a guidance shock, a regulatory event. A model that only looks at price history has no way to know that. Kairos is built around that problem specifically: pair the statistical signal with an AI research layer that can catch fundamental breaks the numbers alone can't see.

## Three pillars

- **Time-series statistics** — cointegration testing (Engle-Granger, ADF), spread construction, rolling z-score signal generation.
- **Machine learning** — Kalman filtering for dynamic hedge ratio estimation, replacing a static OLS regression with an estimate that adapts as the relationship between two stocks drifts.
- **AI agents** — an LLM-based research agent embedded in the trading pipeline itself, not just used to build it, acting as a fundamental risk gate between signal and execution.

## What's built so far

- **Data pipeline** — pulls two years of daily bars for a curated universe of ~20 tickers across energy and consumer staples sectors via the Alpaca Market Data API, stored as Parquet.
- **Cointegration screening** — tests every pairwise combination (150+ pairs) for stationarity (ADF) and cointegration (Engle-Granger), ranking results by p-value.
- **Economic-link filtering** — cross-references statistically significant pairs against a hand-curated set of pairs with a plausible business rationale (e.g. same sector, shared input costs, manufacturer/retailer relationships), rather than trading purely on statistical coincidence.
- **Signal generation** — computes a hedge ratio via OLS, constructs the price spread, and standardizes it into a rolling z-score.
- **Backtesting engine** — simulates a simple threshold-based mean-reversion strategy (enter at ±2σ, exit near 0) with no look-ahead bias, and reports:
  - Cumulative P&L
  - Annualized Sharpe ratio
  - Maximum drawdown
  - Equity curve plots across all approved pairs
- **Logging** — structured logging to console and file across the full pipeline.
- **Kalman filter** — dynamic hedge ratio estimation, benchmarked head-to-head against the static OLS baseline on the same pairs and metrics.

## In progress

- **SQL persistence** — moving pair statistics, trade decisions, and backtest results out of flat files and into MySQL.

## Planned
- **AI research agent (LangGraph)** — for any pair flagged as a trade candidate, retrieves recent news and filings for both tickers and evaluates whether a fundamental event (M&A, guidance change, regulatory action) should override the statistical signal. Flagged trades are skipped and logged with a reason; the reasoning becomes an audit trail.
- **Live paper-trading loop** — moving from historical backtests to actually placing orders against Alpaca's paper trading environment.
- **Reporting agent** — a daily generated memo summarizing trades taken, trades skipped, and why — grounded in the same sources the research agent used.
- **FastAPI service layer** — exposing the pipeline (trigger a scan, view current positions, retrieve the daily memo) as an API.
- **Containerization + Azure deployment** — Docker Compose locally, then a deployed version on Azure Container Apps, using Azure Key Vault for credentials.
- **Tests and evals** — pytest coverage for the statistical core, plus LangSmith-based evaluation of whether the agent's risk calls were justified in hindsight.

## Project structure

```
kairos/
├── data/                  # Parquet price data, cointegration test results (gitignored)
├── figures/               # Generated equity curve plots (gitignored)
├── logs/                  # Application logs (gitignored)
├── src/
│   ├── data_extract.py    # Alpaca API client, historical bar download
│   ├── stat_arb.py        # Cointegration screening, z-score signal, backtest engine
│   ├── kalman.py          # Dynamic hedge ratio estimation (in progress)
│   └── logging_config.py  # Shared logging setup
├── .env.example           # Required environment variables (no real values)
└── .gitignore
```

## Tech stack

**Current:** Python, pandas, NumPy, statsmodels, Alpaca API, matplotlib, `uv` for environment management, Ruff for linting/formatting.

**Planned:** LangGraph, LangSmith, FastAPI, MySQL, Docker, Azure (Container Apps, Key Vault, AI Foundry), pytest.

## Setup

```bash
git clone <repo-url>
cd kairos
uv sync
cp .env.example .env   # then fill in your Alpaca API credentials
```

Requires an [Alpaca](https://alpaca.markets/) account with paper trading API keys.

Run the data pipeline and statistical core:

```bash
uv run python src/data_extract.py
uv run python src/stat_arb.py
```

## Disclaimer

This project trades exclusively against Alpaca's **paper trading** environment. It is a personal research and learning project, not investment advice, and is not intended for use with real capital.

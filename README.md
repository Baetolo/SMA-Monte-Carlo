# SMA 500 — 10,000-run Monte Carlo

## Required data
Place `data/daily_equity.csv` from your **actual SMA 500 chronological portfolio backtest** in this repository. It must have columns `date,equity` (alternatively `datetime`/`timestamp` and `portfolio_equity`/`account_equity`/`balance`). Exactly one end-of-day portfolio equity observation per trading day, sorted automatically. At least 60 observations; 2–5 years preferred. Include no deposits, withdrawals, or equity resets. The historical portfolio backtest MUST already have been run at **2% risk per trade**, including cash, position overlap, simultaneous signals, sizing limits, costs, and correlated stock moves. A signal-scanner export or isolated per-stock trades CSV is NOT sufficient. The script deliberately fails if data is absent or mismatched.

## GitHub steps
1. Create a new private GitHub repository named `sma500-monte-carlo`.
2. Upload all files and folders from this ZIP, preserving `.github/workflows/monte-carlo.yml`.
3. Upload your historical backtest equity file to `data/daily_equity.csv`. If your columns differ, rename to `date,equity`.
4. Verify your SMA 500 backtest was run at **2% risk per trade**, with a single combined chronological portfolio and realistic trading fees/slippage. If not, rerun the historical backtest first.
5. Open **Actions → SMA 500 Monte Carlo (research only) → Run workflow**.
6. After the green tick, open the run, scroll to **Artifacts**, and download `sma500-monte-carlo-results`.
7. Inspect `summary.csv`, `simulations.csv`, and `equity_paths.png`.

## Interpretation
This runs a 10-day circular moving-block bootstrap of historical *portfolio* daily returns, over the same number of trading days as the historical sample, with 10,000 resampled paths and a fixed random seed. Consecutive historical portfolio days remain intact inside blocks. This partially preserves observed clustering, cross-stock correlation and overlapping positions that were already embedded in the backtest. It does NOT rerun the strategy, reproduce all constraints at block boundaries, model new order fills, or guarantee future outcomes. The 10% loss threshold is a STATIC threshold against the initial A$50,000, NOT a funded-account trailing drawdown or daily loss rule. No deposits/withdrawals, taxes or FX conversion are modeled. If backtest equity is denominated in USD, convert the underlying account properly to AUD before interpreting AUD results; do not just relabel the currency. Avoid using optimized/in-sample-only data; test out-of-sample periods and stress costs separately.

To run locally: `pip install -r requirements.txt && python monte_carlo.py`.

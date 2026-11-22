# XAUUSD 3-minute VWAP + 9 EMA research backtester

**Research only. No live orders. No claim of funded-account profitability.**

## What it does
- Reads broker-provided **1-minute XAUUSD OHLCV CSV** with `timestamp,open,high,low,close,volume`. Timestamp must be the **start** of the minute; timezone-aware UTC ISO timestamps preferred. For timezone-naive timestamps use `--timezone`.
- Builds only complete 3-minute candles (no synthetic filling).
- Calculates EMA(9) and a VWAP reset at **midnight Europe/Prague** (a research assumption; can be changed). XAUUSD broker volume is tick volume, not consolidated volume.
- At confirmed candle close, detects 9 EMA cross above/below VWAP. Submits orders for the **next 3-minute candle open**. Every signal is logged.
- Allows **multiple concurrent positions**, long and short simultaneously if signals alternate. **No daily trade cap or strategy drawdown guards.** Positions do not automatically reverse or close on a new crossover.
- Each entry risks approximately **0.5% of current mark-to-market equity** using ATR(14) × 1.5 as stop distance and a **4R take-profit**. Position sizing uses a default **100 ounces per lot** and rounds down to **0.01 lots**. Verify instrument contract specs with your broker/FTMO.
- Applies configurable spread, slippage, and commission **placeholders**. Default spread 0.30 USD, slippage 0.05 USD per side, commission $3.50/lot/side are NOT verified FTMO costs.
- Checks estimated margin availability at entry (default leverage 30:1); rejected orders are logged, not silently skipped. **Does not simulate broker margin calls, liquidation or financing.**
- When stop and target are both touched within a candle, assumes **stop first**. Intrabar sequence is unknown. Gap-through-stop prices are treated adversely.
- Positions still open at the end are **marked to market but not forcibly closed**. Trade CSV contains closed trades only.

## Run on GitHub
1. Create a new GitHub repository (or upload this folder's **contents** to an existing empty repository).
2. Obtain legally licensed historical **1-minute XAUUSD OHLCV** from one broker/provider, ideally covering several years. **Alpaca's US stock API does not supply spot XAUUSD.**
3. Name the data file **`data/xauusd_1m.csv`**. Do NOT rename `FORMAT_EXAMPLE.csv`; it is illustrative only, not usable backtest history. For large datasets, use a data download step or Git LFS / release asset rather than pushing a multi-GB CSV into Git.
4. GitHub **Actions → Gold VWAP EMA backtest and exploratory Monte Carlo → Run workflow**.
5. Download the **gold-backtest-results** artifact. It contains `signals.csv`, `trades.csv`, `equity_curve.csv`, `daily_equity.csv`, `summary.csv`, `monte_carlo_simulations.csv`, `monte_carlo_summary.csv`.

## Run locally
```
pip install -r requirements.txt
python backtest.py --csv data/xauusd_1m.csv --timezone UTC
python monte_carlo.py --simulations 10000
```

## FTMO caveats (critical)
- The backtest **does not stop** trading after an FTMO breach, per the requested strategy. It flags approximate 5% daily and 10% overall breach events in the historical equity series, but these are **not exact FTMO calculations**.
- The Monte Carlo is **exploratory only**. It resamples **daily closed trade P&L**, not intraday paths. It **cannot reliably estimate FTMO pass rates**, because it misses intraday floating losses, carryover positions, exact reset accounting, and margin liquidations. Its reported pass rate is a screening statistic, not a funded challenge pass probability.
- A proper FTMO simulation must replay timestamped equity paths at finer resolution, track all open positions, use current provider rules, enforce the precise reset clock, stop each failed account immediately, and reset equity between phases. Consider overnight swap and news/weekend restrictions for the chosen account type.
- Using many concurrent positions can multiply exposure; 0.5% **per position** does not mean 0.5% **total risk**.
- Backtest parameters are **not optimized**. Use out-of-sample years and broker-realistic execution costs before interpreting results.

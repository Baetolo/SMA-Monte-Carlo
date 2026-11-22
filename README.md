# SMA 500 chronological backtest + Monte Carlo

## Required original files
Copy **your exact original** `backtest.py` (the code you pasted into ChatGPT) and `stocks.csv` (the same original stock universe) into this folder at repository root. They are **not included**, because the exact 500-symbol universe was not supplied and should not be invented.

## GitHub setup
1. Create a **private** GitHub repository, e.g. `sma500-monte-carlo`.
2. Upload **the contents** of this ZIP (not its enclosing directory), preserving `.github/workflows/monte-carlo.yml`.
3. Upload your original `backtest.py` and `stocks.csv` at repository root.
4. Select **Actions** → **SMA 500 Backtest and Monte Carlo** → **Run workflow**.
5. Set `usd_per_aud` to a rate you choose (USD per AUD). Default `0.65` is an **illustration**, not a verified market rate.
6. Download `sma500-backtest-and-monte-carlo` from the completed workflow's **Artifacts**.

## Outputs
- `results/portfolio/trades.csv`: actual chronological trades with 2% risk and 10 position cap
- `results/portfolio/equity_curve_usd.csv`: chronological portfolio equity in USD
- `results/portfolio/summary.csv`: portfolio performance and missing symbol count
- `results/portfolio/missing_symbols.csv`: symbols without usable data
- `results/monte_carlo/summary.csv`: Monte Carlo statistics
- `results/monte_carlo/simulations.csv`: 10,000 simulated outcomes
- `results/monte_carlo/equity_paths.png`: sampled simulated paths

## Method and caveats
The original backtest handles signal timing, simultaneous holdings, cash allocation, stop-first OHLC execution, fees, and a 10-position cap. The Monte Carlo step **resamples blocks of the resulting combined daily portfolio returns**, preserving within-block correlation and portfolio constraints already embedded in historical equity. It does **not** rerun the stock-by-stock portfolio engine in each simulation, preserve original calendar chronology across blocks, model changing market regimes, or reproduce funded-account daily/trailing drawdown rules. The simulated 10% loss threshold is relative to initial capital. FX is assumed constant; AUD/USD volatility and conversion costs are excluded. Yahoo Finance data availability and survivorship bias in the supplied 500-stock list can materially affect results. This is research, not a prediction or trading guarantee.


## Reconstructed stock list limitation
`stocks.csv` contains only the **100 distinct tickers recoverable from the uploaded historical trades**, NOT the original 500-stock universe. The other 400 cannot be recovered from executed trades alone. Replace `stocks.csv` with the original full 500-symbol list before interpreting results as a replication of the original SMA 500 backtest. The 100-ticker run is only a partial-universe test.

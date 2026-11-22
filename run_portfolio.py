
import argparse
from pathlib import Path

import pandas as pd

from backtest import download, simulate

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--capital-aud", type=float, default=50000)
    parser.add_argument("--usd-per-aud", type=float, default=0.65)
    parser.add_argument("--risk", type=float, default=2.0)
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--start", default="2021-11-17")
    parser.add_argument("--end", default="2026-11-17")

    args = parser.parse_args()

    if args.capital_aud <= 0:
        raise ValueError("Starting capital must be positive.")

    if args.usd_per_aud <= 0:
        raise ValueError("Exchange rate must be positive.")

    if not 0 < args.risk <= 100:
        raise ValueError("Invalid risk percentage.")

    if args.max_positions < 1:
        raise ValueError("Maximum positions must be at least 1.")

    capital_usd = args.capital_aud * args.usd_per_aud

    print("=== SMA 500 PORTFOLIO BACKTEST ===")
    print(f"Starting capital AUD: ${args.capital_aud:,.2f}")
    print(f"Starting capital USD: ${capital_usd:,.2f}")
    print(f"Risk per trade: {args.risk}%")
    print(f"Maximum positions: {args.max_positions}")
    print(f"Period: {args.start} to {args.end}")

    # Create output folders
    data_dir = ROOT / "data"
    results_dir = ROOT / "results"

    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load stock universe
    stocks_file = ROOT / "stocks.csv"

    if not stocks_file.exists():
        raise FileNotFoundError("stocks.csv is missing.")

    tickers = [
        line.strip().upper().replace(".", "-")
        for line in stocks_file.read_text().splitlines()
        if line.strip()
    ]

    tickers = list(dict.fromkeys(tickers))

    print(f"Stocks requested: {len(tickers)}")

    # Download historical market data
    prices, failed = download(
        tickers,
        args.start,
        args.end
    )

    if not prices:
        raise RuntimeError(
            "No historical prices downloaded. "
            "Check Yahoo Finance availability."
        )

    print(f"Stocks downloaded: {len(prices)}")
    print(f"Stocks missing: {len(failed)}")

    # Run chronological portfolio backtest
    curve, trades, summary = simulate(
        prices,
        args.start,
        args.end,
        capital=capital_usd,
        risk_pct=args.risk,
        max_positions=args.max_positions,
        fee=0.001,
        atr_mult=2.0,
        reward_risk=3.0,
        max_hold=15
    )

    # Save USD equity curve and historical trades
    curve.to_csv(
        results_dir / "equity_curve.csv",
        index=False
    )

    trades.to_csv(
        results_dir / "trades.csv",
        index=False
    )

    # Save daily equity in AUD for Monte Carlo
    aud = curve.copy()

    for column in ["equity", "cash"]:
        if column in aud.columns:
            aud[column] = aud[column] / args.usd_per_aud

    aud.to_csv(
        data_dir / "daily_equity.csv",
        index=False
    )

    # Save summary
    summary["starting_capital_aud"] = args.capital_aud
    summary["starting_capital_usd"] = capital_usd
    summary["usd_per_aud"] = args.usd_per_aud
    summary["risk_pct"] = args.risk
    summary["max_positions"] = args.max_positions
    summary["symbols_requested"] = len(tickers)
    summary["symbols_with_data"] = len(prices)
    summary["symbols_missing"] = len(failed)

    summary["ending_equity_aud"] = (
        float(curve["equity"].iloc[-1]) / args.usd_per_aud
    )

    pd.Series(summary).to_csv(
        results_dir / "summary.csv",
        header=["value"]
    )

    pd.DataFrame({
        "ticker": failed
    }).to_csv(
        results_dir / "missing_symbols.csv",
        index=False
    )

    # Display results
    print("\n=== BACKTEST RESULTS ===")

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nFiles successfully saved.")
    print("Monte Carlo input: data/daily_equity.csv")
    print("Historical results: results/")
    print("Backtest completed successfully.")


if __name__ == "__main__":
    main()

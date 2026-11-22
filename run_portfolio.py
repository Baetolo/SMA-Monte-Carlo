"""Run the user's original backtest.py with explicit AUD funding and 2% risk."""
import argparse
from pathlib import Path
import pandas as pd
import backtest

ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--start', default='2021-11-17')
    p.add_argument('--end', default='2026-11-17')
    p.add_argument('--capital-aud', type=float, default=50000)
    p.add_argument('--usd-per-aud', type=float, required=True, help='Explicit assumption: USD received for 1 AUD; not live FX')
    p.add_argument('--risk', type=float, default=2)
    p.add_argument('--max-positions', type=int, default=10)
    a = p.parse_args()
    if a.capital_aud <= 0 or not 0 < a.usd_per_aud < 2 or not 0 < a.risk <= 100 or a.max_positions < 1:
        p.error('Invalid capital, FX, risk, or position limit')
    path = ROOT / 'stocks.csv'
    if not path.exists():
        p.error('Missing stocks.csv: upload your ORIGINAL 500-symbol list')
    tickers = list(dict.fromkeys(s.strip().upper().replace('.', '-') for s in path.read_text().splitlines() if s.strip() and s.strip().lower() != 'ticker'))
    prices, failed = backtest.download(tickers, a.start, a.end)
    if not prices:
        raise RuntimeError('No prices downloaded')
    capital_usd = a.capital_aud * a.usd_per_aud
    curve, trades, summary = backtest.simulate(prices, a.start, a.end, capital=capital_usd, risk_pct=a.risk, max_positions=a.max_positions)
    out = ROOT / 'results' / 'portfolio'
    out.mkdir(parents=True, exist_ok=True)
    curve.to_csv(out / 'equity_curve_usd.csv', index=False)
    trades.to_csv(out / 'trades.csv', index=False)
    pd.DataFrame({'ticker': failed}).to_csv(out / 'missing_symbols.csv', index=False)
    # Constant FX assumption: no currency fluctuations, spreads or FX fees modeled.
    aud = curve.copy()
    for c in ['equity', 'cash']:
        aud[c] = aud[c] / a.usd_per_aud
   (ROOT / 'data').mkdir(parents=True, exist_ok=True)
aud.to_csv(ROOT / 'data' / 'daily_equity.csv', index=False)
    summary.update({'capital_aud': a.capital_aud, 'usd_per_aud_assumed': a.usd_per_aud, 'capital_usd': capital_usd, 'symbols_requested_total': len(tickers), 'symbols_with_data': len(prices), 'symbols_missing': len(failed), 'currency_assumption': 'constant USD/AUD; FX fluctuations and conversion costs excluded'})
    pd.Series(summary).to_csv(out / 'summary.csv', header=['value'])
    print(pd.Series(summary).to_string())
    if len(prices) < len(tickers):
        print('WARNING: some symbols missing; see results/portfolio/missing_symbols.csv')

if __name__ == '__main__':
    main()

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_equity(path, initial, verified_risk):
    df = pd.read_csv(path)
    cols = {str(c).lower().strip(): c for c in df.columns}
    date_col = next((cols[x] for x in ['date', 'datetime', 'timestamp', 'day'] if x in cols), None)
    equity_col = next((cols[x] for x in ['equity', 'portfolio_equity', 'account_equity', 'balance', 'end_equity', 'equity_aud'] if x in cols), None)
    if date_col is None or equity_col is None:
        raise ValueError(f'Expected date and equity columns. Found: {list(df.columns)}')
    d = pd.DataFrame({'date': pd.to_datetime(df[date_col], errors='coerce', utc=True), 'equity': pd.to_numeric(df[equity_col], errors='coerce')})
    if d.isna().any().any() or (d.equity <= 0).any():
        raise ValueError('Invalid dates, missing values, or nonpositive equity.')
    d = d.sort_values('date')
    if d.date.duplicated().any():
        raise ValueError('Duplicate dates: provide exactly one end-of-day portfolio equity per trading day.')
    if len(d) < 60:
        raise ValueError('At least 60 daily equity observations required; several years strongly preferred.')
    if verified_risk != 2.0:
        raise ValueError('This simulation requires a historical portfolio backtest already run at 2% risk per trade. Do not rescale daily equity by guessing.')
    # First recorded equity is treated as the start observation, not an extra return.
    returns = d.equity.pct_change().iloc[1:].to_numpy(dtype=float)
    if np.any(~np.isfinite(returns)) or np.any(returns <= -1):
        raise ValueError('Invalid daily returns; ensure no deposits, withdrawals, or equity resets.')
    return d, returns


def simulate(returns, capital, runs, block, seed, horizon, threshold):
    rng = np.random.default_rng(seed)
    n = len(returns)
    if block > n: raise ValueError('Block length exceeds available history')
    if horizon <= 0: raise ValueError('Horizon must be positive')
    # Circular block bootstrap; consecutive days stay together within each block.
    final = np.empty(runs)
    dd = np.empty(runs)
    breaches = np.empty(runs, dtype=bool)
    sample_paths = []
    for i in range(runs):
        starts = rng.integers(0, n, size=(horizon + block - 1)//block)
        ix = (starts[:, None] + np.arange(block)[None, :]) % n
        r = returns[ix.ravel()[:horizon]]
        curve = capital * np.concatenate(([1.0], np.cumprod(1+r)))
        peak = np.maximum.accumulate(curve)
        drawdowns = 1 - curve / peak
        final[i] = curve[-1]
        dd[i] = drawdowns.max()
        breaches[i] = np.any(curve <= capital * (1-threshold))
        if i < 100: sample_paths.append(curve)
    return final, dd, breaches, sample_paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--equity', default='data/daily_equity.csv')
    p.add_argument('--capital', type=float, default=50000)
    p.add_argument('--risk', type=float, default=2.0)
    p.add_argument('--runs', type=int, default=10000)
    p.add_argument('--block', type=int, default=10)
    p.add_argument('--horizon', type=int, default=0, help='Trading days; 0 = observed history length')
    p.add_argument('--loss-limit', type=float, default=0.10, help='Static initial-capital loss threshold, not prop-firm rule')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default='results')
    a = p.parse_args()
    if not 0 < a.loss_limit < 1 or a.runs < 1 or a.capital <= 0 or a.block < 1:
        p.error('Invalid parameters')
    d, r = load_equity(a.equity, a.capital, a.risk)
    horizon = a.horizon or len(r)
    final, dd, breaches, paths = simulate(r, a.capital, a.runs, a.block, a.seed, horizon, a.loss_limit)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'ending_equity_aud': final, 'max_drawdown_pct': dd*100, 'static_loss_limit_breached': breaches}).to_csv(out/'simulations.csv', index=False)
    summary = {
        'simulations': a.runs, 'starting_capital_aud': a.capital,
        'historical_risk_per_trade_pct_verified': a.risk,
        'historical_observations': len(d), 'simulated_trading_days': horizon,
        'block_length_days': a.block,
        'historical_start': str(d.date.iloc[0].date()), 'historical_end': str(d.date.iloc[-1].date()),
        'probability_positive_return_pct': 100*np.mean(final > a.capital),
        'median_ending_equity_aud': np.median(final),
        'p05_ending_equity_aud': np.percentile(final, 5),
        'p95_ending_equity_aud': np.percentile(final, 95),
        'median_max_drawdown_pct': np.median(dd)*100,
        'p95_max_drawdown_pct': np.percentile(dd, 95)*100,
        'probability_static_loss_limit_breached_pct': np.mean(breaches)*100,
        'static_loss_limit_pct_of_initial_capital': a.loss_limit*100,
        'historical_total_return_pct': (d.equity.iloc[-1]/d.equity.iloc[0]-1)*100,
    }
    pd.DataFrame([summary]).to_csv(out/'summary.csv', index=False)
    fig, ax = plt.subplots(figsize=(11, 6))
    for path in paths: ax.plot(path, alpha=.12, linewidth=.7)
    ax.axhline(a.capital, linestyle='--', linewidth=1, label='Starting equity')
    ax.set(title='SMA 500 — Monte Carlo sample portfolio paths', xlabel='Trading days', ylabel='Equity (AUD)')
    ax.legend(); fig.tight_layout(); fig.savefig(out/'equity_paths.png', dpi=150); plt.close(fig)
    print(pd.DataFrame([summary]).to_string(index=False))

if __name__ == '__main__': main()

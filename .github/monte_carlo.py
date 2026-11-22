"""Exploratory day-block bootstrap of realised trade P&L; NOT tick-accurate FTMO certification."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

OUT=Path('outputs')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--simulations',type=int,default=10000)
    p.add_argument('--max-days',type=int,default=250)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--capital',type=float,default=50000)
    a=p.parse_args()
    trades=pd.read_csv(OUT/'trades.csv')
    if trades.empty: raise SystemExit('No closed trades. Monte Carlo needs trade history.')
    trades['exit_time_utc']=pd.to_datetime(trades.exit_time_utc,utc=True)
    trades['session']=trades.exit_time_utc.dt.tz_convert('Europe/Prague').dt.date
    # Includes zero-trade days from the backtest; closed-P&L approximation.
    days=pd.read_csv(OUT/'daily_equity.csv')['session'].astype(str)
    pnl=trades.groupby(trades.session.astype(str)).net_pnl.sum()
    blocks=days.map(pnl).fillna(0).to_numpy(dtype=float)
    if len(blocks)<30: print('WARNING: fewer than 30 observed sessions; results very unstable.')
    rng=np.random.default_rng(a.seed)
    rows=[]
    for k in range(a.simulations):
        phases=[]
        for target_pct in (.10,.05):
            balance=a.capital
            status='TIMEOUT'
            used=0
            # Sample days WITH replacement; does not preserve changing volatility regimes.
            for d in range(1,a.max_days+1):
                # Scale historical daily P&L by current balance to approximate 0.5% sizing.
                day_pnl=float(rng.choice(blocks))*balance/a.capital
                # Daily check is END-OF-DAY only: misses intraday floating breaches.
                if day_pnl <= -.05*a.capital:
                    status='FAIL_DAILY_EOD';used=d;break
                balance+=day_pnl
                if balance <= .90*a.capital:
                    status='FAIL_TOTAL_EOD';used=d;break
                if balance >= a.capital*(1+target_pct) and d>=4:
                    status='PASS';used=d;break
            if not used: used=a.max_days
            phases.append((status,used,balance))
            if status!='PASS': break
        rows.append({'simulation':k+1,'phase1_status':phases[0][0],
                     'phase1_days':phases[0][1],
                     'phase2_status':phases[1][0] if len(phases)>1 else 'NOT_STARTED',
                     'phase2_days':phases[1][1] if len(phases)>1 else np.nan,
                     'overall_pass':len(phases)==2 and phases[1][0]=='PASS'})
    result=pd.DataFrame(rows)
    result.to_csv(OUT/'monte_carlo_simulations.csv',index=False)
    summary={'simulations':a.simulations,'historical_days':len(blocks),
             'phase1_pass_pct':100*(result.phase1_status=='PASS').mean(),
             'overall_two_phase_pass_pct':100*result.overall_pass.mean(),
             'max_days_per_phase':a.max_days,
             'warning':'Exploratory CLOSED-PNL daily bootstrap; ignores intraday/floating breach and cross-day overlapping position effects. NOT a valid FTMO pass probability.'}
    pd.DataFrame([summary]).to_csv(OUT/'monte_carlo_summary.csv',index=False)
    print(pd.Series(summary).to_string())

if __name__=='__main__':main()

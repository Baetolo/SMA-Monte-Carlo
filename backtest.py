import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
ROOT = Path(__file__).resolve().parent

def download(tickers, start, end, batch_size=15):
    prices, failed = {}, []
    for i in range(0, len(tickers), batch_size):
        group = tickers[i:i+batch_size]
        print(f'Downloading {i+1}-{min(i+batch_size,len(tickers))} of {len(tickers)}', flush=True)
        try:
            df = yf.download(group, start=(pd.Timestamp(start)-pd.Timedelta(days=80)).strftime('%Y-%m-%d'), end=(pd.Timestamp(end)+pd.Timedelta(days=1)).strftime('%Y-%m-%d'), auto_adjust=True, group_by='ticker', threads=False, progress=False)
            for t in group:
                try:
                    p = df[t].copy() if isinstance(df.columns,pd.MultiIndex) else df.copy()
                    p = p[['Open','High','Low','Close','Volume']].dropna()
                    if len(p)>=25: prices[t]=p.sort_index()
                    else: failed.append(t)
                except (KeyError,TypeError): failed.append(t)
        except Exception as e:
            print(f'Batch failed: {str(e)[:150]}')
            failed += group
        time.sleep(2)
    return prices, sorted(set(failed))

def prepare(p):
    sma10=p.Close.rolling(10).mean()
    sma20=p.Close.rolling(20).mean()
    adr=((p.High-p.Low)/p.Low*100).rolling(20).mean()
    vol=p.Volume.rolling(20).mean()
    active=(p.Close>sma10)&(p.Close>sma20)&(sma10>sma20)&(adr>5)&(vol>500000)
    previous=p.Close.shift(1)
    tr=pd.concat([(p.High-p.Low),(p.High-previous).abs(),(p.Low-previous).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    return p.assign(buy=active & ~active.shift(1,fill_value=False),atr=atr,strength=p.Close/sma20-1)

def simulate(prices,start,end,capital=1000.,risk_pct=1.,max_positions=10,fee=.001,atr_mult=2.,reward_risk=3.,max_hold=15):
    prepared={t:prepare(p) for t,p in prices.items()}
    dates=sorted({d for p in prepared.values() for d in p.index if pd.Timestamp(start)<=d<=pd.Timestamp(end)})
    if not dates: raise RuntimeError('No dates in requested period.')
    cash=capital; positions={}; last_close={}; pending={}; completed=[]; equity_rows=[]
    for day in dates:
        candidates=sorted(pending.items(),key=lambda x:-x[1][0]); pending={}
        for t,(strength,signal_close,signal_atr) in candidates:
            if t in positions or len(positions)>=max_positions or day not in prepared[t].index: continue
            row=prepared[t].loc[day]; opening=float(row.Open)
            stop=signal_close-atr_mult*signal_atr; distance=opening-stop
            if not np.isfinite(distance) or distance<=0 or opening<=0: continue
            equity_open=cash+sum(v['shares']*(float(prepared[s].loc[day,'Open']) if day in prepared[s].index else last_close.get(s,v['entry'])) for s,v in positions.items())
            risk_budget=max(0,equity_open*risk_pct/100)
            risk_per_share=distance+fee*(opening+(opening+reward_risk*distance))
            by_risk=int(risk_budget/risk_per_share)
            by_slot=int((equity_open/max_positions)/(opening*(1+fee)))
            by_cash=int(cash/(opening*(1+fee)))
            shares=min(by_risk,by_slot,by_cash)
            if shares<1: continue
            cash-=shares*opening*(1+fee)
            positions[t]={'shares':shares,'entry':opening,'entry_day':day,'entry_cost':shares*opening*(1+fee),'stop':stop,'target':opening+reward_risk*distance,'bars':0,'planned_risk':risk_budget,'signal_date':str((day-pd.Timedelta(days=1)).date())}
        for t,v in list(positions.items()):
            if day not in prepared[t].index: continue
            row=prepared[t].loc[day]; v['bars']+=1
            op,hi,lo,cl=map(float,[row.Open,row.High,row.Low,row.Close]); price,reason=None,None
            if op<=v['stop']: price,reason=op,'gap_through_stop'
            elif op>=v['target']: price,reason=op,'gap_through_target'
            elif lo<=v['stop']: price,reason=v['stop'],'stop'
            elif hi>=v['target']: price,reason=v['target'],'target'
            elif v['bars']>=max_hold: price,reason=cl,'time_exit_at_close'
            elif day==dates[-1]: price,reason=cl,'end_of_test'
            if price is not None:
                proceeds=v['shares']*price*(1-fee); cash+=proceeds; pnl=proceeds-v['entry_cost']
                completed.append({'ticker':t,'entry_date':str(v['entry_day'].date()),'exit_date':str(day.date()),'shares':v['shares'],'entry_price':v['entry'],'exit_price':price,'stop_price':v['stop'],'target_price':v['target'],'pnl_usd':round(pnl,4),'return_on_entry_pct':round(100*pnl/v['entry_cost'],3),'planned_risk_usd':round(v['planned_risk'],4),'loss_exceeded_budget':bool(pnl < -v['planned_risk']),'exit_reason':reason,'holding_bars':v['bars']})
                del positions[t]
        for t,p in prepared.items():
            if day not in p.index: continue
            row=p.loc[day]; last_close[t]=float(row.Close)
            if t not in positions and bool(row.buy) and pd.notna(row.atr) and day!=dates[-1]: pending[t]=(float(row.strength),float(row.Close),float(row.atr))
        equity=cash+sum(v['shares']*last_close.get(t,v['entry']) for t,v in positions.items())
        equity_rows.append({'date':str(day.date()),'equity':round(equity,4),'cash':round(cash,4),'open_positions':len(positions)})
    curve=pd.DataFrame(equity_rows); trades=pd.DataFrame(completed)
    pnl=trades.pnl_usd if not trades.empty else pd.Series(dtype=float)
    gains=pnl[pnl>0].sum(); losses=-pnl[pnl<0].sum(); peak=curve.equity.cummax()
    years=(pd.Timestamp(curve.date.iloc[-1])-pd.Timestamp(curve.date.iloc[0])).days/365.25
    total=curve.equity.iloc[-1]/capital-1
    summary={'requested_start':start,'requested_end':end,'actual_start':curve.date.iloc[0],'actual_end':curve.date.iloc[-1],'symbols_requested':len(prices),'initial_capital':capital,'ending_equity':round(float(curve.equity.iloc[-1]),2),'return_pct':round(100*total,2),'annualized_return_pct':round(100*((1+total)**(1/years)-1),2) if years>0 and total>-1 else None,'max_drawdown_pct':round(100*(curve.equity/peak-1).min(),2),'closed_trades':len(trades),'win_rate_pct':round(100*(pnl>0).mean(),2) if len(pnl) else None,'profit_factor':round(gains/losses,3) if losses>0 else None,'losses_exceeding_planned_risk':int(trades.loss_exceeded_budget.sum()) if len(trades) else 0,'risk_pct':risk_pct,'max_positions':max_positions,'fee_each_side_pct':fee*100}
    return curve,trades,summary

def main():
    a=argparse.ArgumentParser(); a.add_argument('--start',default='2021-11-17'); a.add_argument('--end',default='2026-11-17'); a.add_argument('--risk',type=float,default=1.0); a.add_argument('--max-positions',type=int,default=10)
    args=a.parse_args()
    tickers=[s.strip().upper().replace('.','-') for s in (ROOT/'stocks.csv').read_text().splitlines() if s.strip()]
    if not (0<args.risk<=100 and args.max_positions>=1): raise ValueError('Invalid risk or max positions')
    prices,failed=download(tickers,args.start,args.end)
    if not prices: raise RuntimeError('No data downloaded; Yahoo may be rate-limiting GitHub.')
    curve,trades,summary=simulate(prices,args.start,args.end,risk_pct=args.risk,max_positions=args.max_positions)
    summary['symbols_with_data']=len(prices); summary['symbols_missing']=len(failed)
    curve.to_csv(ROOT/'equity_curve.csv',index=False); trades.to_csv(ROOT/'trades.csv',index=False)
    pd.DataFrame({'ticker':failed}).to_csv(ROOT/'missing_symbols.csv',index=False)
    pd.Series(summary).to_csv(ROOT/'summary.csv',header=['value'])
    for k,v in summary.items():print(f'{k}: {v}')
    print('Outputs saved.')
if __name__=='__main__': main()

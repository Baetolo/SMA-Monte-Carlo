import argparse
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path('outputs')

def load_bars(path, timezone):
    df = pd.read_csv(path)
    df.columns = [x.strip().lower() for x in df.columns]
    if 'timestamp' not in df and 'time' in df:
        df = df.rename(columns={'time': 'timestamp'})
    needed = {'timestamp','open','high','low','close','volume'}
    if not needed.issubset(df.columns):
        raise ValueError(f'CSV must contain {sorted(needed)}; found {list(df.columns)}')
    ts = pd.to_datetime(df.timestamp, utc=False, errors='raise')
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(timezone, ambiguous='raise', nonexistent='raise')
    ts = ts.dt.tz_convert('UTC')
    df.index = pd.DatetimeIndex(ts)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='last')]
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['open','high','low','close','volume'])
    df = df[(df.high >= df.low) & (df.volume >= 0) & (df.open > 0)]
    if df.empty: raise ValueError('No usable OHLCV rows')
    return df

def prepare(df, bar_minutes=3):
    # Source timestamps must mark the OPEN of each 1-minute candle.
    # UTC anchoring keeps 3-minute bars aligned on minute 00,03,...
    bars = df.resample(f'{bar_minutes}min', label='left', closed='left').agg(
        {'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
    counts = df.open.resample(f'{bar_minutes}min', label='left', closed='left').count()
    bars = bars[counts == bar_minutes].copy()  # exclude incomplete candles and gaps
    if bars.empty: raise ValueError('No complete 3-minute bars: use 1-minute OHLCV data')
    bars['session'] = bars.index.tz_convert('Europe/Prague').date
    # Daily VWAP reset at midnight Prague, a documented research assumption.
    # Spot XAUUSD volume is broker tick volume, not consolidated exchange volume.
    typical = (bars.high + bars.low + bars.close) / 3
    numerator = typical * bars.volume
    group = bars.session
    vol_cum = bars.volume.groupby(group).cumsum()
    bars['vwap'] = numerator.groupby(group).cumsum() / vol_cum.replace(0, np.nan)
    bars['ema9'] = bars.close.ewm(span=9, adjust=False).mean()
    prev_close = bars.close.shift(1)
    tr = pd.concat([(bars.high-bars.low), (bars.high-prev_close).abs(),
                    (bars.low-prev_close).abs()], axis=1).max(axis=1)
    bars['atr14'] = tr.ewm(alpha=1/14, adjust=False).mean()
    return bars

def run(bars, capital, risk_pct, atr_mult, reward, contract_size, spread, slippage,
        commission_lot_side, max_leverage, min_lot, lot_step):
    cash = capital
    positions = []
    trades, signals, equity_rows, daily_rows = [], [], [], []
    next_id = 1
    prev_session = None
    day_start_equity = capital
    day_low = capital
    daily_breach = False
    total_breach = False
    prev_ema = prev_vwap = np.nan
    pending = []
    # Signals from candle t are executed at OPEN of candle t+1, then stop/target checks.
    for timestamp, b in bars.iterrows():
        t = timestamp.isoformat()
        session = str(b.session)
        if session != prev_session:
            if prev_session is not None:
                daily_rows.append({'session':prev_session,'start_equity':day_start_equity,
                                   'end_equity':last_equity,'min_equity':day_low,
                                   'daily_5pct_breach_approx':daily_breach})
            prev_session = session
            day_start_equity = last_equity if equity_rows else capital
            day_low = day_start_equity
            daily_breach = False
        # Pending orders entered at bar OPEN. Available margin may prevent entry.
        for order in pending:
            side = order['side']
            fill = float(b.open) + (spread/2+slippage)*(1 if side == 1 else -1)
            dist = order['distance']
            if fill <= 0 or dist <= 0: continue
            # Broker-specific volume: lots * contract_size ounces. Round down to lot step.
            raw_lots = (order['risk_cash'] / (dist*contract_size))
            lots = np.floor((raw_lots + 1e-12) / lot_step) * lot_step
            lots = round(float(lots), 8)
            notional = lots * contract_size * fill
            used_margin = sum(p['lots']*contract_size*float(b.open)/max_leverage for p in positions)
            unreal = sum(p['side']*(float(b.open)-p['entry'])*p['lots']*contract_size for p in positions)
            available = max(0, cash+unreal-used_margin)
            if lots < min_lot or notional/max_leverage > available:
                signals[order['signal_index']]['status'] = 'SKIPPED_MARGIN_OR_MIN_LOT'
                continue
            stop = fill - side*dist
            target = fill + side*dist*reward
            p = {'id':next_id,'side':side,'entry':fill,'entry_time':t,'stop':stop,
                 'target':target,'lots':lots,'signal_time':order['signal_time'],
                 'entry_commission':commission_lot_side*lots}
            next_id += 1
            cash -= p['entry_commission']
            positions.append(p)
            signals[order['signal_index']]['status'] = 'FILLED'
            signals[order['signal_index']]['trade_id'] = p['id']
        pending = []
        survivors = []
        for p in positions:
            side = p['side']
            # Conservative tie-break: if stop and target touched within one candle, stop wins.
            stop_hit = (b.low <= p['stop']) if side == 1 else (b.high >= p['stop'])
            target_hit = (b.high >= p['target']) if side == 1 else (b.low <= p['target'])
            if stop_hit or target_hit:
                chosen = p['stop'] if stop_hit else p['target']
                # Gap through stop: assume adverse open fill; gap through target: no better than target.
                if stop_hit:
                    chosen = min(chosen,float(b.open)) if side == 1 else max(chosen,float(b.open))
                exit_fill = chosen - side*(spread/2+slippage)
                gross = side*(exit_fill-p['entry'])*p['lots']*contract_size
                exit_fee = commission_lot_side*p['lots']
                cash += gross-exit_fee
                trades.append({'trade_id':p['id'],'signal_time_utc':p['signal_time'],
                               'entry_time_utc':p['entry_time'],'exit_time_utc':t,
                               'direction':'LONG' if side == 1 else 'SHORT',
                               'lots':p['lots'],'entry':p['entry'],'stop':p['stop'],
                               'target':p['target'],'exit':exit_fill,'reason':'STOP' if stop_hit else 'TARGET',
                               'gross_pnl':gross,'net_pnl':gross-exit_fee-p['entry_commission'],
                               'commission_total':exit_fee+p['entry_commission']})
            else: survivors.append(p)
        positions = survivors
        unreal = sum(p['side']*(float(b.close)-p['entry'])*p['lots']*contract_size for p in positions)
        equity = cash + unreal
        day_low = min(day_low,equity)
        # Approximate only: FTMO reset / floating P&L accounting requires tick-level data.
        if equity <= day_start_equity - capital*.05: daily_breach = True
        if equity <= capital*.90: total_breach = True
        equity_rows.append({'time_utc':t,'session':session,'cash':cash,'unrealized_pnl':unreal,
                            'equity':equity,'open_positions':len(positions),
                            'ftmo_daily_breach_approx':daily_breach,
                            'ftmo_total_breach_approx':total_breach})
        last_equity = equity
        if pd.notna(prev_ema) and pd.notna(prev_vwap) and pd.notna(b.vwap):
            cross_up = prev_ema <= prev_vwap and b.ema9 > b.vwap
            cross_down = prev_ema >= prev_vwap and b.ema9 < b.vwap
            if cross_up or cross_down:
                side = 1 if cross_up else -1
                sig = {'signal_time_utc':t,'direction':'LONG' if side == 1 else 'SHORT',
                       'ema9':b.ema9,'vwap':b.vwap,'atr14':b.atr14,
                       'open_positions_at_signal':len(positions),
                       'status':'PENDING_NEXT_BAR','trade_id':''}
                signals.append(sig)
                pending.append({'side':side,'distance':float(b.atr14)*atr_mult,
                                'risk_cash':max(0,equity)*risk_pct/100,
                                'signal_time':t,'signal_index':len(signals)-1})
        prev_ema,prev_vwap = b.ema9,b.vwap
    for order in pending:
        signals[order['signal_index']]['status'] = 'NO_NEXT_BAR'
    daily_rows.append({'session':prev_session,'start_equity':day_start_equity,
                       'end_equity':last_equity,'min_equity':day_low,
                       'daily_5pct_breach_approx':daily_breach})
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(trades,columns=['trade_id','signal_time_utc','entry_time_utc','exit_time_utc','direction','lots','entry','stop','target','exit','reason','gross_pnl','net_pnl','commission_total']).to_csv(OUT/'trades.csv',index=False)
    pd.DataFrame(signals,columns=['signal_time_utc','direction','ema9','vwap','atr14','open_positions_at_signal','status','trade_id']).to_csv(OUT/'signals.csv',index=False)
    pd.DataFrame(equity_rows).to_csv(OUT/'equity_curve.csv',index=False)
    pd.DataFrame(daily_rows).to_csv(OUT/'daily_equity.csv',index=False)
    closed = pd.DataFrame(trades)
    wins = int((closed.net_pnl > 0).sum()) if not closed.empty else 0
    gross_win = float(closed.loc[closed.net_pnl>0,'net_pnl'].sum()) if not closed.empty else 0
    gross_loss = float(-closed.loc[closed.net_pnl<0,'net_pnl'].sum()) if not closed.empty else 0
    curve = pd.Series([x['equity'] for x in equity_rows])
    dd = curve / curve.cummax().clip(lower=1e-12) - 1
    summary = {'initial_capital':capital,'final_equity_including_open_pnl':last_equity,
               'return_pct':100*(last_equity/capital-1),'closed_trades':len(trades),
               'open_positions_at_end':len(positions),'signals':len(signals),
               'filled_signals':sum(s['status']=='FILLED' for s in signals),
               'skipped_margin_or_lot':sum(s['status']=='SKIPPED_MARGIN_OR_MIN_LOT' for s in signals),
               'win_rate_pct':100*wins/len(trades) if trades else 0,
               'profit_factor':gross_win/gross_loss if gross_loss else None,
               'max_mark_to_market_drawdown_pct':100*float(dd.min()),
               'approx_ftmo_daily_breach_any':any(x['daily_5pct_breach_approx'] for x in daily_rows),
               'approx_ftmo_total_breach_any':total_breach,
               'note':'No forced liquidation at dataset end; unrealized P&L included in final equity.'}
    pd.DataFrame([summary]).to_csv(OUT/'summary.csv',index=False)
    print(pd.Series(summary).to_string())

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv',default='data/xauusd_1m.csv')
    ap.add_argument('--timezone',default='UTC',help='Timezone for naive timestamps; e.g. UTC or Europe/Prague')
    ap.add_argument('--capital',type=float,default=50000)
    ap.add_argument('--risk-pct',type=float,default=.5)
    ap.add_argument('--atr-mult',type=float,default=1.5)
    ap.add_argument('--reward',type=float,default=4)
    ap.add_argument('--contract-size',type=float,default=100,help='Ounces per standard lot; verify broker specification')
    ap.add_argument('--spread',type=float,default=.30,help='Total spread in XAUUSD price units, placeholder')
    ap.add_argument('--slippage',type=float,default=.05,help='Additional adverse price units per side, placeholder')
    ap.add_argument('--commission-lot-side',type=float,default=3.5,help='USD per lot per side, placeholder')
    ap.add_argument('--max-leverage',type=float,default=30,help='Margin assumption, not a strategy guard')
    ap.add_argument('--min-lot',type=float,default=.01)
    ap.add_argument('--lot-step',type=float,default=.01)
    a=ap.parse_args()
    if a.risk_pct <= 0 or a.reward <= 0 or a.atr_mult <= 0 or a.max_leverage <= 0:
        ap.error('Risk, reward, ATR multiplier and leverage must be positive')
    raw=load_bars(a.csv,a.timezone)
    bars=prepare(raw)
    print(f'Loaded {len(raw)} 1-minute candles, constructed {len(bars)} complete 3-minute candles')
    run(bars,a.capital,a.risk_pct,a.atr_mult,a.reward,a.contract_size,
        a.spread,a.slippage,a.commission_lot_side,a.max_leverage,a.min_lot,a.lot_step)

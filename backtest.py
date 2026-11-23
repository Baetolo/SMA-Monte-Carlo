import argparse
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
LONDON = ZoneInfo('Europe/London')
NEW_YORK = ZoneInfo('America/New_York')


def in_session(ts):
    ts = pd.Timestamp(ts)
    ts = ts.tz_localize('UTC') if ts.tzinfo is None else ts.tz_convert('UTC')
    return (time(8) <= ts.tz_convert(LONDON).time() < time(17) or
            time(8) <= ts.tz_convert(NEW_YORK).time() < time(17))


def load_bars(path):
    df = pd.read_csv(path)
    time_col = next((c for c in ('time_utc', 'timestamp', 'datetime', 'time') if c in df), None)
    if time_col is None:
        raise ValueError('Missing UTC timestamp column')
    df.index = pd.to_datetime(df[time_col], utc=True)
    df = df.sort_index()
    if df.index.has_duplicates:
        raise ValueError('Duplicate timestamps in input data')
    for c in ('open', 'high', 'low', 'close'):
        df[c] = pd.to_numeric(df[c], errors='raise')
    if 'volume' not in df:
        df['volume'] = 1.0
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
    agg = df.resample('3min', label='left', closed='left').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    counts = df['close'].resample('3min').count()
    bars = agg.loc[counts.eq(3)].dropna().copy()
    if bars.empty:
        raise ValueError('No complete three-minute candles')
    bars['ema9'] = bars.close.ewm(span=9, adjust=False).mean()
    bars['ema200'] = bars.close.ewm(span=200, adjust=False, min_periods=200).mean()
    prior = bars.close.shift()
    tr = pd.concat([bars.high - bars.low, (bars.high-prior).abs(),
                    (bars.low-prior).abs()], axis=1).max(axis=1)
    bars['atr'] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    return bars


def backtest(bars, capital, risk_pct, reward, atr_mult, buffer, commission, max_lots, output):
    cash = float(capital)
    open_positions, trades, signals, curve = [], [], [], []
    next_id = 0
    peak = cash
    max_dd = 0.0
    daily_breach = total_breach = False
    day_start_equity = cash
    current_day = None
    rejected_session = rejected_trend = rejected_size = 0

    def mark(price):
        return sum((price-p['entry'])*p['side']*p['lots']*100 for p in open_positions)

    for i in range(2, len(bars)):
        ts = bars.index[i]
        bar, sig, pull = bars.iloc[i], bars.iloc[i-1], bars.iloc[i-2]
        open_price = float(bar.open)
        if ts.date() != current_day:
            current_day = ts.date()
            day_start_equity = cash + mark(open_price)

        # Exit existing positions; if both stop and target touch, stop takes priority.
        remaining = []
        for p in open_positions:
            exit_price = None
            reason = None
            if p['side'] == 1:
                if bar.low <= p['stop']:
                    exit_price, reason = min(float(bar.open), p['stop'])-buffer, 'STOP'
                elif bar.high >= p['target']:
                    exit_price, reason = p['target'], 'TARGET'
            else:
                if bar.high >= p['stop']:
                    exit_price, reason = max(float(bar.open), p['stop'])+buffer, 'STOP'
                elif bar.low <= p['target']:
                    exit_price, reason = p['target'], 'TARGET'
            if exit_price is None:
                remaining.append(p)
                continue
            gross = (exit_price-p['entry'])*p['side']*p['lots']*100
            exit_fee = p['lots']*commission
            cash += gross-exit_fee
            trades.append(dict(trade_id=p['id'], signal_time_utc=p['signal_time'],
                entry_time_utc=p['entry_time'], exit_time_utc=ts,
                direction='LONG' if p['side']==1 else 'SHORT', lots=p['lots'],
                entry=p['entry'], stop=p['stop'], target=p['target'], exit=exit_price,
                reason=reason, gross_pnl=gross, net_pnl=gross-exit_fee-p['entry_fee'],
                commission_total=exit_fee+p['entry_fee']))
        open_positions = remaining

        # Pullback candle touches EMA9; next candle confirms in trend direction.
        long_signal = (pd.notna(sig.ema200) and
            pull.low <= pull.ema9 and pull.close <= pull.ema9 and
            sig.close > sig.open and sig.close > sig.ema9 and
            sig.close > sig.ema200)
        short_signal = (pd.notna(sig.ema200) and
            pull.high >= pull.ema9 and pull.close >= pull.ema9 and
            sig.close < sig.open and sig.close < sig.ema9 and
            sig.close < sig.ema200)
        side = 1 if long_signal else (-1 if short_signal else 0)
        if side:
            allowed = in_session(bars.index[i-1])
            signals.append(dict(signal_time_utc=bars.index[i-1],
                direction='LONG' if side==1 else 'SHORT', session_allowed=allowed))
            if not allowed:
                rejected_session += 1
            elif pd.notna(sig.atr) and sig.atr > 0:
                distance = float(sig.atr)*atr_mult
                entry = open_price+(buffer if side==1 else 0.0)
                equity = cash+mark(open_price)
                lots = min(np.floor((max(0,equity)*risk_pct/100)/(distance*100)*100)/100,
                           max_lots)
                existing_notional = sum(p['lots']*100*open_price for p in open_positions)
                if lots >= 0.01 and cash > 0 and existing_notional+lots*100*entry <= max(equity,0)*100:
                    next_id += 1
                    fee = lots*commission
                    cash -= fee
                    open_positions.append(dict(id=next_id, signal_time=bars.index[i-1],
                        entry_time=ts, side=side, entry=entry,
                        stop=entry-side*distance, target=entry+side*distance*reward,
                        lots=lots, entry_fee=fee))
                else:
                    rejected_size += 1

        equity = cash+mark(float(bar.close))
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak-equity)/peak*100 if peak > 0 else 0)
        # Approximation only, not official FTMO compliance calculation.
        daily_breach |= day_start_equity-equity > capital*0.05
        total_breach |= equity < capital*0.90
        curve.append(dict(time_utc=ts, session=str(ts.date()), cash=cash,
            unrealized_pnl=equity-cash, equity=equity,
            open_positions=len(open_positions),
            ftmo_daily_breach_approx=daily_breach,
            ftmo_total_breach_approx=total_breach))

    t = pd.DataFrame(trades, columns=['trade_id','signal_time_utc','entry_time_utc',
        'exit_time_utc','direction','lots','entry','stop','target','exit','reason',
        'gross_pnl','net_pnl','commission_total'])
    e = pd.DataFrame(curve)
    s = pd.DataFrame(signals, columns=['signal_time_utc','direction','session_allowed'])
    net = t.net_pnl if not t.empty else pd.Series(dtype=float)
    winners = net[net > 0].sum()
    losers = -net[net < 0].sum()
    final = float(e.equity.iloc[-1]) if not e.empty else cash
    summary = dict(initial_capital=capital, final_equity_including_open_pnl=final,
        return_pct=(final/capital-1)*100, closed_trades=len(t),
        open_positions_at_end=len(open_positions), signals=len(s),
        filled_signals=next_id, skipped_margin_or_lot=rejected_size,
        session_rejected=rejected_session, trend_rejected=rejected_trend,
        win_rate_pct=(net.gt(0).mean()*100 if len(net) else 0),
        profit_factor=(winners/losers if losers else np.nan),
        max_mark_to_market_drawdown_pct=max_dd,
        approx_ftmo_daily_breach_any=daily_breach,
        approx_ftmo_total_breach_any=total_breach,
        risk_pct=risk_pct, reward_multiple=reward,
        session_filter='London + New York', trend_filter='EMA 200',
        entry_rule='EMA9 two-candle pullback confirmation')
    output.mkdir(parents=True, exist_ok=True)
    t.to_csv(output/'trades.csv', index=False)
    e.to_csv(output/'equity_curve.csv', index=False)
    s.to_csv(output/'signals.csv', index=False)
    if not e.empty:
        e.groupby('session', as_index=False).tail(1).to_csv(output/'daily_equity.csv', index=False)
    pd.DataFrame([summary]).to_csv(output/'summary.csv', index=False)
    print(pd.Series(summary).to_string())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='data/xauusd_1m.csv')
    p.add_argument('--capital', type=float, default=50000)
    p.add_argument('--risk-pct', type=float, default=0.5)
    p.add_argument('--reward', type=float, default=2.0)
    p.add_argument('--atr-multiple', type=float, default=2.0)
    p.add_argument('--execution-buffer', type=float, default=0.20)
    p.add_argument('--commission-per-lot', type=float, default=3.5)
    p.add_argument('--max-lots', type=float, default=100)
    a = p.parse_args()
    path = Path(a.csv)
    if not path.is_absolute():
        path = Path.cwd()/path
    backtest(load_bars(path), a.capital, a.risk_pct, a.reward,
             a.atr_multiple, a.execution_buffer, a.commission_per_lot,
             a.max_lots, Path.cwd()/'outputs')

if __name__ == '__main__':
    main()

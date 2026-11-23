
import argparse
from pathlib import Path
from datetime import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"

LONDON = ZoneInfo("Europe/London")
NEW_YORK = ZoneInfo("America/New_York")


def in_session(timestamp):
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    london_time = ts.tz_convert(LONDON).time()
    new_york_time = ts.tz_convert(NEW_YORK).time()

    return (
        time(8, 0) <= london_time < time(17, 0)
        or time(8, 0) <= new_york_time < time(17, 0)
    )


def load_data(path):
    df = pd.read_csv(path)

    time_col = next(
        (c for c in ["time_utc", "timestamp", "datetime", "time"]
         if c in df.columns),
        None
    )
    if time_col is None:
        raise ValueError("CSV needs a timestamp column.")

    df["time_utc"] = pd.to_datetime(df[time_col], utc=True)
    df = df.set_index("time_utc").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])

    if "volume" not in df.columns:
        df["volume"] = 1.0
    df["volume"] = pd.to_numeric(
        df["volume"], errors="coerce"
    ).fillna(0)

    bars = df.resample(
        "3min", label="left", closed="left"
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    })

    counts = df["close"].resample("3min").count()
    bars = bars[counts == 3].copy()
    bars = bars.dropna()

    return bars


def indicators(df):
    df = df.copy()

    df["ema9"] = df["close"].ewm(
        span=9, adjust=False
    ).mean()

    previous_close = df["close"].shift(1)

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - previous_close).abs(),
        (df["low"] - previous_close).abs()
    ], axis=1).max(axis=1)

    df["atr"] = tr.ewm(
        alpha=1 / 14, adjust=False,
        min_periods=14
    ).mean()

    # Reset VWAP at each UTC calendar day.
    session = df.index.normalize()

    typical = (
        df["high"] + df["low"] + df["close"]
    ) / 3

    volume = df["volume"].clip(lower=0)

    cum_pv = (typical * volume).groupby(
        session
    ).cumsum()
    cum_vol = volume.groupby(session).cumsum()

    df["vwap"] = cum_pv / cum_vol.replace(0, np.nan)

    # EMA crossing VWAP on completed candles.
    previous_diff = (
        df["ema9"] - df["vwap"]
    ).shift(1)
    current_diff = df["ema9"] - df["vwap"]

    df["long_signal"] = (
        (previous_diff <= 0) &
        (current_diff > 0)
    )

    df["short_signal"] = (
        (previous_diff >= 0) &
        (current_diff < 0)
    )

    return df


def run_backtest(
    df,
    initial_capital,
    risk_pct,
    reward,
    atr_multiple,
    execution_buffer,
    commission_per_lot,
    max_lots
):
    cash = float(initial_capital)
    positions = []
    trades = []
    signals = []
    equity_rows = []

    trade_id = 0
    skipped = 0
    previous_day = None
    day_start_equity = initial_capital
    peak_equity = initial_capital
    max_drawdown = 0.0
    daily_breach = False
    total_breach = False

    def unrealized(close):
        total = 0.0
        for p in positions:
            direction = p["side"]
            total += (
                (close - p["entry"])
                * direction * p["lots"] * 100
            )
        return total

    def close_trade(p, exit_price, exit_time, reason):
        nonlocal cash

        gross = (
            (exit_price - p["entry"])
            * p["side"] * p["lots"] * 100
        )
        exit_commission = (
            p["lots"] * commission_per_lot
        )

        cash += gross - exit_commission

        trades.append({
            "trade_id": p["id"],
            "signal_time_utc": p["signal_time"],
            "entry_time_utc": p["entry_time"],
            "exit_time_utc": exit_time,
            "direction": (
                "LONG" if p["side"] == 1 else "SHORT"
            ),
            "lots": p["lots"],
            "entry": p["entry"],
            "stop": p["stop"],
            "target": p["target"],
            "exit": exit_price,
            "reason": reason,
            "gross_pnl": gross,
            "net_pnl": (
                gross - exit_commission
                - p["entry_commission"]
            ),
            "commission_total": (
                exit_commission + p["entry_commission"]
            )
        })

    # Previous completed candle generates the signal;
    # current candle opens the position.
    for i in range(1, len(df)):
        bar = df.iloc[i]
        prev = df.iloc[i - 1]
        timestamp = df.index[i]
        signal_time = df.index[i - 1]

        day = timestamp.date()

        if previous_day != day:
            day_start_equity = cash + unrealized(
                float(bar["open"])
            )
            previous_day = day

        # First manage existing positions.
        # Conservative assumption: stop wins when
        # both stop and target occur in one candle.
        survivors = []

        for p in positions:
            exit_price = None
            reason = None

            if p["side"] == 1:
                if bar["low"] <= p["stop"]:
                    exit_price = min(
                        p["stop"],
                        float(bar["open"])
                    ) - execution_buffer
                    reason = "STOP"
                elif bar["high"] >= p["target"]:
                    exit_price = p["target"]
                    reason = "TARGET"
            else:
                if bar["high"] >= p["stop"]:
                    exit_price = max(
                        p["stop"],
                        float(bar["open"])
                    ) + execution_buffer
                    reason = "STOP"
                elif bar["low"] <= p["target"]:
                    exit_price = p["target"]
                    reason = "TARGET"

            if exit_price is None:
                survivors.append(p)
            else:
                close_trade(
                    p, exit_price, timestamp, reason
                )

        positions = survivors

        side = 0
        if bool(prev["long_signal"]):
            side = 1
        elif bool(prev["short_signal"]):
            side = -1

        if side:
            allowed = in_session(signal_time)
            signals.append({
                "signal_time_utc": signal_time,
                "direction": (
                    "LONG" if side == 1 else "SHORT"
                ),
                "session_allowed": allowed
            })

            if allowed and np.isfinite(prev["atr"]):
                atr = float(prev["atr"])
                stop_distance = atr * atr_multiple

                if stop_distance > 0:
                    # Buy at ask approximation,
                    # sell at bid approximation.
                    entry = float(bar["open"])
                    if side == 1:
                        entry += execution_buffer

                    equity_before = cash + unrealized(
                        float(bar["open"])
                    )
                    risk_dollars = max(
                        0, equity_before * risk_pct / 100
                    )

                    # 100 oz per standard XAUUSD lot.
                    lots = np.floor(
                        risk_dollars /
                        (stop_distance * 100) * 100
                    ) / 100

                    lots = min(lots, max_lots)

                    if lots >= 0.01 and cash > 0:
                        stop = (
                            entry - side * stop_distance
                        )
                        target = (
                            entry + side
                            * stop_distance * reward
                        )

                        entry_commission = (
                            lots * commission_per_lot
                        )

                        # Simplified margin check:
                        # do not allow notional exposure
                        # beyond 100x positive equity.
                        existing_notional = sum(
                            p["lots"] * 100
                            * float(bar["open"])
                            for p in positions
                        )
                        new_notional = (
                            lots * 100 * entry
                        )

                        if (
                            existing_notional + new_notional
                            <= max(equity_before, 0) * 100
                        ):
                            trade_id += 1
                            cash -= entry_commission

                            positions.append({
                                "id": trade_id,
                                "signal_time": signal_time,
                                "entry_time": timestamp,
                                "side": side,
                                "entry": entry,
                                "stop": stop,
                                "target": target,
                                "lots": lots,
                                "entry_commission": (
                                    entry_commission
                                )
                            })
                        else:
                            skipped += 1
                    else:
                        skipped += 1

        current_equity = (
            cash + unrealized(float(bar["close"]))
        )
        peak_equity = max(peak_equity, current_equity)

        if peak_equity > 0:
            drawdown = (
                peak_equity - current_equity
            ) / peak_equity * 100
            max_drawdown = max(
                max_drawdown, drawdown
            )

        # Approximate FTMO limits only.
        day_loss = (
            day_start_equity - current_equity
        )
        if day_loss > initial_capital * 0.05:
            daily_breach = True

        if current_equity < initial_capital * 0.90:
            total_breach = True

        equity_rows.append({
            "time_utc": timestamp,
            "session": str(day),
            "cash": cash,
            "unrealized_pnl": (
                current_equity - cash
            ),
            "equity": current_equity,
            "open_positions": len(positions),
            "ftmo_daily_breach_approx": (
                daily_breach
            ),
            "ftmo_total_breach_approx": (
                total_breach
            )
        })

    final_equity = cash
    if len(df) and positions:
        final_equity += unrealized(
            float(df["close"].iloc[-1])
        )

    trade_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    signal_df = pd.DataFrame(signals)

    if not trade_df.empty:
        wins = (trade_df["net_pnl"] > 0).sum()
        gross_wins = trade_df.loc[
            trade_df["net_pnl"] > 0, "net_pnl"
        ].sum()
        gross_losses = -trade_df.loc[
            trade_df["net_pnl"] < 0, "net_pnl"
        ].sum()

        win_rate = wins / len(trade_df) * 100
        profit_factor = (
            gross_wins / gross_losses
            if gross_losses > 0 else np.nan
        )
    else:
        win_rate = 0
        profit_factor = 0

    summary = {
        "initial_capital": initial_capital,
        "final_equity_including_open_pnl": final_equity,
        "return_pct": (
            (final_equity / initial_capital - 1) * 100
        ),
        "closed_trades": len(trade_df),
        "open_positions_at_end": len(positions),
        "signals": len(signal_df),
        "filled_signals": trade_id,
        "skipped_margin_or_lot": skipped,
        "win_rate_pct": win_rate,
        "profit_factor": profit_factor,
        "max_mark_to_market_drawdown_pct": max_drawdown,
        "approx_ftmo_daily_breach_any": daily_breach,
        "approx_ftmo_total_breach_any": total_breach,
        "risk_pct": risk_pct,
        "reward_multiple": reward,
        "session_filter": "London + New York"
    }

    OUTPUT.mkdir(exist_ok=True, parents=True)

    trade_df.to_csv(
        OUTPUT / "trades.csv", index=False
    )
    equity_df.to_csv(
        OUTPUT / "equity_curve.csv", index=False
    )
    signal_df.to_csv(
        OUTPUT / "signals.csv", index=False
    )

    if not equity_df.empty:
        daily = (
            equity_df.groupby("session", as_index=False)
            .tail(1)
            .copy()
        )
        daily.to_csv(
            OUTPUT / "daily_equity.csv",
            index=False
        )

    pd.DataFrame([summary]).to_csv(
        OUTPUT / "summary.csv", index=False
    )

    print("\n=== GOLD BACKTEST RESULTS ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nFiles saved in outputs/")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        default="data/xauusd_1m.csv"
    )
    parser.add_argument(
        "--capital", type=float, default=50000
    )
    parser.add_argument(
        "--risk-pct", type=float, default=0.5
    )
    parser.add_argument(
        "--reward", type=float, default=2.0
    )
    parser.add_argument(
        "--atr-multiple", type=float, default=2.0
    )
    parser.add_argument(
        "--execution-buffer", type=float,
        default=0.20
    )
    parser.add_argument(
        "--commission-per-lot", type=float,
        default=3.5
    )
    parser.add_argument(
        "--max-lots", type=float, default=100.0
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path

    bars = load_data(csv_path)
    bars = indicators(bars)

    print(f"Loaded {len(bars):,} three-minute bars")

    run_backtest(
        bars,
        args.capital,
        args.risk_pct,
        args.reward,
        args.atr_multiple,
        args.execution_buffer,
        args.commission_per_lot,
        args.max_lots
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download Dukascopy XAUUSD BID M1 daily .bi5 files and build backtester CSV.
Research use only. Check Dukascopy's terms before bulk downloading.
"""
import argparse
import csv
import datetime as dt
import io
import lzma
import struct
import time
import urllib.error
import urllib.request
from pathlib import Path

RECORD = struct.Struct('>IIIIIf')  # seconds, open, close, low, high, volume
HEADERS = ['timestamp','open','high','low','close','volume']


def fetch_day(day, attempts=4):
    month_zero = day.month - 1
    url = f'https://datafeed.dukascopy.com/datafeed/XAUUSD/{day.year}/{month_zero:02d}/{day.day:02d}/BID_candles_min_1.bi5'
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, headers={'User-Agent':'GoldHistoricalResearch/1.0'})
            with urllib.request.urlopen(req, timeout=35) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if exc.code not in (429, 500, 502, 503, 504) or attempt == attempts-1:
                raise
        except (TimeoutError, urllib.error.URLError):
            if attempt == attempts-1:
                raise
        time.sleep(min(30, 3 * 2 ** attempt))
    raise RuntimeError(f'Could not fetch {day}')


def decode(data, day):
    payload = lzma.decompress(data, format=lzma.FORMAT_AUTO)
    if len(payload) % RECORD.size:
        raise ValueError(f'{day}: unexpected decompressed length {len(payload)}')
    result = []
    for seconds, op, cl, lo, hi, volume in RECORD.iter_unpack(payload):
        if seconds >= 86400 or seconds % 60 != 0:
            raise ValueError(f'{day}: invalid minute offset {seconds}')
        if not (lo <= op <= hi and lo <= cl <= hi):
            raise ValueError(f'{day}: malformed OHLC record')
        stamp = dt.datetime.combine(day, dt.time(), dt.timezone.utc) + dt.timedelta(seconds=seconds)
        result.append([stamp.isoformat(),op/1000,hi/1000,lo/1000,cl/1000,round(float(volume),5)])
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--from-date', default='2026-01-01')
    p.add_argument('--to-date', default='2026-02-01', help='Exclusive end date')
    p.add_argument('--output', default='data/xauusd_1m.csv')
    p.add_argument('--pause', type=float, default=0.35)
    a = p.parse_args()
    start, end = dt.date.fromisoformat(a.from_date), dt.date.fromisoformat(a.to_date)
    if not start < end <= dt.datetime.now(dt.timezone.utc).date() + dt.timedelta(days=1):
        p.error('Date range must be nonempty and cannot extend into the future')
    if (end-start).days > 370:
        p.error('Maximum 370 calendar days per run; download longer periods in batches')
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    missing = []
    with out.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADERS)
        day = start
        while day < end:
            if day.weekday() < 5:
                try:
                    content = fetch_day(day)
                    if content is None:
                        missing.append((str(day),'404/no session'))
                    else:
                        rows = decode(content, day)
                        writer.writerows(rows)
                        count += len(rows)
                        print(f'{day}: {len(rows)} candles', flush=True)
                except Exception as exc:
                    missing.append((str(day),repr(exc)))
                    print(f'ERROR {day}: {exc}', flush=True)
            day += dt.timedelta(days=1)
            time.sleep(max(0,a.pause))
    print(f'Wrote {count} rows to {out}; unavailable days: {len(missing)}')
    if missing:
        log = out.with_name('download_issues.csv')
        with log.open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['date','reason'])
            w.writerows(missing)
        print(f'Check {log} for missing days')
    if count < 100:
        raise RuntimeError('Too few bars downloaded; check errors and provider access')
    serious = [x for x in missing if x[1] != '404/no session']
    if serious:
        raise RuntimeError(f'{len(serious)} days failed due to download/decode errors; do not backtest incomplete data')

if __name__ == '__main__':
    main()

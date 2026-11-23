
#!/usr/bin/env python3
"""Resumable Dukascopy XAUUSD M1 BID downloader."""

import argparse
import csv
import datetime as dt
import lzma
import random
import struct
import time
import urllib.error
import urllib.request
from pathlib import Path

RECORD = struct.Struct(">IIIIIf")
HEADERS = ["timestamp", "open", "high", "low", "close", "volume"]

CACHE = Path("data/gold_cache")
CACHE.mkdir(parents=True, exist_ok=True)


def fetch_day(day, attempts=8):
    month_zero = day.month - 1
    url = (
        f"https://datafeed.dukascopy.com/datafeed/XAUUSD/"
        f"{day.year}/{month_zero:02d}/{day.day:02d}/"
        "BID_candles_min_1.bi5"
    )

    for attempt in range(attempts):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "*/*",
                },
            )

            with urllib.request.urlopen(req, timeout=60) as response:
                return response.read()

        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None

            if exc.code not in (429, 500, 502, 503, 504):
                raise

            error = exc

        except (TimeoutError, urllib.error.URLError) as exc:
            error = exc

        if attempt < attempts - 1:
            delay = min(90, 3 * (2 ** attempt))
            delay += random.uniform(0, 2)
            print(
                f"{day}: retry {attempt + 1}/{attempts} "
                f"after {delay:.1f}s ({error})",
                flush=True,
            )
            time.sleep(delay)

    raise RuntimeError(f"{day}: download failed after {attempts} attempts")


def decode(data, day):
    payload = lzma.decompress(data, format=lzma.FORMAT_AUTO)

    if len(payload) % RECORD.size:
        raise ValueError(f"{day}: invalid data length")

    rows = []

    for seconds, op, cl, lo, hi, volume in RECORD.iter_unpack(payload):
        if seconds >= 86400 or seconds % 60:
            raise ValueError(f"{day}: invalid minute timestamp")

        if not (lo <= op <= hi and lo <= cl <= hi):
            raise ValueError(f"{day}: invalid OHLC values")

        stamp = (
            dt.datetime.combine(
                day, dt.time(), dt.timezone.utc
            )
            + dt.timedelta(seconds=seconds)
        )

        rows.append([
            stamp.isoformat(),
            op / 1000,
            hi / 1000,
            lo / 1000,
            cl / 1000,
            round(float(volume), 5),
        ])

    return rows


def cache_path(day):
    return CACHE / f"{day.isoformat()}.csv"


def read_cached_day(day):
    path = cache_path(day)

    if not path.exists():
        return None

    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)

        if header != HEADERS:
            raise ValueError(f"{path}: invalid cached header")

        rows = list(reader)

    if not rows:
        raise ValueError(f"{path}: empty cached file")

    return rows


def save_day(day, rows):
    path = cache_path(day)
    temp = path.with_suffix(".tmp")

    with temp.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADERS)
        writer.writerows(rows)

    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--from-date", default="2026-01-01")
    parser.add_argument("--to-date", default="2026-02-01")
    parser.add_argument("--output", default="data/xauusd_1m.csv")
    parser.add_argument("--pause", type=float, default=1.0)

    args = parser.parse_args()

    start = dt.date.fromisoformat(args.from_date)
    end = dt.date.fromisoformat(args.to_date)

    if not start < end:
        parser.error("End date must be after start date")

    if end > dt.datetime.now(dt.timezone.utc).date() + dt.timedelta(days=1):
        parser.error("Cannot download future data")

    if (end - start).days > 370:
        parser.error("Maximum 370 days per run")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    missing = []
    all_rows = []
    downloaded = 0
    cached = 0

    day = start

    while day < end:
        if day.weekday() >= 5:
            day += dt.timedelta(days=1)
            continue

        try:
            rows = read_cached_day(day)

            if rows is not None:
                cached += 1
                print(f"{day}: loaded from cache", flush=True)

            else:
                content = fetch_day(day)

                if content is None:
                    missing.append((str(day), "404/no session"))
                    day += dt.timedelta(days=1)
                    continue

                rows = decode(content, day)

                if not rows:
                    missing.append((str(day), "empty session"))
                    day += dt.timedelta(days=1)
                    continue

                save_day(day, rows)
                downloaded += 1

                print(
                    f"{day}: downloaded {len(rows)} candles",
                    flush=True,
                )

                time.sleep(max(0, args.pause))

            all_rows.extend(rows)

        except Exception as exc:
            missing.append((str(day), str(exc)))
            print(f"ERROR {day}: {exc}", flush=True)

        day += dt.timedelta(days=1)

    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADERS)
        writer.writerows(all_rows)

    issues = output.with_name("download_issues.csv")

    with issues.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "reason"])
        writer.writerows(missing)

    print(f"Total candles: {len(all_rows)}")
    print(f"New days downloaded: {downloaded}")
    print(f"Days restored from cache: {cached}")
    print(f"Unavailable days: {len(missing)}")

    if len(all_rows) < 100:
        raise RuntimeError("Insufficient historical data")

    if missing:
        raise RuntimeError(
            f"{len(missing)} days unavailable. "
            "Backtest stopped to avoid incomplete data."
        )

    print("Historical download completed successfully.")


if __name__ == "__main__":
    main()

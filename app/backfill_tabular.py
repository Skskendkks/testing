"""P1 (revised): backfill snapshots.csv from NOAA ISD-Lite hourly obs + HKO warndb labels.

Why not the data.gov.hk historical archive (original v4 plan): verified 2026-08 that
rhrread/warnsum JSON are NOT archived there (0 versions in 2021/2023/2025 probes);
the archived RSS warning feed only gets ~1 snapshot/day (even during Signal 10),
so warning timelines cannot be reconstructed from it. What IS available:

  * NOAA ISD-Lite hourly observations for Hong Kong Intl Airport, station
    450070-99999 (verified): temp, dewpoint (-> RH), pressure, wind, 1h precip,
    hourly back to the 1990s.
    https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{year}/450070-99999-{year}.gz
  * HKO Warnings/Signals Database (warndb) for official warning issue/cancel
    times (Rainstorm since 1998, TC since 1946) -> data/warning_events.csv.
  * F3 gridded nowcast archive (verified from ~2023) — handled by backfill.py.

Usage (one-off workflow_dispatch, not recurring):

    python app/backfill_tabular.py 2023 2025            # inclusive year range
    python app/backfill_tabular.py 2024 2024 --rain-scale 15

Requires data/warning_events.csv with columns: type,start,end
  type  in {AMBER, RED, BLACK, TC1, TC3, TC8, TC9, TC10, WTS, WMSGNL}
  start/end in HKT, e.g. 2023-09-07 23:05 (warndb shows HKT)
Rows without it are built with all warning flags 0 (features only — labels wrong;
the script warns loudly and continues only with --allow-no-events).

Semantics (v4.1, aligned with fetch.py):
  rain_1h  = rainfall in the past hour, mm (station precip1h; live = mean over districts)
  rain_3h  = sum of the last three hourly rain_1h values
  rain_total = same as rain_1h here (dashboard-only column, NOT a model feature)
--rain-scale defaults to 1.0; the old x15 "district-sum" scaling is gone because
rain_total is no longer a feature. NOTE: ISD-Lite precip for 450070 is almost
entirely missing, so backfilled rain features are ~all zero — check the
"rain>0 hours" count printed per year before trusting them.
Backfilled rows leave TC-track and F3 features blank (train-time defaults apply).
"""

import argparse
import csv
import gzip
import math
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SNAPSHOT_CSV = DATA_DIR / "snapshots.csv"
EVENTS_CSV = DATA_DIR / "warning_events.csv"

ISD_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{year}/{station}-{year}.gz"
STATION = "450070-99999"  # Hong Kong International Airport (hourly, verified available)

HK_OFFSET = timedelta(hours=8)

EVENT_FLAGS = {
    "AMBER": ["w_RAIN_AMBER"],
    "RED": ["w_RAIN_RED"],
    "BLACK": ["w_RAIN_BLACK"],
    "TC1": ["w_TC1", "w_TCSGNL"],
    "TC3": ["w_TC3", "w_TCSGNL"],
    "TC8": ["w_TC8", "w_TCSGNL"],
    "TC9": ["w_TC8", "w_TCSGNL"],
    "TC10": ["w_TC8", "w_TCSGNL"],
    "WTS": ["w_WTS"],
    "WMSGNL": ["w_WMSGNL"],
}


def http_get_bytes(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Testing/1.0 (personal weather nowcast)"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read()


def rel_humidity(t_c, td_c):
    """Magnus formula; returns % clamped to [1, 100]."""
    try:
        e = math.exp(17.625 * td_c / (243.04 + td_c))
        es = math.exp(17.625 * t_c / (243.04 + t_c))
        return max(1.0, min(100.0, 100.0 * e / es))
    except (OverflowError, ZeroDivisionError):
        return None


def parse_isd_lite(raw_gz):
    """ISD-Lite: whitespace-separated year month day hour T*10 Td*10 SLP*10
    winddir wspd*10 sky precip1h*10 precip6h*10; -9999 = missing, -1 = trace.
    Timestamps are UTC."""
    out = []
    for line in gzip.decompress(raw_gz).decode("ascii", errors="replace").splitlines():
        f = line.split()
        if len(f) < 12:
            continue
        try:
            ts = datetime(int(f[0]), int(f[1]), int(f[2]), int(f[3]), tzinfo=timezone.utc)
        except ValueError:
            continue

        def val(i, scale=10.0):
            v = int(f[i])
            if v == -9999:
                return None
            if v == -1 and i >= 10:  # trace precip
                return 0.0
            return v / scale

        t = val(4)
        td = val(5)
        out.append({
            "ts": ts,
            "temp": t,
            "rh": rel_humidity(t, td) if t is not None and td is not None else None,
            "precip1h": max(0.0, val(10) or 0.0),
        })
    out.sort(key=lambda r: r["ts"])
    return out


def load_events():
    """data/warning_events.csv: type,start,end (HKT) -> list of (flags, start_utc, end_utc)."""
    events = []
    if not EVENTS_CSV.exists():
        return events
    with open(EVENTS_CSV, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            typ = (row.get("type") or "").strip().upper()
            flags = EVENT_FLAGS.get(typ)
            if not flags:
                continue
            try:
                start = datetime.fromisoformat(row["start"].strip().replace(" ", "T")) - HK_OFFSET
                end = datetime.fromisoformat(row["end"].strip().replace(" ", "T")) - HK_OFFSET
            except (ValueError, KeyError):
                continue
            events.append((flags, start.replace(tzinfo=timezone.utc), end.replace(tzinfo=timezone.utc)))
    return events


def flags_at(events, ts):
    active = {}
    for flags, start, end in events:
        if start <= ts <= end:
            for fl in flags:
                active[fl] = 1
    return active


def build_rows(obs, events, rain_scale, csv_columns):
    rows = []
    history = []  # (ts, rain_1h, temp, rh)
    for o in obs:
        rain_mm = o["precip1h"] * rain_scale

        def back(minutes):
            cutoff = o["ts"] - timedelta(minutes=minutes)
            for h in history:  # oldest-first
                if h[0] >= cutoff:
                    return h
            return None

        p60 = back(60 + 5)
        p120 = back(120 + 5)
        # rain_3h = this hour + the two previous hourly readings (same as fetch.py)
        rain_3h = rain_mm
        for prev in (p60, p120):
            if prev and prev[0] >= o["ts"] - timedelta(minutes=125):
                rain_3h += prev[1]
        hk = o["ts"] + HK_OFFSET
        row = dict.fromkeys(csv_columns, 0)
        row.update({
            "ts": o["ts"].isoformat(timespec="seconds"),
            "temp_mean": round(o["temp"], 1) if o["temp"] is not None else "",
            "hum_mean": round(o["rh"], 1) if o["rh"] is not None else "",
            "rain_total": round(rain_mm, 1),
            "rain_main": round(rain_mm, 1),
            "rain_1h": round(rain_mm, 1),
            "rain_3h": round(rain_3h, 1),
            "temp_1h_delta": round(o["temp"] - p60[2], 1) if p60 and None not in (o["temp"], p60[2]) else 0.0,
            "hum_1h_delta": round(o["rh"] - p60[3], 1) if p60 and None not in (o["rh"], p60[3]) else 0.0,
            "hour": hk.hour,
            "season": 1 if 5 <= hk.month <= 11 else 0,
            "tc_dist_km": "",
            "tc_wind_kts": "",
            "tc_24h_dist_km": "",
            "tc_trend_toward": "",
        })
        row.update(flags_at(events, o["ts"]))
        history.append((o["ts"], rain_mm, o["temp"], o["rh"]))
        history[:] = history[-8:]
        rows.append(row)
    return rows


def merge_into_csv(new_rows, csv_columns):
    existing = []
    if SNAPSHOT_CSV.exists():
        with open(SNAPSHOT_CSV, newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    live_ts = {r["ts"] for r in existing}
    live_hours = {r["ts"][:13] for r in existing}
    merged = list(existing)
    added = 0
    for r in new_rows:
        # live-polled rows are richer (TC/F3 features) — skip archived row if a
        # live row already covers that hour
        if r["ts"] in live_ts or r["ts"][:13] in live_hours:
            continue
        merged.append(r)
        added += 1
    merged.sort(key=lambda r: r["ts"])
    DATA_DIR.mkdir(exist_ok=True)
    with open(SNAPSHOT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction="ignore")
        writer.writeheader()
        for r in merged:
            writer.writerow(r)
    return len(existing), added, len(merged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start_year", type=int)
    ap.add_argument("end_year", type=int)
    ap.add_argument("--rain-scale", type=float, default=1.0,
                    help="multiplier on station precip (1.0 = raw mm; legacy x15 no longer needed)")
    ap.add_argument("--station", default=STATION)
    ap.add_argument("--allow-no-events", action="store_true",
                    help="build feature rows even without data/warning_events.csv (labels all 0)")
    args = ap.parse_args()

    from fetch import CSV_COLUMNS

    events = load_events()
    if not events and not args.allow_no_events:
        print("[backfill-tab] data/warning_events.csv missing or empty — backfilled rows would")
        print("               carry all-zero warning labels and poison training. Export the HKO")
        print("               warndb events first, or pass --allow-no-events to override.")
        sys.exit(1)
    print(f"[backfill-tab] {len(events)} warning events loaded")

    all_rows = []
    for year in range(args.start_year, args.end_year + 1):
        url = ISD_URL.format(year=year, station=args.station)
        try:
            raw = http_get_bytes(url)
        except Exception as e:
            print(f"[backfill-tab] {year}: download failed ({e}) — skipping")
            continue
        obs = parse_isd_lite(raw)
        rows = build_rows(obs, events, args.rain_scale, CSV_COLUMNS)
        all_rows.extend(rows)
        n_amber = sum(1 for r in rows if r.get("w_RAIN_AMBER") == 1)
        n_tc3 = sum(1 for r in rows if r.get("w_TC3") == 1)
        n_rain = sum(1 for r in rows if r.get("rain_1h", 0) > 0)
        print(f"[backfill-tab] {year}: {len(obs)} obs -> {len(rows)} rows "
              f"(amber-hours={n_amber}, tc3-hours={n_tc3}, rain>0 hours={n_rain})")

    if not all_rows:
        print("[backfill-tab] nothing fetched — snapshots.csv untouched")
        return
    n_before, added, n_after = merge_into_csv(all_rows, CSV_COLUMNS)
    print(f"[backfill-tab] merged: {n_before} existing + {added} archived -> {n_after} rows")

    # v4.2: tropical-cyclone features for the same years from IBTrACS (non-live rows only)
    try:
        from backfill_tc import fill_tc
        with open(SNAPSHOT_CSV, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r.setdefault("live", 0 if r.get("tc_dist_km", "") in ("", None) else r.get("live", ""))
        if fill_tc(rows, range(args.start_year, args.end_year + 1)):
            with open(SNAPSHOT_CSV, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
                w.writeheader()
                for r in rows:
                    w.writerow(r)
    except Exception as e:
        print(f"[backfill-tab] TC backfill skipped: {e}")
    for name, fl in [("AMBER", "w_RAIN_AMBER"), ("RED", "w_RAIN_RED"), ("BLACK", "w_RAIN_BLACK"),
                     ("TC3", "w_TC3"), ("TC8+", "w_TC8")]:
        print(f"  {name}: {sum(1 for r in all_rows if r.get(fl) == 1)} flagged hours")


if __name__ == "__main__":
    main()

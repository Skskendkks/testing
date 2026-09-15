"""P2/P7: F3 gridded-nowcast backfill — event-oriented day selection, real-future labels.

v4 changes vs the old backfill:
  * Label = observed rain ~2h later (the first lead frame of the snapshot 90-150 min
    ahead), NOT the current snapshot's own 4th lead frame. The old label taught the
    CNN to reproduce HKO's extrapolation instead of predicting real future rain.
  * Input = 5 channels: all 4 lead frames + a frame-to-frame delta (P7).
  * Stores B (advection/persistence baseline: max of current lead-4 frame) so cnn.py
    can verify the CNN beats "rain keeps moving the same way" (P7).
  * Event-oriented day selection (P2): rain-warning days from snapshots.csv and/or
    data/event_days.txt (fill from the HKO warndb), plus an equal number of random
    quiet days for negatives.
  * Merges with the existing dataset (dedupe by snapshot time) so repeated
    workflow_dispatch runs extend coverage.

v4.1: every run ALSO writes f3_max / f3_mean / f3_trend (+ f3_ok=1) into the
matching hourly rows of data/snapshots.csv (disable with --no-tabular-f3), so
the tabular models can be trained on rows that actually carry F3 rain
information (train.py switches to F3-only rows automatically once enough
exist). The existing "f3-events" workflow mode therefore needs no YAML change.

Usage:
    python app/backfill.py 7                                # last 7 days (recent top-up)
    python app/backfill.py --range 2025-04-01 2025-10-31 --events
    python app/backfill.py --range 2025-06-01 2025-06-30    # every day in range
    python app/backfill.py --range 2023-01-01 2026-08-01 --events --no-cnn
"""

import argparse
import csv
import io
import random
import sys
import urllib.parse
import urllib.request
import zipfile
from bisect import bisect_left
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grid import N_LEADS, downsample, f3_scalars, parse_grid_csv, window_indices

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATASET_PATH = DATA_DIR / "grid_dataset.npz"
SNAPSHOT_CSV = DATA_DIR / "snapshots.csv"
EVENT_DAYS_TXT = DATA_DIR / "event_days.txt"

ARCHIVE_GET = "https://app.data.gov.hk/v1/historical-archive/get-file?url={}&time={}"
TARGET_URL = "https://data.weather.gov.hk/weatherAPI/hko_data/F3/Gridded_rainfall_nowcast.csv"
EVENTS_CSV = DATA_DIR / "warning_events.csv"
# Note: F3 archive coverage verified to start 2022-12 (2966 snaps on 2022-12-01, ~8 GB/month).
INPUT_SCALE = 50.0
N_CHANNELS = 5  # 4 lead frames + delta(lead1 - lead0)
THRESHOLDS_MM = [15.0, 25.0, 35.0]
HK_OFFSET = timedelta(hours=8)


def http_get_bytes(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Testing/1.0 (personal weather nowcast)"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read()


def fetch_day(day, verbose=False):
    url = ARCHIVE_GET.format(urllib.parse.quote(TARGET_URL, safe=""), day.strftime("%Y%m%d"))
    data = http_get_bytes(url)
    zf = zipfile.ZipFile(io.BytesIO(data))
    snapshots = []
    for name in zf.namelist():
        if "Gridded_rainfall_nowcast.csv" not in name:
            continue
        text = zf.read(name).decode("utf-8", errors="replace")
        frames, latv, lonv = parse_grid_csv(text)
        if not frames:
            continue
        ilat, ilon = window_indices(latv, lonv)
        if not ilat or not ilon:
            continue
        endings = sorted(frames.keys())
        if len(endings) < N_LEADS:
            continue
        leads = [downsample(frames[e], ilat, ilon) for e in endings[:N_LEADS]]
        ts = datetime.strptime(endings[0], "%Y%m%d%H%M").replace(tzinfo=timezone.utc) - HK_OFFSET
        snapshots.append((ts, leads))
    snapshots.sort(key=lambda s: s[0])
    if verbose:
        print(f"[backfill] {day:%Y-%m-%d}: {len(snapshots)} snapshots")
    return snapshots


def make_input(leads):
    """5-channel input: 4 scaled lead frames + scaled delta(lead1-lead0)."""
    frames = [np.array(g, dtype=np.float32) for g in leads[:4]]
    delta = frames[1] - frames[0]
    x = np.stack(frames + [delta]) / INPUT_SCALE
    x[:4] = np.clip(x[:4], 0.0, 1.0)
    x[4] = np.clip(x[4], -1.0, 1.0)
    return x


def build_samples(all_snap):
    """Pair each snapshot with the snapshot ~120 min later; label from its lead-0 frame."""
    all_snap.sort(key=lambda s: s[0])
    times = [s[0] for s in all_snap]
    X, y, B, T = [], [], [], []
    for ts, leads in all_snap:
        lo = bisect_left(times, ts + timedelta(minutes=90))
        best = None
        for m in range(lo, len(times)):
            dt_min = (times[m] - ts).total_seconds() / 60.0
            if dt_min > 150:
                break
            if best is None or abs(dt_min - 120) < abs(best[1] - 120):
                best = (m, dt_min)
        if best is None:
            continue
        future_leads = all_snap[best[0]][1]
        observed_max = float(np.max(np.array(future_leads[0], dtype=np.float32)))
        X.append(make_input(leads))
        y.append([1.0 if observed_max >= t else 0.0 for t in THRESHOLDS_MM])
        # advection/persistence baseline: HKO's own ~2h lead frame from the current snapshot
        B.append(float(np.max(np.array(leads[3], dtype=np.float32))))
        T.append(ts.isoformat())
    return X, y, B, T


def load_existing():
    if not DATASET_PATH.exists():
        return None
    d = np.load(DATASET_PATH, allow_pickle=False)
    if "X" not in d or d["X"].ndim != 4 or d["X"].shape[1] != N_CHANNELS or "B" not in d:
        print("[backfill] existing dataset uses the old schema — rebuilding from scratch")
        return None
    return d


def save_dataset(X, y, B, T):
    existing = load_existing()
    if existing is not None:
        seen = set(T)
        keep = [i for i, t in enumerate(existing["times"]) if str(t) not in seen]
        X = list(existing["X"][keep]) + X
        y = list(existing["y"][keep]) + y
        B = list(existing["B"][keep]) + B
        T = [str(t) for t in existing["times"][keep]] + T
    order = np.argsort(np.array(T))
    Xa = np.stack([np.asarray(X[i], dtype=np.float32) for i in order])
    ya = np.stack([np.asarray(y[i], dtype=np.float32) for i in order])
    Ba = np.array([B[i] for i in order], dtype=np.float32)
    Ta = np.array([T[i] for i in order])
    DATA_DIR.mkdir(exist_ok=True)
    np.savez_compressed(DATASET_PATH, X=Xa, y=ya, B=Ba, times=Ta)
    print(f"[backfill] dataset: {Xa.shape[0]} samples ({Xa.shape[1]}ch) -> {DATASET_PATH} "
          f"({DATASET_PATH.stat().st_size / 1e6:.1f} MB)")
    for k, name in enumerate(["rain120_15mm", "rain120_25mm", "rain120_35mm"]):
        print(f"  {name}: {int(ya[:, k].sum())} positives / {ya.shape[0]}")


F3_MATCH_MINUTES = 20   # hourly snapshot row <-> nearest F3 snapshot tolerance


def add_f3_rolling(rows):
    """f3_max_1h / f3_max_3h = max of f3_max over this row and rows within the past 1h / 3h
    that carry real F3 data. Rows are assumed sorted by ts."""
    hist = []  # (ts, f3_max) for f3_ok rows
    for r in rows:
        try:
            ts = datetime.fromisoformat(r["ts"])
        except (KeyError, ValueError):
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if str(r.get("f3_ok", "")) != "1":
            continue
        cur = float(r.get("f3_max") or 0.0)
        hist = [(t, v) for t, v in hist if (ts - t) <= timedelta(minutes=185)]
        m1 = max([cur] + [v for t, v in hist if (ts - t) <= timedelta(minutes=65)])
        m3 = max([cur] + [v for t, v in hist])
        r["f3_max_1h"], r["f3_max_3h"] = round(m1, 2), round(m3, 2)
        hist.append((ts, cur))


def write_f3_to_snapshots(all_snap):
    """Attach f3_* scalars to the hourly rows of snapshots.csv nearest each F3 snapshot."""
    if not SNAPSHOT_CSV.exists():
        print("[backfill] snapshots.csv missing — nothing to attach F3 features to")
        return
    from fetch import CSV_COLUMNS
    with open(SNAPSHOT_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    all_snap.sort(key=lambda s: s[0])
    times = [s[0] for s in all_snap]
    updated = 0
    for r in rows:
        if str(r.get("f3_ok", "")) == "1":
            continue  # live row (or already backfilled) — keep the real-time value
        try:
            ts = datetime.fromisoformat(r["ts"])
        except (KeyError, ValueError):
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        k = bisect_left(times, ts)
        best = None
        for m in (k - 1, k):
            if 0 <= m < len(times):
                gap = abs((times[m] - ts).total_seconds()) / 60.0
                if gap <= F3_MATCH_MINUTES and (best is None or gap < best[1]):
                    best = (m, gap)
        if best is None:
            continue
        r.update(f3_scalars(all_snap[best[0]][1]))
        r["f3_ok"] = 1
        updated += 1
    add_f3_rolling(rows)
    with open(SNAPSHOT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    n_ok = sum(1 for r in rows if str(r.get("f3_ok", "")) == "1")
    print(f"[backfill] F3 scalars attached to {updated} snapshot rows (f3_ok rows now: {n_ok}/{len(rows)})")


def event_days_from_snapshots(start, end):
    """HK dates with any rain warning active, from backfilled/live snapshots.csv."""
    days = set()
    if not SNAPSHOT_CSV.exists():
        return days
    with open(SNAPSHOT_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if any(str(row.get(fl, 0)) == "1" for fl in ("w_RAIN_AMBER", "w_RAIN_RED", "w_RAIN_BLACK")):
                try:
                    d = (datetime.fromisoformat(row["ts"]) + HK_OFFSET).date()
                except ValueError:
                    continue
                if start <= d <= end:
                    days.add(d)
    return days


def event_days_from_events_csv(start, end):
    """Rain-warning days from data/warning_events.csv (warndb export; HKT times)."""
    days = set()
    if not EVENTS_CSV.exists():
        return days
    with open(EVENTS_CSV, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if (row.get("type") or "").strip().upper() not in ("AMBER", "RED", "BLACK"):
                continue
            try:
                s = datetime.fromisoformat(row["start"].strip().replace(" ", "T")).date()
                e = datetime.fromisoformat(row["end"].strip().replace(" ", "T")).date()
            except (ValueError, KeyError):
                continue
            d = s
            while d <= e:
                if start <= d <= end:
                    days.add(d)
                d += timedelta(days=1)
    return days


def event_days_from_file(start, end):
    """Manual event dates (one YYYY-MM-DD per line) — fill from the HKO warndb."""
    days = set()
    if not EVENT_DAYS_TXT.exists():
        return days
    for line in EVENT_DAYS_TXT.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            d = datetime.strptime(line, "%Y-%m-%d").date()
        except ValueError:
            continue
        if start <= d <= end:
            days.add(d)
    return days


def pick_days(args):
    if args.range:
        start = datetime.strptime(args.range[0], "%Y-%m-%d").date()
        end = datetime.strptime(args.range[1], "%Y-%m-%d").date()
        if args.events:
            events = (event_days_from_snapshots(start, end) | event_days_from_file(start, end)
                      | event_days_from_events_csv(start, end))
            if not events:
                print("[backfill] no event days found in range — provide data/warning_events.csv "
                      "(warndb export), run backfill_tabular.py first, or add dates to "
                      "data/event_days.txt")
                return []
            all_days = {start + timedelta(days=i) for i in range((end - start).days + 1)}
            quiet_pool = sorted(all_days - events)
            rng = random.Random(0)
            quiet = set(rng.sample(quiet_pool, min(len(events), len(quiet_pool))))
            days = sorted(events | quiet)
            print(f"[backfill] event mode: {len(events)} event days + {len(quiet)} quiet days")
        else:
            days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        if args.max_days and len(days) > args.max_days:
            days = days[-args.max_days:]
        return [datetime(d.year, d.month, d.day) for d in days]
    n = args.days if args.days is not None else 6
    end = datetime.now(timezone.utc) - timedelta(days=1)
    return [end - timedelta(days=i) for i in range(n, -1, -1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("days", nargs="?", type=int, help="last N days (legacy mode)")
    ap.add_argument("--range", nargs=2, metavar=("START", "END"), help="YYYY-MM-DD YYYY-MM-DD")
    ap.add_argument("--events", action="store_true",
                    help="within --range, fetch only rain-warning days + equal random quiet days")
    ap.add_argument("--max-days", type=int, default=0, help="cap number of days fetched")
    ap.add_argument("--no-tabular-f3", action="store_true",
                    help="do NOT write f3_* scalars into matching hourly rows of data/snapshots.csv")
    ap.add_argument("--no-cnn", action="store_true", help="skip building grid_dataset.npz")
    args = ap.parse_args()

    days = pick_days(args)
    if not days:
        return
    all_snap = []
    for d in days:
        try:
            all_snap.extend(fetch_day(d, verbose=True))
        except Exception as e:
            print(f"[backfill] {d:%Y-%m-%d}: failed ({e}) — skipping")
    if not all_snap:
        print("[backfill] no snapshots fetched")
        return
    if not args.no_tabular_f3:
        write_f3_to_snapshots(all_snap)
    if args.no_cnn:
        return
    X, y, B, T = build_samples(all_snap)
    if not X:
        print("[backfill] no labelable samples (need snapshots ~2h apart)")
        return
    save_dataset(X, y, B, T)


if __name__ == "__main__":
    main()

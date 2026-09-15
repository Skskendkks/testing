"""One-off: convert data/snapshots.csv rain columns to the v4.1 semantics.

Old semantics
  backfill rows : rain_total = never-resetting cumulative sum (x15); rain_1h/3h = diffs
  live rows     : rain_total = SUM over ~18 districts of past-hour rainfall;
                  rain_1h/3h = differences of that sum (could go negative)

New semantics (fetch.py / backfill_tabular.py v4.1)
  rain_1h    = mean past-hour rainfall, mm        (live: rain_total / N_DISTRICTS)
  rain_3h    = rain_1h now + ~1h ago + ~2h ago
  rain_total = live: district sum (unchanged); backfill: = rain_1h

Also adds the f3_ok column (1 on live rows that carried an F3 snapshot).

Runs automatically (once) from fetch.py / train.py when data/.v41_migrated is
missing, so the hourly poll workflow migrates and commits the data by itself.
Manual: python app/migrate_rain_v41.py
Outside CI a backup is written to data/snapshots.pre_v41.csv (git-ignored).
"""

import csv
import shutil
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "snapshots.csv"
BACKUP = ROOT / "data" / "snapshots.pre_v41.csv"
MARKER = ROOT / "data" / ".v41_migrated"
N_DISTRICTS = 18  # rhrread rainfall districts; approximate for already-collected rows


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def is_live(row):
    return row.get("tc_dist_km", "") not in ("", None)


def ensure():
    """Run the migration if it has not been applied yet (idempotent via marker file)."""
    if MARKER.exists():
        return False
    main()
    return True


def main():
    if MARKER.exists():
        print("[migrate] already applied (data/.v41_migrated present)")
        return
    if not CSV.exists():
        print("no snapshots.csv")
        return
    import os
    if not BACKUP.exists() and not os.environ.get("GITHUB_ACTIONS"):
        shutil.copy(CSV, BACKUP)
    with open(CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fetch import CSV_COLUMNS as cols
    for r in rows:
        r["_dt"] = datetime.fromisoformat(r["ts"])
    rows.sort(key=lambda r: r["_dt"])

    # pass 1: rain_1h
    n_live = n_bk = 0
    for r in rows:
        if is_live(r):
            n_live += 1
            r["rain_1h"] = round(_f(r["rain_total"]) / N_DISTRICTS, 1)
            if not r.get("f3_ok"):
                r["f3_ok"] = 1 if r.get("f3_max", "") not in ("", None) else 0
        else:
            n_bk += 1
            r.setdefault("f3_ok", 0)
            # backfill: old rain_1h was cumulative diff = station precip x15 -> undo the x15
            r["rain_1h"] = round(_f(r["rain_1h"]) / 15.0, 1)
            r["rain_total"] = r["rain_1h"]
            r["rain_main"] = r["rain_1h"]

    # pass 2: rain_3h = rolling sum of hourly rain_1h (nearest rows ~60 and ~120 min back)
    hist = []
    for r in rows:
        total = _f(r["rain_1h"])
        for mins in (60, 120):
            target = r["_dt"] - timedelta(minutes=mins)
            best, gap_best = None, None
            for h in reversed(hist):
                gap = abs((h[0] - target).total_seconds()) / 60
                if gap <= 25 and (gap_best is None or gap < gap_best):
                    best, gap_best = h, gap
                if h[0] < target - timedelta(minutes=25):
                    break
            if best:
                total += best[1]
        r["rain_3h"] = round(total, 1)
        hist.append((r["_dt"], _f(r["rain_1h"])))
        hist = hist[-12:]

    with open(CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    MARKER.write_text(datetime.now().isoformat(timespec="seconds") + "\n", encoding="utf-8")
    neg = sum(1 for r in rows if _f(r["rain_1h"]) < 0 or _f(r["rain_3h"]) < 0)
    print(f"[migrate] {len(rows)} rows ({n_live} live, {n_bk} backfill) rewritten; negative rain rows: {neg}")
    if BACKUP.exists():
        print(f"[migrate] backup at {BACKUP}")


if __name__ == "__main__":
    main()

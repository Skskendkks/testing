"""v4.2: backfill tropical-cyclone features for historical rows from IBTrACS.

IBTrACS (NOAA NCEI) carries the JTWC best track for every Western Pacific
system, 3-hourly, back to the 1940s, as one CSV:
  https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.WP.list.v04r01.csv

For each hourly snapshot row that was not polled live (live != 1) we compute the
same four features fetch.py gets from the JTWC warning text:
  tc_dist_km       distance HK -> nearest active cyclone (linear interpolation between fixes)
  tc_wind_kts      its 1-min max wind (USA_WIND, fallback WMO_WIND)
  tc_24h_dist_km   distance to the position extrapolated 24 h ahead with the last-6h motion
                   (a persistence forecast — we deliberately do NOT use the true future
                   position, which live rows never see)
  tc_trend_toward  1 if that extrapolated position is closer to HK than now
  tc_dist_rate     tc_dist_km now minus one hour earlier (km/h; negative = approaching)
Rows with no cyclone within NEAR_RADIUS_KM get the 2000 km defaults, like live rows.

Usage (also invoked by backfill_tabular.py after merging ISD rows):
    python app/backfill_tc.py 1998 2026
"""

import csv
import io
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from jtwc import HK_LAT, HK_LON, NEAR_RADIUS_KM, haversine_km

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SNAPSHOT_CSV = DATA_DIR / "snapshots.csv"
CACHE = DATA_DIR / "ibtracs_wp.csv"   # git-ignored download cache

IBTRACS_URLS = [
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.WP.list.v04r01.csv",
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs.WP.list.v04r00.csv",
]
DEFAULT_DIST = 2000.0
MIN_WIND = 20


def http_get_bytes(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Testing/1.0 (personal weather nowcast)"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return resp.read()


def load_ibtracs(years):
    """{sid: [(dt, lat, lon, wind), ...]} for storms touching the given years."""
    if CACHE.exists():
        raw = CACHE.read_bytes()
    else:
        raw, last_err = None, None
        for url in IBTRACS_URLS:
            try:
                raw = http_get_bytes(url)
                print(f"[backfill-tc] downloaded {url.rsplit('/', 1)[-1]} ({len(raw) / 1e6:.1f} MB)")
                break
            except Exception as e:
                last_err = e
        if raw is None:
            raise RuntimeError(f"IBTrACS download failed: {last_err}")
        DATA_DIR.mkdir(exist_ok=True)
        CACHE.write_bytes(raw)
    reader = csv.reader(io.StringIO(raw.decode("utf-8", "replace")))
    header = next(reader)
    next(reader, None)  # units row
    col = {name: i for i, name in enumerate(header)}
    need = ["SID", "SEASON", "ISO_TIME", "LAT", "LON"]
    for n in need:
        if n not in col:
            raise RuntimeError(f"IBTrACS column {n} missing")
    i_uw, i_ww = col.get("USA_WIND"), col.get("WMO_WIND")
    storms = defaultdict(list)
    for f in reader:
        try:
            if int(f[col["SEASON"]]) not in years:
                continue
            dt = datetime.strptime(f[col["ISO_TIME"]], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            lat, lon = float(f[col["LAT"]]), float(f[col["LON"]])
        except (ValueError, IndexError):
            continue
        wind = 0
        for i in (i_uw, i_ww):
            if i is not None and f[i].strip():
                try:
                    wind = int(float(f[i]))
                    break
                except ValueError:
                    pass
        storms[f[col["SID"]]].append((dt, lat, lon, wind))
    for sid in storms:
        storms[sid].sort()
    return storms


def _interp(fixes, t):
    """Linear interpolation of (lat, lon, wind) at time t; None outside the track."""
    if t < fixes[0][0] or t > fixes[-1][0]:
        return None
    lo, hi = 0, len(fixes) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fixes[mid][0] <= t:
            lo = mid
        else:
            hi = mid
    a, b = fixes[lo], fixes[hi]
    span = (b[0] - a[0]).total_seconds()
    w = 0.0 if span <= 0 else (t - a[0]).total_seconds() / span
    return (a[1] + w * (b[1] - a[1]), a[2] + w * (b[2] - a[2]), a[3] + w * (b[3] - a[3]))


def storm_hourly_features(fixes):
    """{hour_dt: feature dict} for every whole hour the storm is within NEAR_RADIUS_KM."""
    out = {}
    start = fixes[0][0].replace(minute=0, second=0, microsecond=0)
    t = start
    while t <= fixes[-1][0]:
        cur = _interp(fixes, t)
        if cur and cur[2] >= MIN_WIND:
            dist = haversine_km(HK_LAT, HK_LON, cur[0], cur[1])
            if dist <= NEAR_RADIUS_KM:
                prev6 = _interp(fixes, t - timedelta(hours=6)) or cur
                prev1 = _interp(fixes, t - timedelta(hours=1))
                lat24 = cur[0] + 4 * (cur[0] - prev6[0])
                lon24 = cur[1] + 4 * (cur[1] - prev6[1])
                d24 = haversine_km(HK_LAT, HK_LON, lat24, lon24)
                d1 = haversine_km(HK_LAT, HK_LON, prev1[0], prev1[1]) if prev1 else dist
                out[t] = {
                    "tc_dist_km": round(min(dist, DEFAULT_DIST)),
                    "tc_wind_kts": int(round(cur[2])),
                    "tc_24h_dist_km": round(min(d24, DEFAULT_DIST)),
                    "tc_trend_toward": 1 if d24 < dist else 0,
                    "tc_dist_rate": round(dist - d1, 1),
                }
        t += timedelta(hours=1)
    return out


def fill_tc(rows, years, verbose=True):
    """Set tc_* on non-live rows whose hour falls inside the given years. Returns rows updated."""
    storms = load_ibtracs(set(years))
    best = {}  # hour -> features of the nearest storm
    for sid, fixes in storms.items():
        for h, feats in storm_hourly_features(fixes).items():
            if h not in best or feats["tc_dist_km"] < best[h]["tc_dist_km"]:
                best[h] = feats
    updated = near = 0
    for r in rows:
        if str(r.get("live", "")).strip() == "1":
            continue
        try:
            ts = datetime.fromisoformat(r["ts"])
        except (KeyError, ValueError):
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts.year not in years:
            continue
        h = ts.replace(minute=0, second=0, microsecond=0)
        feats = best.get(h)
        if feats:
            near += 1
        else:
            feats = {"tc_dist_km": DEFAULT_DIST, "tc_wind_kts": 0, "tc_24h_dist_km": DEFAULT_DIST,
                     "tc_trend_toward": 0, "tc_dist_rate": 0.0}
        r.update(feats)
        updated += 1
    if verbose:
        print(f"[backfill-tc] {len(storms)} storms; tc_* set on {updated} rows ({near} with a cyclone within {NEAR_RADIUS_KM} km)")
    return updated


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    years = range(int(sys.argv[1]), int(sys.argv[2]) + 1)
    from fetch import CSV_COLUMNS
    with open(SNAPSHOT_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fill_tc(rows, years)
    with open(SNAPSHOT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    main()

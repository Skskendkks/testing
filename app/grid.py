import csv
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

F3_URL = "https://data.weather.gov.hk/weatherAPI/hko_data/F3/Gridded_rainfall_nowcast.csv"

HK_LAT_MIN, HK_LAT_MAX = 22.05, 22.65
HK_LON_MIN, HK_LON_MAX = 113.75, 114.55
OUT_SIZE = 32
N_LEADS = 4


def http_get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Testing/1.0 (personal weather nowcast)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def parse_grid_csv(text):
    frames = {}
    for row in csv.DictReader(text.splitlines()):
        ending = row.get("Ending Date and Time (in Hong Kong Time)", "")
        lat_s = row.get("Latitude (degree)", "")
        lon_s = row.get("Longitude (degree)", "")
        rain_s = row.get("Half-hourly Nowcast Accumulated Rainfall (mm)", "")
        if not ending or not lat_s or not lon_s or len(ending) != 12 or not ending.isdigit():
            continue
        try:
            lat = float(lat_s)
            lon = float(lon_s)
            rain = float(rain_s or "0")
        except ValueError:
            continue
        frames.setdefault(ending, {})[(lat, lon)] = rain
    if not frames:
        return {}, [], []
    latv = sorted({lat for lat, _ in next(iter(frames.values())).keys()})
    lonv = sorted({lon for _, lon in next(iter(frames.values())).keys()})
    out = {}
    for ending, cells in frames.items():
        grid = [[cells.get((lat, lon), 0.0) for lon in lonv] for lat in latv]
        out[ending] = grid
    return out, latv, lonv


def window_indices(latv, lonv):
    ilat = [i for i, v in enumerate(latv) if HK_LAT_MIN <= v <= HK_LAT_MAX]
    ilon = [i for i, v in enumerate(lonv) if HK_LON_MIN <= v <= HK_LON_MAX]
    return ilat, ilon


def downsample(grid, ilat, ilon, out_size=OUT_SIZE):
    rows = [grid[i] for i in ilat]
    cols = [[r[j] for j in ilon] for r in rows]
    nlat, nlon = len(rows), len(ilon)
    block_lat = max(1, nlat // out_size)
    block_lon = max(1, nlon // out_size)
    out = []
    for i in range(out_size):
        r0 = min(i * block_lat, nlat)
        r1 = min((i + 1) * block_lat, nlat)
        if r1 <= r0:
            r1 = r0 + 1
        row_out = []
        for j in range(out_size):
            c0 = min(j * block_lon, nlon)
            c1 = min((j + 1) * block_lon, nlon)
            if c1 <= c0:
                c1 = c0 + 1
            vals = [cols[r][c] for r in range(r0, r1) for c in range(c0, c1)]
            row_out.append(round(sum(vals) / len(vals), 2))
        out.append(row_out)
    return out


def fetch_snapshot():
    text = http_get(F3_URL)
    frames, latv, lonv = parse_grid_csv(text)
    if not frames:
        return None
    ilat, ilon = window_indices(latv, lonv)
    if not ilat or not ilon:
        return None
    endings = sorted(frames.keys())
    leads = [downsample(frames[e], ilat, ilon) for e in endings[:N_LEADS]]
    if len(leads) < N_LEADS:
        return None
    return {"ts": endings[0], "leads": leads, "n_leads": len(leads)}


def f3_scalars(leads):
    """Scalar summaries of the downsampled lead frames (shared by fetch.py and backfill.py)."""
    if not leads:
        return {"f3_max": 0.0, "f3_mean": 0.0, "f3_trend": 0.0}
    maxes = [max(max(r) for r in g) for g in leads]
    means = [sum(sum(r) for r in g) / (len(g) * len(g[0])) for g in leads]
    return {
        "f3_max": round(max(maxes), 2),
        "f3_mean": round(sum(means) / len(means), 3),
        "f3_trend": round(maxes[-1] - maxes[0], 2),
    }


def lead_input(leads):
    return leads[:3]


def hk_ts_to_utc(hk_ts):
    return datetime.strptime(hk_ts, "%Y%m%d%H%M").replace(tzinfo=timezone.utc) - timedelta(hours=8)


if __name__ == "__main__":
    snap = fetch_snapshot()
    if snap:
        print(f"[grid] snapshot {snap['ts']} with {snap['n_leads']} leads, grid 32x32")
    else:
        print("[grid] fetch failed")

"""V2 tropical-cyclone features.

v4.1 fix: the previous source (NHC ATCF, ftp.nhc.noaa.gov/atcf/btk) only carries
Atlantic / East & Central Pacific basins (bal*, bep*, bcp*) — it has NO Western
Pacific files, so find_storms() never returned anything and tc_dist_km was
always the 2000 km default. Western Pacific warnings come from JTWC itself:

  RSS   https://www.metoc.navy.mil/jtwc/rss/jtwc.rss        (active systems + links)
  text  https://www.metoc.navy.mil/jtwc/products/wp{NN}{YY}web.txt   (warning text)

The warning text carries the current fix ("WARNING POSITION: ... NEAR 17.2N 127.8W",
"MAX SUSTAINED WINDS - 050 KT") and 12/24/36/48/72/96/120 h forecast fixes.
JTWC warnings do not include central pressure, so pressure_mb is None.
"""

import math
import re
import urllib.request
from datetime import datetime, timezone

JTWC_RSS_URL = "https://www.metoc.navy.mil/jtwc/rss/jtwc.rss"
JTWC_TEXT_URL = "https://www.metoc.navy.mil/jtwc/products/wp{num}{yy}web.txt"

HK_LAT = 22.3027
HK_LON = 114.1742
NEAR_RADIUS_KM = 2500
REPORT_RADIUS_KM = 1500

_POS_RE = re.compile(r"(\d{6})Z\s*-+\s*(?:NEAR\s+)?(\d{1,2}\.\d)([NS])\s+(\d{1,3}\.\d)([EW])")
_WIND_RE = re.compile(r"MAX SUSTAINED WINDS\s*-\s*(\d{2,3})\s*KT")
_FHR_RE = re.compile(r"^\s*(\d{2,3})\s*HRS?,?\s*VALID AT", re.I)
_RSS_WP_RE = re.compile(r"products/wp(\d{2})(\d{2})web\.txt", re.I)


def http_get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Testing/1.0 (personal weather nowcast)"})
    with urllib.request.urlopen(req, timeout=45) as resp:
        return resp.read()


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def parse_warning(text, storm_id):
    """JTWC warning text -> (current_fix, {forecast_hr: fix}).

    fix = {"dt": datetime (UTC, day/hour/minute from DDHHMMZ), "lat", "lon", "wind"}.
    Lines are scanned in order: "WARNING POSITION:" starts the current fix,
    "NN HRS, VALID AT:" starts a forecast fix; the next position / wind line
    is attached to whichever block is open.
    """
    now = datetime.now(timezone.utc)
    current, forecasts = None, {}
    open_fhr = None   # 0 = current, N = forecast hour
    for raw in text.splitlines():
        line = raw.strip().upper()
        if "WARNING POSITION" in line:
            open_fhr = 0
        m = _FHR_RE.match(line)
        if m:
            open_fhr = int(m.group(1))
        m = _POS_RE.search(line)
        if m and open_fhr is not None:
            lat = float(m.group(2)) * (-1 if m.group(3) == "S" else 1)
            lon = float(m.group(4)) * (-1 if m.group(5) == "W" else 1)
            ddhhmm = m.group(1)
            try:
                dt = now.replace(day=int(ddhhmm[:2]), hour=int(ddhhmm[2:4]),
                                 minute=int(ddhhmm[4:6]), second=0, microsecond=0)
            except ValueError:
                dt = now
            fix = {"dt": dt, "lat": lat, "lon": lon, "wind": 0}
            if open_fhr == 0:
                current = fix
            else:
                forecasts[open_fhr] = fix
            continue
        m = _WIND_RE.search(line)
        if m and open_fhr is not None:
            target = current if open_fhr == 0 else forecasts.get(open_fhr)
            if target is not None and target["wind"] == 0:
                target["wind"] = int(m.group(1))
    if current is None or current["wind"] < 20:
        return None, {}
    return current, forecasts


def find_storms():
    """Active Western Pacific systems from the JTWC RSS feed (numbers < 90; 9x = invest/TCFA)."""
    try:
        rss = http_get(JTWC_RSS_URL).decode("utf-8", "replace")
    except Exception:
        return []
    seen, storms = set(), []
    for m in _RSS_WP_RE.finditer(rss):
        num, yy = m.group(1), m.group(2)
        if int(num) >= 90 or (num, yy) in seen:
            continue
        seen.add((num, yy))
        storms.append({"basin": "WP", "num": num, "yy": yy, "id": f"WP{num}"})
    return storms


def fetch_warning(storm):
    return http_get(JTWC_TEXT_URL.format(num=storm["num"], yy=storm["yy"])).decode("utf-8", "replace")


def storm_info(storm, latest, forecasts):
    if not latest:
        return None
    dist = haversine_km(HK_LAT, HK_LON, latest["lat"], latest["lon"])
    bearing = bearing_deg(HK_LAT, HK_LON, latest["lat"], latest["lon"])
    target_24 = min(forecasts, key=lambda f: abs(f - 24)) if forecasts else None
    forecast = forecasts.get(target_24) if target_24 is not None else None
    info = {
        "id": storm["id"],
        "ts": latest["dt"].strftime("%Y-%m-%dT%H:%M"),
        "lat": round(latest["lat"], 1),
        "lon": round(latest["lon"], 1),
        "wind_kts": latest["wind"],
        "pressure_mb": latest.get("pressure") or None,
        "distance_km": round(dist),
        "bearing_deg": round(bearing),
        "forecast_24h_km": round(haversine_km(HK_LAT, HK_LON, forecast["lat"], forecast["lon"])) if forecast else None,
    }
    info["moving_toward_hk"] = bool(info["forecast_24h_km"] and info["forecast_24h_km"] < dist)
    return info


def scan():
    storms = find_storms()
    results = []
    for storm in storms:
        try:
            current, forecasts = parse_warning(fetch_warning(storm), storm["id"])
            info = storm_info(storm, current, forecasts)
            if info:
                results.append(info)
        except Exception as e:
            print(f"[jtwc] {storm['id']}: skipped ({e})")
            continue
    near = [s for s in results if s["distance_km"] <= NEAR_RADIUS_KM]
    near.sort(key=lambda s: s["distance_km"])
    return {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scanned": [s["id"] for s in results],
        "nearest": near[0] if near else None,
    }


def snapshot_features(tc_state):
    n = tc_state.get("nearest") or {}
    dist = n.get("distance_km") or 2000
    return {
        "tc_dist_km": min(dist, 2000),
        "tc_wind_kts": n.get("wind_kts") or 0,
        "tc_24h_dist_km": min(n.get("forecast_24h_km") or 2000, 2000),
        "tc_trend_toward": 1 if n.get("moving_toward_hk") else 0,
    }

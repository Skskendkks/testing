import csv
import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import TARGETS, TARGET_FLAG, TARGET_LABELS, blend_weights, predict_ai
import jtwc
from notify import ai_alert_keys, alert_thresholds, load_notified, save_notified, send_email
from rules import rule_probs
import grid as gridmod

try:
    import cnn as cnnmod
except ImportError:
    cnnmod = None

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STATE_DIR = ROOT / "state"
SITE_DATA_DIR = ROOT / "site" / "data"
SNAPSHOT_CSV = DATA_DIR / "snapshots.csv"
LATEST_JSON = DATA_DIR / "latest.json"
HISTORY_JSON = DATA_DIR / "history.json"
LAST_WARN = STATE_DIR / "last_warnings.json"

WEATHER_URL = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=rhrread&lang=en"
WARNSUM_URL = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=warnsum&lang=en"

HK_OFFSET = timedelta(hours=8)

CSV_COLUMNS = [
    "ts",
    "temp_mean",
    "hum_mean",
    "rain_total",
    "rain_main",
    "rain_1h",
    "rain_3h",
    "hum_1h_delta",
    "temp_1h_delta",
    "hour",
    "season",
    "w_TCSGNL",
    "w_TC1",
    "w_TC3",
    "w_TC8",
    "w_RAIN_AMBER",
    "w_RAIN_RED",
    "w_RAIN_BLACK",
    "w_WTS",
    "w_WMSGNL",
    "w_WL",
    "w_WFNTSA",
    "w_WHOT",
    "w_WCOLD",
    "w_WFROST",
    "w_WFIRE",
    "tc_dist_km",
    "tc_wind_kts",
    "tc_24h_dist_km",
    "tc_trend_toward",
    "tc_dist_rate",
    "f3_max",
    "f3_mean",
    "f3_trend",
    "f3_ok",
    "f3_max_1h",
    "f3_max_3h",
    "live",        # 1 = polled in real time (full feature set); 0/blank = backfilled
]

WARNSUMS_TO_FLAGS = {
    "WTCSGNL": "w_TCSGNL",
    "WRAIN": "w_RAIN",
    "WTS": "w_WTS",
    "WMSGNL": "w_WMSGNL",
    "WL": "w_WL",
    "WFNTSA": "w_WFNTSA",
    "WHOT": "w_WHOT",
    "WCOLD": "w_WCOLD",
    "WFROST": "w_WFROST",
    "WFIRE": "w_WFIRE",
}

LEVEL_NAMES = {
    "w_TC1": "TC Signal No. 1",
    "w_TC3": "TC Signal No. 3",
    "w_TC8": "TC Signal No. 8/9/10",
    "w_RAIN_AMBER": "Amber Rainstorm",
    "w_RAIN_RED": "Red Rainstorm",
    "w_RAIN_BLACK": "Black Rainstorm",
}


def http_get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Testing/1.0 (personal weather nowcast)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_levels(messages):
    flags = {}
    for text in messages:
        up = text.upper()
        if "RAINSTORM" in up:
            flags["w_RAIN_AMBER"] = "AMBER" in up
            flags["w_RAIN_RED"] = "RED" in up
            flags["w_RAIN_BLACK"] = "BLACK" in up
        if "TROPICAL CYCLONE" in up and "SIGNAL" in up:
            m = re.search(r"(\d{1,2})", up)
            if m:
                level = int(m.group(1))
                flags["w_TC1"] = level == 1
                flags["w_TC3"] = level == 3
                flags["w_TC8"] = level >= 8
    return flags


def load_csv_rows():
    if not SNAPSHOT_CSV.exists():
        return []
    with open(SNAPSHOT_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rows_within(rows, minutes):
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    out = []
    for r in rows:
        ts = datetime.fromisoformat(r["ts"])
        if ts >= cutoff:
            out.append(r)
    return out


def _f(row, key):
    return float(row.get(key, 0.0) or 0.0)


def row_near(rows, minutes_ago, tolerance=25):
    """Prior row whose timestamp is closest to now - minutes_ago (within tolerance min)."""
    target = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    best, best_gap = None, None
    for r in rows:
        try:
            gap = abs((datetime.fromisoformat(r["ts"]) - target).total_seconds()) / 60.0
        except (KeyError, ValueError):
            continue
        if gap <= tolerance and (best_gap is None or gap < best_gap):
            best, best_gap = r, gap
    return best


def f3_features(leads):
    """P6: scalar summaries of the F3 gridded nowcast for the tabular models."""
    out = gridmod.f3_scalars(leads)
    out["f3_ok"] = 1 if leads else 0   # 1 = real F3 data behind the f3_* values
    return out


def build_row(weather, warnsum, messages, levels, prior_rows, tc_feats, f3_feats=None):
    now = datetime.now(timezone.utc)
    temps = [d["value"] for d in weather.get("temperature", {}).get("data", []) if isinstance(d.get("value"), (int, float))]
    hums = [d["value"] for d in weather.get("humidity", {}).get("data", []) if isinstance(d.get("value"), (int, float))]
    rains = [d["max"] for d in weather.get("rainfall", {}).get("data", []) if isinstance(d.get("max"), (int, float))]
    temp_mean = round(sum(temps) / len(temps), 1) if temps else None
    hum_mean = round(sum(hums) / len(hums), 1) if hums else None
    # rhrread rainfall = rainfall in the PAST HOUR per district (not cumulative).
    #   rain_1h    = mean over districts  -> comparable to the single-station backfill
    #   rain_main  = max over districts   (dashboard only)
    #   rain_total = sum over districts   (dashboard only; NOT a model feature)
    #   rain_3h    = this hour + the two previous hourly rain_1h values
    rain_total = round(sum(rains), 1) if rains else None
    rain_main = round(max(rains), 1) if rains else None
    rain_1h = round(sum(rains) / len(rains), 1) if rains else 0.0
    prev_1h = row_near(prior_rows, 60)
    prev_2h = row_near(prior_rows, 120)
    rain_3h = rain_1h + (_f(prev_1h, "rain_1h") if prev_1h else 0.0) + (_f(prev_2h, "rain_1h") if prev_2h else 0.0)

    recent_60 = rows_within(prior_rows, 60)
    ref = prev_1h  # v4.2: deltas against the row ~60 min ago (not "oldest row within 60 min")
    hum_1h_delta = (hum_mean - _f(ref, "hum_mean")) if ref and hum_mean is not None and ref.get("hum_mean") not in ("", None) else 0.0
    temp_1h_delta = (temp_mean - _f(ref, "temp_mean")) if ref and temp_mean is not None and ref.get("temp_mean") not in ("", None) else 0.0

    hk_now = now + HK_OFFSET
    row = {
        "ts": now.isoformat(timespec="seconds"),
        "temp_mean": temp_mean if temp_mean is not None else "",
        "hum_mean": hum_mean if hum_mean is not None else "",
        "rain_total": rain_total if rain_total is not None else "",
        "rain_main": rain_main if rain_main is not None else "",
        "rain_1h": round(rain_1h, 1),
        "rain_3h": round(rain_3h, 1),
        "hum_1h_delta": round(hum_1h_delta, 1),
        "temp_1h_delta": round(temp_1h_delta, 1),
        "hour": hk_now.hour,
        "season": 1 if 5 <= hk_now.month <= 11 else 0,
    }
    for code, flag in WARNSUMS_TO_FLAGS.items():
        row[flag] = 1 if code in warnsum else 0
    for flag in ("w_TC1", "w_TC3", "w_TC8", "w_RAIN_AMBER", "w_RAIN_RED", "w_RAIN_BLACK"):
        row[flag] = 1 if levels.get(flag) else 0
    for key, value in tc_feats.items():
        row[key] = value
    # P6: JTWC distance rate of change (km/h; negative = approaching)
    prev_dist = _f(recent_60[0], "tc_dist_km") if recent_60 and recent_60[0].get("tc_dist_km") not in ("", None) else None
    row["tc_dist_rate"] = round(tc_feats.get("tc_dist_km", 2000) - prev_dist, 1) if prev_dist else 0.0
    for key, value in (f3_feats or f3_features(None)).items():
        row[key] = value
    # v4.2: rolling max of f3_max over the previous ~1h / ~3h rows (is the rain building?)
    prev_3h = row_near(prior_rows, 180)
    f3_prev = [_f(r, "f3_max") for r in (prev_1h,) if r]
    row["f3_max_1h"] = round(max([row["f3_max"]] + f3_prev), 2)
    f3_prev3 = [_f(r, "f3_max") for r in (prev_1h, prev_2h, prev_3h) if r]
    row["f3_max_3h"] = round(max([row["f3_max"]] + f3_prev3), 2)
    row["live"] = 1
    return row, temps, hums, rains


def active_targets(row):
    """Onset targets whose warning is already in force right now."""
    return {t for t, flag in TARGET_FLAG.items() if _f(row, flag) > 0}


def blend_probs(rules_p, ai_p, w_map, active=()):
    """P5: per-target learned blend weight (falls back to rules where AI is absent).

    Models are trained on ONSET (not-in-force -> issued within horizon), so when a
    warning is already in force the question is settled: report 1.0.
    """
    out = {}
    for t in TARGETS:
        r = rules_p.get(t, 0.0)
        if t in active:
            out[t] = 1.0
        elif t in ai_p:
            w = w_map.get(t, 0.0)
            out[t] = round(r * (1 - w) + ai_p[t] * w, 3)
        else:
            out[t] = round(r, 3)
    return out


def active_warning_names(warnsum):
    return [f"{entry.get('name', code)} ({entry.get('actionCode', '')})" for code, entry in warnsum.items()]


def official_changes(prev_state, warnsum, levels, messages):
    changes = []
    prev_warnsum = prev_state.get("warnsum", {}) if prev_state else {}
    prev_levels = prev_state.get("levels", {}) if prev_state else {}
    if not prev_state:
        return changes, False
    for code, entry in warnsum.items():
        name = entry.get("name", code)
        action = entry.get("actionCode", "")
        if code not in prev_warnsum:
            changes.append(f"NEW: {name} ({action})")
        elif prev_warnsum[code].get("actionCode") != action and action:
            changes.append(f"UPDATE: {name} ({action})")
    for code, entry in prev_warnsum.items():
        if code not in warnsum:
            changes.append(f"CANCELED: {entry.get('name', code)}")
    for flag in ("w_TC1", "w_TC3", "w_TC8"):
        cur = 1 if levels.get(flag) else 0
        prev = 1 if prev_levels.get(flag) else 0
        if cur and not prev:
            changes.append(f"ESCALATED: {LEVEL_NAMES[flag]} now in force")
        elif not cur and prev:
            changes.append(f"DOWNGRADED: {LEVEL_NAMES[flag]} no longer in force")
    for flag in ("w_RAIN_AMBER", "w_RAIN_RED", "w_RAIN_BLACK"):
        cur = 1 if levels.get(flag) else 0
        prev = 1 if prev_levels.get(flag) else 0
        if cur and not prev:
            changes.append(f"ESCALATED: {LEVEL_NAMES[flag]} now in force")
        elif not cur and prev:
            changes.append(f"DOWNGRADED: {LEVEL_NAMES[flag]} no longer in force")
    return changes, True


def write_csv(rows, new_row):
    DATA_DIR.mkdir(exist_ok=True)
    all_rows = rows + [new_row]
    with open(SNAPSHOT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)
    return all_rows


def build_history(rows):
    buckets = {}
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    for r in rows:
        ts = datetime.fromisoformat(r["ts"])
        if ts < cutoff:
            continue
        key = (ts + HK_OFFSET).strftime("%Y-%m-%dT%H")
        b = buckets.setdefault(key, {"temps": [], "hums": [], "rain": [], "rain1h": [], "warns": 0})
        if r.get("temp_mean") != "":
            b["temps"].append(_f(r, "temp_mean"))
        if r.get("hum_mean") != "":
            b["hums"].append(_f(r, "hum_mean"))
        b["rain"].append(_f(r, "rain_total"))
        b["rain1h"].append(_f(r, "rain_1h"))
        active = sum(1 for k, v in r.items() if k and k.startswith("w_") and v == "1")
        b["warns"] = max(b["warns"], active)
    out = []
    for key in sorted(buckets):
        b = buckets[key]
        out.append({
            "h": key,
            "temp": round(sum(b["temps"]) / len(b["temps"]), 1) if b["temps"] else None,
            "hum": round(sum(b["hums"]) / len(b["hums"]), 1) if b["hums"] else None,
            "rain": round(max(b["rain"]), 1) if b["rain"] else None,
            "rain1h": round(max(b["rain1h"]), 1) if b["rain1h"] else None,
            "warns": b["warns"],
        })
    return out[-168:]


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def pages_url():
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if "/" in repo:
        owner = repo.split("/")[0]
        return f"https://{owner}.github.io/{repo.split('/')[1]}/"
    return "https://<your-org>.github.io/testing/"


def main():
    # v4.1: one-time data migration, committed by this workflow's "git add data"
    try:
        import migrate_rain_v41
        if migrate_rain_v41.ensure():
            print("[poll] applied v4.1 snapshot migration")
    except Exception as e:
        print(f"[poll] v4.1 migration skipped: {e}")

    weather = http_get(WEATHER_URL)
    warnsum = http_get(WARNSUM_URL)
    messages = weather.get("warningMessage", []) or []
    levels = parse_levels(messages)
    tc_state = jtwc.scan()
    tc_feats = jtwc.snapshot_features(tc_state)

    snap = None
    try:
        snap = gridmod.fetch_snapshot()
    except Exception as e:
        print(f"[poll] F3 grid fetch skipped: {e}")

    rows = load_csv_rows()
    f3_feats = f3_features(snap["leads"]) if snap else None
    row, temps, hums, rains = build_row(weather, warnsum, messages, levels, rows, tc_feats, f3_feats)

    rules_p = rule_probs(row)
    ai_p = predict_ai(row)
    w_map = blend_weights()
    active = active_targets(row)
    probs = blend_probs(rules_p, ai_p, w_map, active)

    v3 = None
    v3_grid_ts = None
    if cnnmod is not None and snap:
        try:
            v3_grid_ts = snap["ts"]
            v3 = cnnmod.predict_frames(snap["leads"])
            if v3:
                probs.update(v3)
        except Exception as e:
            print(f"[poll] v3 cnn skipped: {e}")

    prev_state = None
    if LAST_WARN.exists():
        with open(LAST_WARN, encoding="utf-8-sig") as f:
            prev_state = json.load(f)
    changes, has_prev = official_changes(prev_state, warnsum, levels, messages)

    notify_lines = []
    for c in changes:
        notify_lines.append(f"* {c}")
    if has_prev and changes:
        notify_lines.append("")
        notify_lines.append("Official HKO warning state changed. Details in email.")

    now = datetime.now(timezone.utc)
    notified = load_notified()
    # warnings already in force are covered by the official-change email, not lead alerts
    alert_keys = ai_alert_keys({t: p for t, p in probs.items() if t not in active}, notified, now)
    for k in alert_keys:
        notify_lines.append(f"* {TARGET_LABELS[k]}: probability {probs[k]:.0%} (lead alert)")
        notified[k] = now.isoformat(timespec="seconds")

    nearest_tc = tc_state.get("nearest")
    tc_line = None
    if nearest_tc and nearest_tc["distance_km"] <= jtwc.REPORT_RADIUS_KM:
        n = nearest_tc
        toward = "closing in" if n.get("moving_toward_hk") else "not closing"
        tc_line = (
            f"Tropical cyclone {n['id']}: {n['distance_km']} km from HK, bearing {n['bearing_deg']}°, "
            f"wind {n['wind_kts']} kts, pressure {n['pressure_mb']} mb, 24h forecast distance "
            f"{n['forecast_24h_km']} km ({toward})"
        )

    if notify_lines and os.environ.get("DISABLE_EMAIL") != "1":
        hk_now = now + HK_OFFSET
        body_lines = [f"Testing alert — {hk_now.strftime('%Y-%m-%d %H:%M')} HKT", ""]
        body_lines.extend(notify_lines)
        body_lines.append("")
        body_lines.append("AI nowcast probabilities (next 1-6h):")
        for t in TARGETS:
            body_lines.append(f"  {TARGET_LABELS[t]}: {probs[t]:.0%}")
        if tc_line:
            body_lines.append("")
            body_lines.append(tc_line)
        body_lines.append("")
        body_lines.append(f"Dashboard: {pages_url()}")
        body_lines.append("")
        body_lines.append("Experimental, unofficial prediction. Always check https://www.hko.gov.hk for official warnings.")
        send_email("[Testing] Weather alert", "\n".join(body_lines))

    rows = write_csv(rows, row)
    write_json(LATEST_JSON, {
        "ts": row["ts"],
        "temp_mean": row["temp_mean"],
        "hum_mean": row["hum_mean"],
        "rain_total": row["rain_total"],
        "rain_main": row["rain_main"],
        "rain_1h": row["rain_1h"],
        "rain_3h": row["rain_3h"],
        "active_warnings": active_warning_names(warnsum),
        "special_tips": weather.get("specialWxTips", []),
        "predictions": probs,
        "v3_grid_ts": v3_grid_ts,
        "official_levels": {k: v for k, v in LEVEL_NAMES.items() if levels.get(k)},
        "tc": tc_state.get("nearest"),
        "tc_scanned": tc_state.get("scanned", []),
        "blend_ai_weight": {t: round(w_map.get(t, 0.0), 2) for t in TARGETS},
        "alert_threshold": alert_thresholds(),
        "in_force": sorted(active),
    })
    metrics_path = ROOT / "model" / "metrics.json"
    if metrics_path.exists():
        write_json(SITE_DATA_DIR / "metrics.json", json.loads(metrics_path.read_text(encoding="utf-8")))
    write_json(HISTORY_JSON, build_history(rows))
    write_json(SITE_DATA_DIR / "latest.json", json.loads(LATEST_JSON.read_text(encoding="utf-8")))
    write_json(SITE_DATA_DIR / "history.json", json.loads(HISTORY_JSON.read_text(encoding="utf-8")))

    STATE_DIR.mkdir(exist_ok=True)
    write_json(LAST_WARN, {"warnsum": warnsum, "levels": levels})
    save_notified(notified)

    print(f"[poll] {row['ts']} temp={row['temp_mean']} hum={row['hum_mean']} rain={row['rain_total']}mm")
    print(f"[poll] probs: " + ", ".join(f"{t}={p}" for t, p in probs.items()))
    if tc_line:
        print(f"[poll] {tc_line}")


if __name__ == "__main__":
    main()

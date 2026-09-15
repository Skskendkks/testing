# Rain inputs are mm of mean past-hour rainfall (rain_1h) and the last-3-hour
# sum (rain_3h) — see features.py. HKO issues Amber at ~30 mm/h and Red at
# ~50 mm/h over a wide area, so thresholds below are set against those.
def rule_probs(row):
    temp = float(row.get("temp_mean", 0.0) or 0.0)
    hum = float(row.get("hum_mean", 0.0) or 0.0)
    rain_total = float(row.get("rain_total", 0.0) or 0.0)
    rain_1h = float(row.get("rain_1h", 0.0) or 0.0)
    rain_3h = float(row.get("rain_3h", 0.0) or 0.0)
    hum_delta = float(row.get("hum_1h_delta", 0.0) or 0.0)
    hour = int(row.get("hour", 0.0) or 0.0)
    season = int(row.get("season", 0.0) or 0.0)
    w_wts = int(row.get("w_WTS", 0.0) or 0.0)
    w_tc = int(row.get("w_TCSGNL", 0.0) or 0.0)
    w_tc3 = int(row.get("w_TC3", 0.0) or 0.0)
    w_amber = int(row.get("w_RAIN_AMBER", 0.0) or 0.0)
    w_red = int(row.get("w_RAIN_RED", 0.0) or 0.0)
    tc_dist = float(row.get("tc_dist_km", 2000) or 2000)
    tc_wind = float(row.get("tc_wind_kts", 0.0) or 0.0)
    tc_toward = int(row.get("tc_trend_toward", 0.0) or 0.0)

    def clip(p):
        return max(0.0, min(0.95, p))

    p_rain = 0.10
    p_rain += 0.25 * (rain_1h > 0.5)
    p_rain += 0.25 * (rain_1h > 3.0)
    p_rain += 0.20 * (hum > 90)
    p_rain += 0.10 * (hum_delta > 2)
    p_rain += 0.15 * w_wts

    p_amber = 0.05
    p_amber += 0.20 * (rain_1h > 10)
    p_amber += 0.20 * (rain_1h > 20)
    p_amber += 0.15 * (rain_3h > 30)
    p_amber += 0.15 * (hum > 92)
    p_amber += 0.20 * w_wts
    p_amber += 0.30 * w_amber   # only matters for display; onset rows exclude this

    p_red = 0.02
    p_red += 0.20 * (rain_1h > 20)
    p_red += 0.20 * (rain_1h > 35)
    p_red += 0.15 * (rain_3h > 60)
    p_red += 0.25 * w_amber
    p_red += 0.35 * w_red

    p_tc3 = 0.04 if season else 0.01
    if tc_dist < 2500:
        p_tc3 += 0.20
    if tc_dist < 1200:
        p_tc3 += 0.10
    if tc_dist < 700:
        p_tc3 += 0.15
    if tc_dist < 400:
        p_tc3 += 0.20
    if tc_wind >= 50:
        p_tc3 += 0.15
    if tc_wind >= 64:
        p_tc3 += 0.10
    if tc_toward and tc_dist < 1200:
        p_tc3 += 0.10
    p_tc3 += 0.30 * w_tc
    p_tc3 += 0.35 * w_tc3

    return {
        "rain_1h": clip(p_rain),
        "amber_3h": clip(p_amber),
        "red_3h": clip(p_red),
        "tc3_6h": clip(p_tc3),
    }

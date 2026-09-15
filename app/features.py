import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
WEIGHTS_JSON = MODEL_DIR / "weights.json"
TREES_JSON = MODEL_DIR / "trees.json"
BLEND_JSON = MODEL_DIR / "blend.json"

# v4: added f3_* (gridded-nowcast scalars), tc_trend_toward, tc_dist_rate.
# v4.1: rain features re-defined so backfilled and live rows mean the same thing
#   rain_1h = rainfall in the past hour (mm; live = mean over districts, backfill = station)
#   rain_3h = sum of the last three hourly rain_1h values
#   rain_total (district sum / cumulative) dropped — its semantics differed between sources.
#   Current warning-state flags added so the model sees what the persistence baseline sees.
# Old model files whose coef length mismatches are ignored at predict time.
FEATURE_COLS = [
    "temp_mean",
    "hum_mean",
    "rain_1h",
    "rain_3h",
    "hum_1h_delta",
    "temp_1h_delta",
    "hour",
    "season",
    "w_WTS",
    "w_TCSGNL",
    "w_TC1",
    "w_TC3",
    "w_RAIN_AMBER",
    "w_RAIN_RED",
    "tc_dist_km",
    "tc_wind_kts",
    "tc_24h_dist_km",
    "tc_trend_toward",
    "tc_dist_rate",
    "f3_max",
    "f3_mean",
    "f3_trend",
    "f3_max_1h",    # v4.2: rolling max of f3_max over the previous ~1h / ~3h of rows
    "f3_max_3h",
    "month_sin",    # v4.2: smooth seasonality derived from ts (replaces the coarse `season` step)
    "month_cos",
]

FEATURE_DEFAULTS = {
    "tc_dist_km": 2000.0,
    "tc_24h_dist_km": 2000.0,
}

TARGETS = ["rain_1h", "amber_3h", "red_3h", "tc3_6h"]

# Warning targets are ONSET targets: "not in force now, issued within the horizon".
# TARGET_FLAG is the flag whose onset is predicted; rows where it is already
# active are excluded from training/eval, and reported as 1.0 (in force) live.
TARGET_FLAG = {"amber_3h": "w_RAIN_AMBER", "red_3h": "w_RAIN_RED", "tc3_6h": "w_TC3"}

# Escalation baseline (replaces plain persistence, which is identically 0 for
# onset targets): "the next-lower warning is already in force".
BASELINE_FLAG = {"amber_3h": "w_WTS", "red_3h": "w_RAIN_AMBER", "tc3_6h": "w_TC1"}

TARGET_LABELS = {
    "rain_1h": "Rain next 1h",
    "amber_3h": "Amber Rainstorm ≤3h",
    "red_3h": "Red Rainstorm ≤3h",
    "tc3_6h": "TC Signal 3+ ≤6h",
}


def sigmoid(x):
    if x <= -700:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _load_json(path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_weights():
    return _load_json(WEIGHTS_JSON)


def load_trees():
    return _load_json(TREES_JSON)


def load_blend():
    return _load_json(BLEND_JSON)


def _month_feats(row):
    try:
        m = int(str(row.get("ts", ""))[5:7])
    except ValueError:
        return 0.0, 0.0
    if not 1 <= m <= 12:
        return 0.0, 0.0
    ang = 2 * math.pi * (m - 1) / 12.0
    return math.sin(ang), math.cos(ang)


def _feat(row, col):
    if col == "month_sin":
        return _month_feats(row)[0]
    if col == "month_cos":
        return _month_feats(row)[1]
    v = row.get(col, None)
    if v in (None, ""):
        return FEATURE_DEFAULTS.get(col, 0.0)
    try:
        value = float(v)
    except (TypeError, ValueError):
        return FEATURE_DEFAULTS.get(col, 0.0)
    return value if math.isfinite(value) else FEATURE_DEFAULTS.get(col, 0.0)


def feature_vector(row):
    return [_feat(row, col) for col in FEATURE_COLS]


def apply_cal(p, cal):
    """Platt calibration: sigmoid(a*p + b), fitted on the validation fold."""
    if not cal:
        return p
    return sigmoid(cal["a"] * p + cal["b"])


def _lr_prob(entry, x):
    """Return a logistic-regression probability from a portable model artifact.

    Older artifacts rounded near-zero standard deviations to ``0.0``. A zero
    standard deviation with a zero coefficient is a constant feature and can
    safely be ignored; a non-zero coefficient would make the artifact invalid.
    """
    try:
        intercept = float(entry["intercept"])
        mean = entry["mean"]
        std = entry["std"]
        coef = entry["coef"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid LR artifact structure") from exc
    if not (len(x) == len(mean) == len(std) == len(coef)):
        raise ValueError("invalid LR artifact feature dimensions")
    if not math.isfinite(intercept):
        raise ValueError("invalid LR artifact intercept")

    z = intercept
    for xi, m, s, c in zip(x, mean, std, coef):
        try:
            xi, m, s, c = float(xi), float(m), float(s), float(c)
        except (TypeError, ValueError) as exc:
            raise ValueError("non-numeric LR artifact value") from exc
        if not all(math.isfinite(v) for v in (xi, m, s, c)):
            raise ValueError("non-finite LR artifact value")
        if abs(s) < 1e-12:
            if abs(c) < 1e-12:
                continue
            raise ValueError("zero LR standard deviation with non-zero coefficient")
        z += ((xi - m) / s) * c
    return sigmoid(z)


def _tree_raw(tree, x):
    """Pure-Python traversal of one exported HistGradientBoosting tree."""
    feat, thr, left, right, leaf, val = (
        tree["feat"], tree["thr"], tree["left"], tree["right"], tree["leaf"], tree["val"],
    )
    i = 0
    while not leaf[i]:
        i = left[i] if x[feat[i]] <= thr[i] else right[i]
    return val[i]


def _trees_prob(entry, x):
    raw = entry["baseline"]
    for tree in entry["trees"]:
        raw += _tree_raw(tree, x)
    return sigmoid(raw)


def predict_ai(row):
    """Calibrated probability per target from the shipped model (trees preferred, LR fallback).

    A target only appears in trees.json / weights.json if it beat the persistence
    baseline on time-ordered validation (train.py enforces this), so presence == shipped.
    """
    trees = load_trees() or {}
    weights = load_weights() or {}
    x = feature_vector(row)
    out = {}
    for target in TARGETS:
        entry = trees.get(target)
        if isinstance(entry, dict) and entry.get("trees"):
            if entry.get("n_features", len(FEATURE_COLS)) != len(FEATURE_COLS):
                continue
            out[target] = apply_cal(_trees_prob(entry, x), entry.get("cal"))
            continue
        entry = weights.get(target)
        if not isinstance(entry, dict) or "coef" not in entry:
            continue
        if len(entry["coef"]) != len(FEATURE_COLS):
            continue  # stale model trained on an older feature set
        try:
            probability = apply_cal(_lr_prob(entry, x), entry.get("cal"))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            print(f"[model] skipped invalid LR artifact for {target}: {exc}")
            continue
        if math.isfinite(probability) and 0.0 <= probability <= 1.0:
            out[target] = probability
        else:
            print(f"[model] skipped non-finite LR prediction for {target}")
    return out


# Backwards-compatible alias
predict_from_row = predict_ai


def blend_weight(weights):
    """Legacy global ramp — fallback when model/blend.json is absent."""
    if not weights or "meta" not in weights:
        return 0.0
    trained = [t for t in TARGETS if weights.get(t, {}).get("coef")]
    n = weights["meta"].get("n_total", 0)
    if not trained:
        return 0.0
    return max(0.2, min(0.5, 0.2 + (n - 48) / 952 * 0.3))


def blend_weights():
    """Per-target AI blend weight learned on the validation fold (P5).

    Returns {target: w} with w in [0, 1]. Falls back to the legacy global ramp.
    """
    blend = load_blend()
    if blend and isinstance(blend.get("targets"), dict):
        return {t: float(blend["targets"].get(t, 0.0)) for t in TARGETS}
    legacy = blend_weight(load_weights())
    return {t: legacy for t in TARGETS}

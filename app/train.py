"""v4 trainer: LR vs gradient-boosted trees, time-ordered eval, calibration, blend weights.

Outputs (all pure-Python-consumable at inference; see features.py):
  model/weights.json  — LR coefficients (target present only if LR chosen AND ships)
  model/trees.json    — exported HGB trees (target present only if trees chosen AND ships)
  model/blend.json    — per-target rules-vs-AI blend weight (P5)
  model/metrics.json  — per-target PR-AUC / Brier vs persistence & climatology (P4)

v4.1 changes (methodology fixes):
  * Warning targets are ONSET labels: rows where the warning is already in
    force are excluded; label = 1 if it is issued within the horizon.
  * Time-ordered THREE-WAY split: train (60%) / calibration (20%) / test (20%).
    Platt calibration and the rules-vs-AI blend weight are fitted on the
    calibration fold; every reported metric and the ship decision use the
    untouched test fold.
  * Baseline is "escalation" (next-lower warning in force) instead of plain
    persistence, which is identically zero for onset targets.
  * Metrics are also reported on live-polled rows only (rows with TC/F3
    features), because backfilled rows lack rain/F3/TC features.

A target "ships" only if the calibrated model beats the baseline on BOTH PR-AUC
and Brier on the test fold. Otherwise that target stays rules-only (blend weight 0).
"""

import csv
import json
import os
import sys
from bisect import bisect_right
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import (
    BASELINE_FLAG, FEATURE_COLS, TARGET_FLAG, TARGETS, apply_cal, feature_vector, sigmoid,
    _trees_prob,
)
from rules import rule_probs

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
SNAPSHOT_CSV = DATA_DIR / "snapshots.csv"
WEIGHTS_JSON = MODEL_DIR / "weights.json"
TREES_JSON = MODEL_DIR / "trees.json"
BLEND_JSON = MODEL_DIR / "blend.json"
METRICS_JSON = MODEL_DIR / "metrics.json"

# True horizons (hours), matched to target names. Labels are computed by
# timestamp (bisect on ts), so polling cadence never changes the window (P4).
HORIZON_HOURS = {
    "rain_1h": 1.0,
    "amber_3h": 3.0,
    "red_3h": 3.0,
    "tc3_6h": 6.0,
}

FLAG = TARGET_FLAG

MIN_ROWS = 48
MIN_POSITIVES = 4       # need a few in every fold
MIN_CAL_POSITIVES = 5   # minimum calibration-fold positives to fit Platt calibration
TRAIN_FRAC = 0.6        # time-ordered: first 60% train, next 20% calibration, last 20% test
CAL_FRAC = 0.2
RAIN_MM = 1.0           # rain_1h above this counts as "rain"
EPS = 1e-6


MIN_F3_ROWS = 2000      # switch to F3-only training once this many rows carry real F3 data


def has_f3(row):
    return str(row.get("f3_ok", "")) == "1"


def select_training_rows(rows):
    """F3-only mode: once enough rows carry real F3 scalars, train/evaluate on those
    only, so the model is not dominated by 25 years of rows with f3_*=0 and rain=0.
    Override with F3_ONLY=0 / F3_ONLY=1 in the environment."""
    n_f3 = sum(1 for r in rows if has_f3(r))
    force = os.environ.get("F3_ONLY")
    use = (force == "1") if force in ("0", "1") else n_f3 >= MIN_F3_ROWS
    if use and n_f3 > 0:
        return [r for r in rows if has_f3(r)], {"mode": "f3_only", "n_f3_rows": n_f3}
    return rows, {"mode": "all_rows", "n_f3_rows": n_f3}


def is_live_row(row):
    """Live-polled rows are flagged live=1 (v4.2); older files: TC features non-blank."""
    live = str(row.get("live", "")).strip()
    if live in ("0", "1"):
        return live == "1"
    return row.get("tc_dist_km", "") not in ("", None)


ALERT_MIN_PRECISION = 0.30   # lead alert fires when P(onset) >= threshold with >=30% precision on the cal fold
ALERT_MIN_HITS = 3


def alert_threshold(p_cal, y_cal):
    """Lowest probability threshold whose precision on the calibration fold is >= ALERT_MIN_PRECISION.
    Returns None if no threshold reaches it (alerts stay off for that target)."""
    pairs = sorted(zip(p_cal, y_cal), reverse=True)
    best = None
    tp = n = 0
    for p, t in pairs:
        n += 1
        tp += int(t)
        if tp >= ALERT_MIN_HITS and tp / n >= ALERT_MIN_PRECISION:
            best = p
    return round(float(best), 4) if best is not None else None


def _f(row, key, default=0.0):
    v = row.get(key, None)
    if v in (None, ""):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_rows():
    if not SNAPSHOT_CSV.exists():
        return []
    with open(SNAPSHOT_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        try:
            r["_dt"] = datetime.fromisoformat(r["ts"])
        except (ValueError, KeyError):
            continue
        out.append(r)
    out.sort(key=lambda r: r["_dt"])
    return out


def build_samples(rows, target):
    """Timestamp-based lookahead labels. Returns (X, y, row_indices)."""
    ts = [r["_dt"] for r in rows]
    horizon = timedelta(hours=HORIZON_HOURS[target])
    last = ts[-1]
    X, y, idx = [], [], []
    for i, r in enumerate(rows):
        if last < r["_dt"] + horizon * 0.8:
            break  # tail rows lack lookahead coverage
        j0 = bisect_right(ts, r["_dt"])
        j1 = bisect_right(ts, r["_dt"] + horizon)
        future = rows[j0:j1]
        if not future:
            continue
        if target == "rain_1h":
            # rain_1h of a future row = rainfall in the hour before that row
            label = 1 if max(_f(fr, "rain_1h") for fr in future) > RAIN_MM else 0
        else:
            if _f(r, FLAG[target]) > 0:
                continue  # onset target: warning already in force — not a forecast question
            label = 1 if any(_f(fr, FLAG[target]) > 0 for fr in future) else 0
        X.append(feature_vector(r))
        y.append(label)
        idx.append(i)
    return X, y, idx


def baseline_scores(rows, idx, target):
    """Naive baseline the model must beat.

    rain_1h: persistence (raining now -> raining next hour).
    warning onset targets: escalation — the next-lower warning is in force now
    (WTS -> Amber, Amber -> Red, TC1 -> TC3). Plain persistence is always 0 for
    onset rows, so it is not a meaningful comparator.
    """
    out = []
    for i in idx:
        r = rows[i]
        if target == "rain_1h":
            out.append(1.0 if _f(r, "rain_1h") > RAIN_MM else 0.0)
        else:
            out.append(1.0 if _f(r, BASELINE_FLAG[target]) > 0 else 0.0)
    return out


def baseline_name(target):
    return "persistence" if target == "rain_1h" else f"escalation ({BASELINE_FLAG[target]} in force)"


def fit_platt(p_val, y_val):
    """1-D logistic map on raw probabilities; None if not enough signal."""
    from sklearn.linear_model import LogisticRegression
    if sum(y_val) < MIN_CAL_POSITIVES or sum(y_val) == len(y_val):
        return None
    lr = LogisticRegression(max_iter=1000)
    lr.fit([[p] for p in p_val], y_val)
    return {"a": float(lr.coef_[0][0]), "b": float(lr.intercept_[0])}


def export_hgb(clf):
    """Export HistGradientBoostingClassifier to plain lists for pure-Python traversal."""
    import numpy as np
    trees = []
    for stage in clf._predictors:
        for pred in stage:
            nodes = pred.nodes
            names = nodes.dtype.names
            thr_field = "num_threshold" if "num_threshold" in names else "threshold"
            trees.append({
                "feat": [int(v) for v in nodes["feature_idx"]],
                "thr": [float(v) for v in nodes[thr_field]],
                "left": [int(v) for v in nodes["left"]],
                "right": [int(v) for v in nodes["right"]],
                "leaf": [bool(v) for v in nodes["is_leaf"]],
                "val": [float(v) for v in nodes["value"]],
            })
    baseline = float(np.asarray(clf._baseline_prediction).ravel()[0])
    return {"baseline": baseline, "trees": trees, "n_features": len(FEATURE_COLS)}


def parity_ok(entry, clf, X_check):
    """Pure-Python traversal must reproduce sklearn's predict_proba."""
    ps = clf.predict_proba(X_check)[:, 1]
    for x, p_ref in zip(X_check, ps):
        if abs(_trees_prob(entry, list(x)) - float(p_ref)) > 1e-4:
            return False
    return True


def blend_search(rules_p, ai_p, y):
    """Grid-search w minimizing Brier of (1-w)*rules + w*ai on the val fold (P5)."""
    best_w, best_b = 0.0, None
    for step in range(11):
        w = step / 10.0
        b = sum(((1 - w) * r + w * a - t) ** 2 for r, a, t in zip(rules_p, ai_p, y)) / len(y)
        if best_b is None or b < best_b - 1e-12:
            best_w, best_b = w, b
    return best_w, best_b


def main():
    try:
        import migrate_rain_v41
        if migrate_rain_v41.ensure():
            print("[train] applied v4.1 snapshot migration")
    except Exception as e:
        print(f"[train] v4.1 migration skipped: {e}")
    rows, row_mode = select_training_rows(load_rows())
    n = len(rows)
    generated = datetime.now().isoformat(timespec="seconds")
    meta = {
        "artifact_version": 2,
        "feature_cols": list(FEATURE_COLS),
        "n_total": n,
        "generated": generated,
    }
    weights_out = {"meta": dict(meta)}
    trees_out = {"meta": dict(meta)}
    blend_out = {"meta": dict(meta), "targets": {t: 0.0 for t in TARGETS}, "alert_threshold": {}}
    metrics = {"n_total": n, "generated": generated, "rows": row_mode, "targets": {}}
    if rows:
        metrics["train_range"] = [rows[0]["ts"], rows[-1]["ts"]]

    MODEL_DIR.mkdir(exist_ok=True)
    if n < MIN_ROWS:
        weights_out["meta"]["note"] = f"only {n} rows; need {MIN_ROWS} to train — rules-only mode"
        _write_all(weights_out, trees_out, blend_out, metrics)
        print(f"[train] skipped: {n} rows < {MIN_ROWS}")
        return

    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score

    def brier(y, p):
        y = np.asarray(y, dtype=float)
        p = np.asarray(p, dtype=float)
        return float(np.mean((p - y) ** 2))

    for target in TARGETS:
        X, y, idx = build_samples(rows, target)
        report = {"n_samples": len(y), "n_pos": int(sum(y))}
        metrics["targets"][target] = report
        if len(y) < MIN_ROWS or sum(y) < MIN_POSITIVES:
            report["status"] = f"skipped ({sum(y)} positives < {MIN_POSITIVES} or too few rows)"
            print(f"[train] {target}: {report['status']}")
            continue

        # time-ordered three-way split: train / calibration / test
        n_s = len(y)
        i_tr = max(1, int(n_s * TRAIN_FRAC))
        i_cal = max(i_tr + 1, int(n_s * (TRAIN_FRAC + CAL_FRAC)))
        X_tr, X_ca, X_te = np.array(X[:i_tr]), np.array(X[i_tr:i_cal]), np.array(X[i_cal:])
        y_tr, y_ca, y_te = np.array(y[:i_tr]), np.array(y[i_tr:i_cal]), np.array(y[i_cal:])
        cal_idx, test_idx = idx[i_tr:i_cal], idx[i_cal:]
        report["n_train"], report["n_train_pos"] = len(y_tr), int(y_tr.sum())
        report["n_cal"], report["n_cal_pos"] = len(y_ca), int(y_ca.sum())
        report["n_test"], report["n_test_pos"] = len(y_te), int(y_te.sum())
        if test_idx:
            report["test_range"] = [rows[test_idx[0]]["ts"], rows[test_idx[-1]]["ts"]]
        if (y_tr.sum() == 0 or y_ca.sum() == 0 or y_te.sum() == 0
                or y_tr.sum() == len(y_tr) or y_te.sum() == len(y_te)):
            report["status"] = "skipped (a fold has a single class — need more positives)"
            print(f"[train] {target}: {report['status']}")
            continue

        # --- candidate 1: standardized Logistic Regression (existing baseline model)
        mean = X_tr.mean(axis=0)
        std = X_tr.std(axis=0)
        std[std < 1e-6] = 1e-6
        lr = LogisticRegression(max_iter=2000, class_weight="balanced")
        lr.fit((X_tr - mean) / std, y_tr)
        p_lr_ca = lr.predict_proba((X_ca - mean) / std)[:, 1]
        p_lr_te = lr.predict_proba((X_te - mean) / std)[:, 1]

        # --- candidate 2: gradient-boosted trees (P3)
        w_pos = len(y_tr) / (2.0 * max(1, y_tr.sum()))
        w_neg = len(y_tr) / (2.0 * max(1, (len(y_tr) - y_tr.sum())))
        sw = np.where(y_tr == 1, w_pos, w_neg)
        hgb = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.08, min_samples_leaf=10,
            early_stopping=False, random_state=0,
        )
        hgb.fit(X_tr, y_tr, sample_weight=sw)
        p_hgb_ca = hgb.predict_proba(X_ca)[:, 1]
        p_hgb_te = hgb.predict_proba(X_te)[:, 1]

        # --- model selection on the CALIBRATION fold (never on test)
        ap_lr = float(average_precision_score(y_ca, p_lr_ca))
        ap_hgb = float(average_precision_score(y_ca, p_hgb_ca))
        report["lr"] = {"cal_pr_auc": round(ap_lr, 4), "cal_brier": round(brier(y_ca, p_lr_ca), 4),
                        "test_pr_auc": round(float(average_precision_score(y_te, p_lr_te)), 4)}
        report["trees"] = {"cal_pr_auc": round(ap_hgb, 4), "cal_brier": round(brier(y_ca, p_hgb_ca), 4),
                           "test_pr_auc": round(float(average_precision_score(y_te, p_hgb_te)), 4)}

        use_trees = ap_hgb > ap_lr or (ap_hgb == ap_lr and brier(y_ca, p_hgb_ca) < brier(y_ca, p_lr_ca))
        entry = None
        if use_trees:
            try:
                entry = export_hgb(hgb)
                if not parity_ok(entry, hgb, X_ca[: min(20, len(X_ca))]):
                    print(f"[train] {target}: tree export parity failed — falling back to LR")
                    entry, use_trees = None, False
            except Exception as e:
                print(f"[train] {target}: tree export failed ({e}) — falling back to LR")
                entry, use_trees = None, False
        p_raw_ca = p_hgb_ca if use_trees else p_lr_ca
        p_raw_te = p_hgb_te if use_trees else p_lr_te

        # --- Platt calibration fitted on the calibration fold, applied to test (P4)
        cal = fit_platt(list(p_raw_ca), list(y_ca))
        p_cal_ca = np.array([apply_cal(float(p), cal) for p in p_raw_ca])
        if brier(y_ca, p_cal_ca) > brier(y_ca, p_raw_ca):
            cal, p_cal_ca = None, p_raw_ca  # calibration hurt (tiny fold) — drop it
        p_cal_te = np.array([apply_cal(float(p), cal) for p in p_raw_te])

        # --- baselines on the TEST fold (P4): escalation/persistence + climatology
        p_base = np.array(baseline_scores(rows, test_idx, target))
        p_clim = np.full(len(y_te), float(y_tr.mean()))
        ap_model = float(average_precision_score(y_te, p_cal_te))
        ap_base = float(average_precision_score(y_te, p_base))
        b_model, b_base, b_clim = brier(y_te, p_cal_te), brier(y_te, p_base), brier(y_te, p_clim)
        report["model"] = "trees" if use_trees else "lr"
        report["calibrated"] = cal is not None
        report["pr_auc"] = round(ap_model, 4)
        report["brier"] = round(b_model, 4)
        report["baseline"] = {"name": baseline_name(target),
                              "pr_auc": round(ap_base, 4), "brier": round(b_base, 4)}
        report["climatology"] = {"brier": round(b_clim, 4)}

        # --- same metrics on live-polled test rows only (the distribution served in production)
        live_mask = np.array([is_live_row(rows[i]) for i in test_idx], dtype=bool)
        live = {"n": int(live_mask.sum()), "n_pos": int(y_te[live_mask].sum()) if live_mask.any() else 0}
        if live["n_pos"] > 0 and live["n_pos"] < live["n"]:
            live["pr_auc"] = round(float(average_precision_score(y_te[live_mask], p_cal_te[live_mask])), 4)
            live["brier"] = round(brier(y_te[live_mask], p_cal_te[live_mask]), 4)
            live["baseline_pr_auc"] = round(float(average_precision_score(y_te[live_mask], p_base[live_mask])), 4)
            live["baseline_brier"] = round(brier(y_te[live_mask], p_base[live_mask]), 4)
        else:
            live["note"] = "too few live positives in the test fold to score"
        report["live_only"] = live

        # Ship gate: beat the naive baseline on PR-AUC and Brier, AND beat
        # climatology on Brier (a binary baseline has a poor Brier by construction,
        # so climatology is the real calibration hurdle for rare events).
        ships = (
            ap_model >= ap_base - 1e-9 and b_model <= b_base + 1e-9
            and (ap_model > ap_base + EPS or b_model < b_base - EPS)
            and b_model < b_clim - EPS
        )
        report["ships"] = bool(ships)
        if not ships:
            report["status"] = "rules-only (did not beat baseline/climatology on the test fold)"
            print(f"[train] {target}: NOT shipped — PR-AUC {ap_model:.3f} vs base {ap_base:.3f}, "
                  f"Brier {b_model:.4f} vs base {b_base:.4f} / clim {b_clim:.4f}")
            continue

        # --- learned blend weight on the CALIBRATION fold (P5); its test Brier reported honestly
        rules_ca = [rule_probs(rows[i])[target] for i in cal_idx]
        rules_te = [rule_probs(rows[i])[target] for i in test_idx]
        w_blend, _ = blend_search(rules_ca, list(p_cal_ca), list(y_ca))
        b_blend_te = sum(((1 - w_blend) * r + w_blend * a - t) ** 2
                         for r, a, t in zip(rules_te, p_cal_te, y_te)) / len(y_te)
        blend_out["targets"][target] = w_blend
        report["blend_w"] = w_blend
        # lead-alert threshold chosen on the calibration fold (P(onset) after blending)
        blended_ca = [(1 - w_blend) * r + w_blend * a for r, a in zip(rules_ca, p_cal_ca)]
        thr = alert_threshold(blended_ca, list(y_ca))
        blend_out["alert_threshold"][target] = thr
        report["alert_threshold"] = thr
        # per-year breakdown on the test fold — is the score stable or carried by one season?
        by_year = {}
        for i, (yt, pc, pb) in enumerate(zip(y_te, p_cal_te, p_base)):
            yr = rows[test_idx[i]]["ts"][:4]
            b = by_year.setdefault(yr, {"n": 0, "n_pos": 0, "_y": [], "_p": []})
            b["n"] += 1; b["n_pos"] += int(yt); b["_y"].append(yt); b["_p"].append(pc)
        for yr, b in by_year.items():
            if 0 < b["n_pos"] < b["n"]:
                b["pr_auc"] = round(float(average_precision_score(b.pop("_y"), b.pop("_p"))), 4)
            else:
                b.pop("_y"); b.pop("_p")
        report["by_year"] = by_year
        report["blend_brier_test"] = round(b_blend_te, 4)
        report["rules_only_brier_test"] = round(brier(y_te, np.array(rules_te)), 4)

        if use_trees:
            entry["cal"] = cal
            entry["n_pos"] = int(sum(y))
            trees_out[target] = entry
        else:
            weights_out[target] = {
                "intercept": float(lr.intercept_[0]),
                "coef": [float(c) for c in lr.coef_[0]],
                # Preserve full precision: rounding a protected 1e-6 standard
                # deviation to four decimal places previously created 0.0 and
                # broke production inference.
                "mean": [float(m) for m in mean],
                "std": [float(s) for s in std],
                "n_features": len(FEATURE_COLS),
                "cal": cal,
                "n_pos": int(sum(y)),
                "n_neg": int(len(y) - sum(y)),
            }
        report["status"] = "shipped"
        print(f"[train] {target}: shipped {report['model']} — test PR-AUC {ap_model:.3f} "
              f"(base {ap_base:.3f}), Brier {b_model:.4f} (base {b_base:.4f}), blend w={w_blend}")

    _write_all(weights_out, trees_out, blend_out, metrics)
    shipped = [t for t in TARGETS if metrics["targets"].get(t, {}).get("ships")]
    print(f"[train] done: {len(shipped)}/{len(TARGETS)} targets shipped: {shipped or 'none'}")


def _write_all(weights_out, trees_out, blend_out, metrics):
    WEIGHTS_JSON.write_text(json.dumps(weights_out, indent=2), encoding="utf-8")
    TREES_JSON.write_text(json.dumps(trees_out), encoding="utf-8")
    BLEND_JSON.write_text(json.dumps(blend_out, indent=2), encoding="utf-8")
    METRICS_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import features


class PortableModelArtifactTests(unittest.TestCase):
    def setUp(self):
        self.row = {name: 0.0 for name in features.FEATURE_COLS}
        self.row["temp_mean"] = 28.0
        self.row["hum_mean"] = 80.0

    def lr_entry(self, *, std=None, coef=None):
        n = len(features.FEATURE_COLS)
        return {
            "intercept": 0.0,
            "mean": [0.0] * n,
            "std": [1.0] * n if std is None else std,
            "coef": [0.0] * n if coef is None else coef,
        }

    def test_legacy_zero_std_with_zero_coef_is_safe(self):
        entry = self.lr_entry()
        entry["std"][2] = 0.0
        entry["coef"][2] = 0.0
        with patch.object(features, "load_trees", return_value={}), patch.object(
            features, "load_weights", return_value={"amber_3h": entry}
        ):
            out = features.predict_ai(self.row)
        self.assertIn("amber_3h", out)
        self.assertTrue(math.isfinite(out["amber_3h"]))
        self.assertGreaterEqual(out["amber_3h"], 0.0)
        self.assertLessEqual(out["amber_3h"], 1.0)

    def test_invalid_zero_std_with_nonzero_coef_is_skipped(self):
        entry = self.lr_entry()
        entry["std"][2] = 0.0
        entry["coef"][2] = 0.2
        with patch.object(features, "load_trees", return_value={}), patch.object(
            features, "load_weights", return_value={"amber_3h": entry}
        ):
            out = features.predict_ai(self.row)
        self.assertNotIn("amber_3h", out)

    def test_non_finite_input_is_replaced_by_feature_default(self):
        row = dict(self.row, temp_mean="NaN")
        self.assertEqual(features.feature_vector(row)[0], 0.0)



class JtwcWarningParserTests(unittest.TestCase):
    SAMPLE = """WTPN31 PGTW 121500
1. TROPICAL STORM 14W (SAMPLE) WARNING NR 019
   WARNING POSITION:
   121200Z --- NEAR 19.2N 117.8E
     MOVEMENT PAST SIX HOURS - 300 DEGREES AT 09 KTS
   PRESENT WIND DISTRIBUTION:
   MAX SUSTAINED WINDS - 050 KT, GUSTS 065 KT
   FORECASTS:
   12 HRS, VALID AT:
   130000Z --- 19.9N 116.4E
   MAX SUSTAINED WINDS - 055 KT, GUSTS 070 KT
   24 HRS, VALID AT:
   131200Z --- 20.8N 115.0E
   MAX SUSTAINED WINDS - 060 KT, GUSTS 075 KT
"""

    def test_parse_warning_current_and_forecasts(self):
        import jtwc
        cur, fc = jtwc.parse_warning(self.SAMPLE, "WP14")
        self.assertEqual((cur["lat"], cur["lon"], cur["wind"]), (19.2, 117.8, 50))
        self.assertEqual(sorted(fc), [12, 24])
        self.assertEqual(fc[24]["wind"], 60)
        info = jtwc.storm_info({"id": "WP14"}, cur, fc)
        self.assertLess(info["distance_km"], 600)
        self.assertTrue(info["moving_toward_hk"])
        feats = jtwc.snapshot_features({"nearest": info})
        self.assertEqual(feats["tc_trend_toward"], 1)
        self.assertLess(feats["tc_dist_km"], 2000)

    def test_rss_regex_skips_invests(self):
        import jtwc
        rss = "x products/wp9826web.txt y products/wp1426web.txt z products/ep1526web.txt"
        ids = [m.group(1) for m in jtwc._RSS_WP_RE.finditer(rss)]
        self.assertEqual(ids, ["98", "14"])


class TrainerLogicTests(unittest.TestCase):
    def _rows(self, flags):
        from datetime import datetime, timedelta, timezone
        t0 = datetime(2026, 6, 1, tzinfo=timezone.utc)
        rows = []
        for i, fl in enumerate(flags):
            r = {name: 0.0 for name in features.FEATURE_COLS}
            r["ts"] = (t0 + timedelta(hours=i)).isoformat()
            r["_dt"] = t0 + timedelta(hours=i)
            r["w_RAIN_AMBER"] = fl
            rows.append(r)
        return rows

    def test_onset_labels_skip_rows_already_in_force(self):
        import train
        # amber in force at hours 4-5; horizon 3h -> hours 1,2,3 are onset positives, 4,5 excluded
        rows = self._rows([0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
        X, y, idx = train.build_samples(rows, "amber_3h")
        self.assertNotIn(4, idx)
        self.assertNotIn(5, idx)
        lab = dict(zip(idx, y))
        self.assertEqual([lab[1], lab[2], lab[3]], [1, 1, 1])
        self.assertEqual(lab[0], 0)

    def test_alert_threshold_respects_precision_floor(self):
        import train
        p = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
        y = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        # lowest threshold whose cumulative precision stays >= 0.30: all 10 rows (3/10)
        self.assertEqual(train.alert_threshold(p, y), 0.05)
        p2 = p + [0.04] * 5                     # extra negatives push the tail below 0.30
        self.assertEqual(train.alert_threshold(p2, y + [0] * 5), 0.05)
        self.assertIsNone(train.alert_threshold(p, [1] + [0] * 9))   # < ALERT_MIN_HITS
        self.assertIsNone(train.alert_threshold(p, [0] * 10))


if __name__ == "__main__":
    unittest.main()

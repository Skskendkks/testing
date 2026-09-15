# Testing

<https://skskendkks.github.io/testing/site/index.html>

Unofficial Hong Kong weather-warning nowcast. Polls HKO Open Data hourly, predicts
warning probability (rain, Amber/Red Rainstorm, Typhoon Signal 3+) using a hybrid of
hand-written rules and a nightly-retrained logistic-regression model, emails you on
official warning changes and AI lead alerts, and renders a static dashboard.

V2 adds JTWC tropical-cyclone best-track/forecast ingestion (distance, bearing, wind,
pressure, 24h forecast distance from HK) as model features and dashboard/email content.

V3 adds a CNN rain nowcast on HKO's gridded rainfall-nowcast data (121×121 grid at
~2km, 4 lead times, updated every 12 min): a small pure-NumPy conv net trained on
archived grids from the data.gov.hk historical archive predicts heavy-rain
probability ~2h ahead. Radar imagery is NOT exposed via HKO Open Data (no archive),
so the gridded nowcast is used instead of radar images.

## v4.1 / v4.2 — methodology fixes (2026-09)

An audit of the v4 pipeline found the reported accuracy could not be trusted:
backfilled rows (99.8% of training data) had no rain / F3 / cyclone features,
`rain_*` meant different things in backfilled vs live rows, labels rewarded
"warning stays in force" rather than "warning gets issued", and calibration +
blend weights were fitted on the same fold they were scored on. Changes:

| Area | Now |
|---|---|
| Labels | **Onset**: rows with the warning already in force are excluded; label = issued within the horizon. Live: an in-force warning is reported as 1.0 / "IN FORCE" and never triggers a lead alert. |
| Split | Time-ordered **train 60 / calibration 20 / test 20**. Platt calibration, LR-vs-trees choice, blend weight and alert threshold are all fitted on the calibration fold; `model/metrics.json` reports the untouched test fold, plus `by_year` and `live_only` breakdowns. |
| Baseline | "Escalation" (next-lower warning in force: WTS→Amber, Amber→Red, TC1→TC3). A target ships only if it beats that on PR-AUC and Brier **and** beats climatology on Brier. |
| Rain features | `rain_1h` = mean past-hour rainfall (mm) over rhrread districts / station precip in backfill; `rain_3h` = last three `rain_1h`. `rain_total` is dashboard-only. |
| Cyclones | Live source fixed: NHC ATCF has no Western Pacific files, so `jtwc.py` now reads the JTWC RSS + warning text. History from **IBTrACS** via `app/backfill_tc.py` (persistence-extrapolated 24 h distance, no future leakage). |
| F3 | `backfill.py` also writes `f3_max/mean/trend` + rolling `f3_max_1h/3h` into `snapshots.csv` (`f3_ok=1`); `train.py` switches to F3-only rows once ≥2000 exist (`F3_ONLY=0/1` overrides). |
| Alerts | Per-target threshold learned on the calibration fold (precision ≥ 30%), stored in `model/blend.json`; targets without a reliable threshold have alerts off. |
| Data migration | `app/migrate_rain_v41.py` runs automatically from `fetch.py` / `train.py` (markers `data/.v41_migrated`, `data/.v42_migrated`). |

Backfills to run once (Actions → backfill): `tabular 1998 2026` (ISD-Lite rows + IBTrACS cyclone
features) and `f3-events 2023-01-01 2026-08-01` (F3 grids → CNN dataset + tabular F3 columns),
then `retrain`.

## Architecture

```
GitHub Actions (hourly)                     GitHub Actions (03:00 UTC daily)
app/fetch.py ──► data/snapshots.csv ──► app/train.py (scikit-learn)
   │  rules + AI blend                          │  app/cnn.py (NumPy ConvNet)
   │  └─► email alerts (Gmail SMTP)             ▼
   │  └─► data/latest.json, history.json   model/weights.json + cnn_weights.json
   ├─► app/grid.py ──► HKO F3 gridded nowcast (v3 CNN input, live)
   ├─► app/jtwc.py ──► data/tc_state.json (JTWC RSS + warning text)
   └─► site/data/* ──► GitHub Pages dashboard
```

- `app/fetch.py` — stdlib-only poller: fetches `rhrread` + `warnsum`, appends a snapshot,
  runs rules + AI (sigmoid over JSON weights — no ML lib needed), sends emails on
  triggers, writes dashboard JSON.
- `app/grid.py` — v3: fetches HKO's gridded rainfall nowcast (`hko_data/F3/...`), takes
  the HK-region window and downsamples each lead frame to 32×32; stdlib only.
- `app/cnn.py` — v3: small pure-NumPy ConvNet (2 conv layers → dense → 3 sigmoid heads)
  trained from archived grid data; predicts max rainfall ≥15/25/35 mm in the 30-min
  window ~2h ahead; inference is NumPy-only, weights exported to `model/cnn_weights.json`.
- `app/backfill.py` — one-off: pulls historical daily ZIPs of the gridded nowcast from
  the data.gov.hk historical archive (`app.data.gov.hk/v1/historical-archive/get-file`)
  and builds `data/grid_dataset.npz` for CNN training.
- `app/jtwc.py` — V2 (source fixed in v4.1): reads the JTWC RSS feed for active
  Western-Pacific systems and parses each `wpNNYYweb.txt` warning (current fix, wind,
  12–120 h forecast fixes); computes distance/bearing from Hong Kong and the 24h
  forecast distance; persists `data/tc_state.json`.
- `app/backfill_tc.py` — v4.2: IBTrACS best tracks → hourly `tc_*` features for
  historical rows (called by `backfill_tabular.py`).
- `app/rules.py` — interpretable nowcast rules (rainfall trend, humidity, active
  warnings, cyclone distance/wind/approach).
- `app/train.py` — nightly: trains one balanced logistic regression per target on
  accumulated data with lookahead labels; exports coefficients + feature scaling to
  `model/weights.json`; skips quietly until ≥48 rows / ≥2 positive samples per target.
- `site/index.html` — static dashboard (GitHub Pages).

## Notification triggers

1. Official HKO warning state changes (issue / extend / escalate / downgrade / cancel) — always emailed.
2. Lead alerts: hybrid (rules + AI) probability ≥60% for Amber/Red Rainstorm or TC Signal 3+
   within the horizon — emailed, then silent for 6h per signal type (cooldown).

## Setup

1. Create a **private** repo named `testing`, push this folder to it.
2. Create a **Gmail app password** (Google Account → Security → 2-Step Verification →
   App passwords) — needed because GitHub Actions can't use your normal Gmail login.
3. Repo → Settings → Secrets and variables → Actions → add:
   - `SMTP_USER` — your Gmail address
   - `SMTP_APP_PASSWORD` — the 16-char app password
   - `NOTIFY_TO` — the address that receives alerts (can be the same Gmail)
4. Repo → Settings → Pages → **Build and deployment** → Source: **Deploy from a branch** →
   Branch: `main`, folder: `/site`. GitHub builds the dashboard from the pushed `site/`
   folder automatically (no workflow needed); the URL will be
   `https://<you>.github.io/testing/`. Private-repo Pages needs Pro — included in the
   student pack. (The old `pages.yml` workflow was removed — workflow-based Pages fails
   with "Resource not accessible by integration" because the Actions token cannot
   create a Pages site on a private repo.)
5. Actions will start polling on the schedule (hourly) and retraining nightly.
   The AI model activates once ~2 days of snapshots (~48 rows) accumulate.
   To test immediately, open Actions → `poll` → Run workflow (workflow_dispatch).
   Set the `DISABLE_EMAIL` environment variable to 1 on a run if you want to test
   without sending mail.
6. V3 CNN: a 7-day training set is committed in `data/grid_dataset.npz`. To refresh it
   with more history, run `python app/backfill.py <days>` locally (downloads the daily
   grid ZIPs from the data.gov.hk historical archive) and commit the new dataset; the
   nightly retrain will then train `model/cnn_weights.json`.

## Historical data — what's available

- **Hourly `rhrread` / `warnsum` snapshots: no official history.** These are real-time
  only; the model's tabular labels still accumulate from day one.
- **Daily climate series: yes, since 1884+.** `data.gov.hk` hosts per-station CSVs
  (rainfall, temperature, humidity, pressure, wind, sunshine, …) e.g.
  `data.weather.gov.hk/weatherAPI/cis/csvfile/HKO/ALL/daily_HKO_RF_ALL.csv`.
- **Gridded rainfall nowcast (v3 CNN): yes, full history.** The data.gov.hk historical
  archive (`app.data.gov.hk/v1/historical-archive/get-file?url=…&time=YYYYMMDD`) serves
  daily ZIPs with a timestamped snapshot every ~15 min (~24 MB/day).
- **Radar imagery: no.** Not exposed via HKO Open Data and no public archive, so the
  v3 CNN is trained on the gridded nowcast fields instead.

## Future

- Pressure-drop features from daily climate CSVs (e.g. `daily_HKO_MSLP_ALL.csv`) to
  enrich the tabular model.
- Public-repo flip when the student discount expires (unlimited Actions minutes)

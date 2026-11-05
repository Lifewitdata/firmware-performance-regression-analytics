# Firmware Version Performance & Hardware Regression Analytics

Does firmware v2.0 improve device performance, or introduce hardware problems?

A hardware team ships firmware **v2.0.0** to a staged cohort. This repo measures the update against devices that stayed
on **v1.0.0** and answers: are error rates up, is uptime down, are sensor readings steadier, has response time changed,
which individual devices regressed, and should the rollout continue, be investigated, or be rolled back.

**Conclusion of the shipped analysis (152 devices, 36,480 telemetry intervals):**
`CONTINUE for hardware revision A / ROLLBACK for revision B`. Rev A gains −45% firmware errors, −64% sensor jitter and
−12% response time. Rev B regresses +162% firmware errors, +200% reboots (MTBF 120 h → 22 h) and +22% P95 response time.
Fleet averages hide this because the two segments partly cancel.

---

## Data: real, and experimental

There is **no public dataset** of real hardware readings carrying genuine firmware-release labels. This project is
therefore built from real hardware signals plus a clearly labelled experimental firmware-release layer. Nothing pretends
to be more than it is.

| Source | What is real | Used for |
|---|---|---|
| [NASA IMS Bearing Dataset](https://physicsopenlab.org/2017/09/01/bearing-dataset/) (Univ. of Cincinnati IMS / NASA PCoE) | 20 kHz vibration snapshots of 4 bearings, run to failure, 3 test rigs (9,464 raw files) | Every device's vibration signal; hardware degradation and reliability |
| [Numenta Anomaly Benchmark](https://github.com/numenta/NAB) | 5 real-world labelled time series (machine temperature, ec2 request latency, ambient temperature, cpu) | Real response-time profile for devices; validation of the anomaly detector against labelled windows |
| `src/experiment_design.py` | nothing — synthetic, seeded, documented | Devices, firmware versions, rollout, error/reboot/latency/filter effects |

The injected effects are written to `data/firmware_metadata/firmware_effects_ground_truth.json`. They are **never an
input to the analysis** — they exist so the final step can check whether the analysis recovered them. It does
(see notebook 05). Per-column origin is in that file and in `COLUMN_ORIGIN`.

Device construction: each real IMS bearing history is cut into non-overlapping 240-snapshot (40 h) windows; each window
is one *device*. So 152 devices with authentic, differently-aged vibration signals, including 4 run-to-failure units.
Assignment to the treatment/control arms is randomised, stratified by hardware revision, test rig and end-of-life status.
Hardware faults are derived from the **real** signal only, so firmware can neither cause nor hide them.

## Why the comparison needs a control arm

Bearing wear and real latency incidents both happen during the 40 h observation window. A naive before/after on updated
devices attributes that drift to the firmware. Two designs avoid this:

* **Difference-in-differences** on device-level KPIs: `mean(post − pre | treatment) − mean(post − pre | control)`.
* Welch t-test on device deltas (primary), Mann-Whitney, 95% bootstrap CI, OLS with device-clustered standard errors,
  Benjamini-Hochberg FDR across the 8 metrics, plus exact Poisson rate-ratio tests for event counts.

Notebook 04 shows both pitfalls with numbers: naive before/after reports firmware errors as non-significant
(p=0.20) while segment-level DiD finds +162% on rev B (adj. p<1e-6).

## Repository layout

```
data/nasa_ims/       raw IMS snapshots (1st_test, 2nd_test, 4th_test/txt) — git-ignored, 6.2 GB
data/nab/            5 NAB series + official anomaly labels — git-ignored
data/firmware_metadata/  devices, firmware releases, injected ground truth
data/processed/      analysis outputs — CSV/parquet committed (~4.8 MB), SQLite DB git-ignored
notebooks/01..06     the analysis, executed with stored outputs
sql/                 schema + comparison / metrics / reliability queries
src/                 modules (preprocessing, experiment design, metrics, tests, detection, reliability)
src/pipeline.py      runs everything and writes every artefact
dashboards/          Tableau extracts + how-to
app/streamlit_app.py interactive dashboard
tests/               19 tests
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Unpack IMS. Files are tab-separated text named by timestamp, with no extension.
#    Windows' built-in tar handles .rar:  tar -xf 1st_test.rar -C data/nasa_ims
# 2. Get NAB series + labels into data/nab/ (5 files used are listed in src/preprocessing.py:NAB_SERIES)
# 3. Run:
python -m src.pipeline            # ~10 s once features are cached; ~90 s to build them the first time
pytest                            # 19 tests
streamlit run app/streamlit_app.py
```

The committed `data/processed/` outputs mean a fresh clone can run the app and open the Tableau extracts
immediately, without unpacking 6.2 GB of raw data. `python -m src.pipeline` regenerates them from the raw inputs.

If the app reports missing analysis outputs, your checkout predates commit `fa8e783` (2026-11-05), when those
outputs were committed — run `git pull origin main` and restart Streamlit.

## Modules

| Module | Responsibility |
|---|---|
| [src/preprocessing.py](file:///C:/Users/hp/Documents/Qoder/2026-11-05/dbc2c567/src/preprocessing.py) | Parse IMS snapshots, 8 vibration features per bearing per snapshot (RMS, peak, kurtosis, crest factor, HF energy ratio); load NAB with official labels |
| [src/experiment_design.py](file:///C:/Users/hp/Documents/Qoder/2026-11-05/dbc2c567/src/experiment_design.py) | Fleet construction, stratified rollout, the single `FirmwareEffects` table of injected effects |
| [src/performance_metrics.py](file:///C:/Users/hp/Documents/Qoder/2026-11-05/dbc2c567/src/performance_metrics.py) | Device×period KPIs, fleet KPIs, hourly trend; `METRICS` registry carrying direction + guardrail + question mapping |
| [src/statistical_tests.py](file:///C:/Users/hp/Documents/Qoder/2026-11-05/dbc2c567/src/statistical_tests.py) | DiD, Welch/Mann-Whitney/bootstrap/clustered OLS, BH-FDR, exact Poisson rate tests |
| [src/regression_detection.py](file:///C:/Users/hp/Documents/Qoder/2026-11-05/dbc2c567/src/regression_detection.py) | Per-device robust-z regression scores, CUSUM change points, NAB detector evaluation, CONTINUE/INVESTIGATE/ROLLBACK rules |
| [src/reliability_analysis.py](file:///C:/Users/hp/Documents/Qoder/2026-11-05/dbc2c567/src/reliability_analysis.py) | MTBF with exact Poisson CIs, availability, Kaplan-Meier + log-rank on time to first reboot, bearing health index |

Notebooks, all executed: 01 sensor exploration · 02 experiment design · 03 performance metrics ·
04 statistical comparison · 05 regression detection + decision · 06 reliability.

## SQL

`sql/schema.sql` builds a SQLite DB (`data/processed/firmware_analytics.db`) with `firmware_releases`, `devices`,
`telemetry`, `ims_features` and a `v_device_period_kpis` view. The three report files are split on `-- name:` markers by
`src/pipeline.py` and exported to `data/processed/sql_results/*.csv` — DiD by revision, device change ranking,
KPI/percentile/trend tables, MTBF, fault attribution, bearing degradation. Portable to Postgres/Snowflake;
`PERCENT_RANK` needs SQLite 3.25+ or an approximation elsewhere.

## Dashboards

* **Streamlit** (`app/streamlit_app.py`): decision, DiD forests, trends, device regression map + drill-down,
  reliability, raw sensor browser, provenance/validation tab.
* **Tableau**: `dashboards/tableau_extracts/*.csv` — 14 data sources covering every sheet (schema-on-read, no `.hyper`).
  The 12 summary extracts (~250 KB) are committed; `telemetry.csv` and `bearing_health_index.csv` are git-ignored and
  rebuilt by `python -m src.pipeline`. For **Tableau Public**, the five files you need are
  `comparison_by_segment`, `device_regression_scores`, `fleet_hourly`, `kaplan_meier_reboot`, `bearing_degradation`.
  Build steps in [dashboards/README.md](file:///C:/Users/hp/Documents/Qoder/2026-11-05/dbc2c567/dashboards/README.md).
  A real `.twbx` must be assembled in Tableau Desktop; it is not machine-generated here.

## Reading the result

| Question | Answer |
|---|---|
| Better than the previous version? | Yes on rev A, no on rev B |
| Error rates increased? | Rev B: +162% (adj. p<1e-6). Rev A: −45% |
| Uptime improved or declined? | No significant change anywhere; reboots are the signal (+200% on rev B) |
| Sensor readings steadier? | Yes, jitter −64% on A and −54% on B; rev B also reports +8% RMS from a calibration bug |
| Response time changed? | −12% on rev A, +22% P95 on rev B |
| Which devices regressed? | 27/92 updated devices flagged, 26 of them rev B (`data/processed/device_regression_scores.csv`) |
| Continue, investigate, roll back? | Continue A; halt and roll back B; keep the fleet-level dashboard segmented |

## Limitations

* Firmware effects are injected, not observed; real-world confounders (batch effects, environment, operator actions) are only partly represented.
* Only 2 hardware revisions and 152 devices; rev-B CIs are wide, and per-device flags are descriptive rather than individually tested.
* Vibration is measured at 10-minute cadence, so faults are detected within one interval, not continuously.
* NAB precision of the simple z-score detector is low (0.09–1.0 by series). It is reported as-is: a single unsupervised
  alarm should not trigger a rollback on its own, which is why the decision rests on the controlled comparison.

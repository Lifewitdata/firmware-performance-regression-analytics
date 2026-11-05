# Dashboards

## Streamlit (ready to run)

```bash
python -m src.pipeline                       # produces data/processed/*
streamlit run app/streamlit_app.py
```

Seven tabs: Decision · KPI comparison · Trends · Device regressions · Reliability · Real sensor data · Data provenance.
All seven were verified to render without exceptions against the shipped dataset; `screenshots/` is empty until you
capture your own (the app was checked programmatically, not visually).

## Tableau

`firmware_comparison.twbx` is **not** generated here: a real `.twbx` is a zip written by Tableau Desktop.
`dashboards/tableau_extracts/*.csv` are the data sources. All 14 are refreshed by `python -m src.pipeline`
(schema-on-read, so no `.hyper` step is needed). Twelve are committed; `telemetry.csv` (8 MB) and
`bearing_health_index.csv` (4.8 MB) are git-ignored, so run the pipeline if you want those two.

For **Tableau Public**, upload these five (~250 KB total): `comparison_by_segment.csv`,
`device_regression_scores.csv`, `fleet_hourly.csv`, `kaplan_meier_reboot.csv`, `bearing_degradation.csv`.

| Extract | Grain | Rows | Use |
|---|---|---|---|
| `telemetry.csv` | device × 10-min interval | 36,480 | trends, device drill-down |
| `device_period_kpis.csv` | device × pre/post | 304 | distributions, DiD from raw |
| `devices.csv` | device | 152 | attributes for colour/filter |
| `comparison_overall.csv` | metric (fleet) | 8 | forest plot, verdict table |
| `comparison_by_segment.csv` | segment × metric | 16 | forest plot, verdict table |
| `device_regression_scores.csv` | updated device | 92 | regression map |
| `fleet_hourly.csv` | hourly bucket × `group` × `hw_revision` | 160 | line charts |
| `kpis_by_version.csv` | firmware version | 2 | headline KPI strip |
| `mtbf.csv` | firmware × revision | 4 | reliability table |
| `kaplan_meier_reboot.csv` | cohort × event time | 32 | survival step chart, precomputed |
| `time_to_first_reboot.csv` | device | 152 | survival input, censored durations |
| `bearing_degradation.csv` | bearing | 12 | real IMS degradation table |
| `bearing_health_index.csv` | bearing × snapshot | 37,856 | degradation line charts |
| `nab_detector_evaluation.csv` | NAB series | 5 | detector quality table |

`comparison_overall.csv` and `comparison_by_segment.csv` already contain the DiD estimate, 95% bootstrap CI,
FDR-adjusted p-value and verdict per metric, so the forest plot needs no table calculation.

Build steps:

1. New workbook → connect to `comparison_by_segment.csv` (add `comparison_overall.csv` for the fleet row).
2. Sheet 1 "Decision": rows `label`, columns `segment`, colour `verdict` (custom palette: improved green,
   regressed red, no significant change grey), size by `|relative_change_pct|`. Filter to `guardrail = True`
   for the executive view. Add `ci_low` / `ci_high` as an error bar via the two CI columns on Rows.
3. Sheet 2 "Regression map": `device_regression_scores.csv` → scatter `z_fw_errors_per_100` × `z_p95_response_ms`,
   colour `hw_revision`, shape `regression_flag`, reference lines at 3 on both axes.
4. Sheet 3 "Trends": `fleet_hourly.csv` → Columns `timestamp` (discrete, hourly), Rows the measure you want
   (`fw_errors_per_100`, `p50_response_ms`, `uptime_pct`, `mean_rms`, `alarm_rate_pct`, `reboots` are separate
   columns), colour `group`, detail `hw_revision`, line style by `group`.
   Add a vertical annotation at **2026-01-05 20:00** (interval 120 = the v2.0 rollout; `bucket` is the interval
   index of each hourly point, 0-234).
   Use `telemetry.csv` instead if you want to filter to a single device.
5. Sheet 4 "Reliability": `mtbf.csv` as a KPI table, plus `kaplan_meier_reboot.csv` as a step line
   (`time` on Columns, `survival` on Rows, `cohort` on Colour, mark type Line with Step interpolation;
   `ci_low`/`ci_high` for the shaded band).
6. Sheet 5 "Real hardware degradation": `bearing_health_index.csv` → `hours` on Columns, `health_index_smooth`
   on Rows, `test_id`/`bearing` on Detail, colour `failure_mode`, reference line at 2.5.
7. Publish, then save the workbook as `.twbx` into this folder so the extracts are packaged.

Keep segment-level views visible on the same sheet as fleet averages: the fleet-level numbers for this dataset
deliberately understate the rev-B regression, which is the main finding.

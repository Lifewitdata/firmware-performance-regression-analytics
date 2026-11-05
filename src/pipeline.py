"""End-to-end pipeline: features -> experiment -> KPIs -> tests -> decision -> SQLite + Tableau extracts.

Run from the project root:  python -m src.pipeline [--rebuild-features]
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from pathlib import Path

import pandas as pd

from src import preprocessing as pp
from src.experiment_design import build_experiment
from src.performance_metrics import device_period_kpis, fleet_kpis_by_version, fleet_timeseries
from src.regression_detection import evaluate_detector_on_nab, fleet_change_point, rollout_decision
from src.reliability_analysis import (bearing_health, degradation_summary, kaplan_meier, logrank_test, mtbf_table,
                                      time_to_first_reboot)
from src.statistical_tests import event_rate_tests, naive_comparison

ROOT = pp.ROOT
SQL_DIR = ROOT / "sql"
DB_PATH = pp.PROCESSED / "firmware_analytics.db"
TABLEAU = ROOT / "dashboards" / "tableau_extracts"
SQL_RESULTS = pp.PROCESSED / "sql_results"


def named_queries(path: Path) -> dict[str, str]:
    parts = re.split(r"^-- name:\s*(\w+)\s*$", path.read_text(), flags=re.M)
    return {parts[i]: parts[i + 1].strip().rstrip(";") for i in range(1, len(parts), 2)}


def load_sqlite(features, fleet, telemetry, db_path: Path = DB_PATH) -> None:
    with sqlite3.connect(db_path) as con:
        con.executescript((SQL_DIR / "schema.sql").read_text())
        from src.experiment_design import FIRMWARE_RELEASES
        FIRMWARE_RELEASES.to_sql("firmware_releases", con, if_exists="append", index=False)
        fleet.rename(columns={"group": "arm"}).astype({"source_start_time": str}) \
            .to_sql("devices", con, if_exists="append", index=False)
        telemetry.rename(columns={"group": "arm"}).astype({"timestamp": str, "source_timestamp": str}) \
            .to_sql("telemetry", con, if_exists="append", index=False, chunksize=5000)
        features.astype({"timestamp": str}).to_sql("ims_features", con, if_exists="append", index=False, chunksize=5000)


def run_sql_reports(db_path: Path = DB_PATH) -> dict[str, pd.DataFrame]:
    SQL_RESULTS.mkdir(parents=True, exist_ok=True)
    out = {}
    with sqlite3.connect(db_path) as con:
        for f in ("firmware_comparison.sql", "performance_metrics.sql", "reliability_analysis.sql"):
            for name, q in named_queries(SQL_DIR / f).items():
                out[name] = pd.read_sql_query(q, con)
                out[name].to_csv(SQL_RESULTS / f"{name}.csv", index=False)
    return out


def main(rebuild_features: bool = False) -> dict:
    t0 = time.time()
    feat_path = pp.PROCESSED / "ims_features.parquet"
    if rebuild_features or not feat_path.exists():
        print("Extracting IMS vibration features (one-off, a few minutes)...")
        pp.build_ims_features().to_parquet(feat_path, index=False)
    features = pp.load_ims_features(feat_path)
    print(f"[{time.time() - t0:5.0f}s] IMS features: {len(features):,} rows")

    fleet, tel = build_experiment(features)
    print(f"[{time.time() - t0:5.0f}s] Fleet: {len(fleet)} devices, telemetry {len(tel):,} rows")

    kpis = device_period_kpis(tel)
    kpis.to_parquet(pp.PROCESSED / "device_period_kpis.parquet", index=False)
    decision = rollout_decision(kpis)
    decision["overall"].to_csv(pp.PROCESSED / "comparison_overall.csv", index=False)
    decision["by_segment"].to_csv(pp.PROCESSED / "comparison_by_segment.csv", index=False)
    decision["device_scores"].to_csv(pp.PROCESSED / "device_regression_scores.csv", index=False)
    rates = event_rate_tests(tel)
    rates.to_csv(pp.PROCESSED / "event_rate_tests.csv", index=False)
    naive = pd.DataFrame([naive_comparison(kpis, m) for m in decision["overall"].metric])
    naive.to_csv(pp.PROCESSED / "naive_comparison.csv", index=False)

    cps = [fleet_change_point(tel, m, rev) for m in ("fw_error_count", "response_time_ms")
           for rev in (None, "A", "B")]
    cp_df = pd.DataFrame([{k: v for k, v in c.items() if k not in ("gap",)} for c in cps])
    cp_df.to_csv(pp.PROCESSED / "change_points.csv", index=False)

    nab_eval = evaluate_detector_on_nab(pp.load_all_nab())
    nab_eval.to_csv(pp.PROCESSED / "nab_detector_evaluation.csv", index=False)

    mtbf = mtbf_table(tel)
    mtbf.to_csv(pp.PROCESSED / "mtbf.csv", index=False)
    ttf = time_to_first_reboot(tel)
    ttf.to_csv(pp.PROCESSED / "time_to_first_reboot.csv", index=False)
    lr = {}
    for rev, g in ttf.groupby("hw_revision"):
        a, b = g[g.group == "control"], g[g.group == "treatment"]
        lr[rev] = logrank_test(a.duration_h, a.event, b.duration_h, b.event)
    health = bearing_health(features)
    degradation_summary(health).to_csv(pp.PROCESSED / "bearing_degradation.csv", index=False)
    print(f"[{time.time() - t0:5.0f}s] Analysis complete")

    load_sqlite(features, fleet, tel)
    sql = run_sql_reports()
    print(f"[{time.time() - t0:5.0f}s] SQLite loaded, {len(sql)} SQL reports exported")

    TABLEAU.mkdir(parents=True, exist_ok=True)
    tel.to_csv(TABLEAU / "telemetry.csv", index=False)
    kpis.to_csv(TABLEAU / "device_period_kpis.csv", index=False)
    fleet.to_csv(TABLEAU / "devices.csv", index=False)
    decision["overall"].assign(segment="fleet").to_csv(TABLEAU / "comparison_overall.csv", index=False)
    decision["by_segment"].to_csv(TABLEAU / "comparison_by_segment.csv", index=False)
    decision["device_scores"].to_csv(TABLEAU / "device_regression_scores.csv", index=False)
    fleet_timeseries(tel).to_csv(TABLEAU / "fleet_hourly.csv", index=False)
    fleet_kpis_by_version(tel).to_csv(TABLEAU / "kpis_by_version.csv", index=False)
    mtbf.to_csv(TABLEAU / "mtbf.csv", index=False)
    ttf.to_csv(TABLEAU / "time_to_first_reboot.csv", index=False)
    km = pd.concat([kaplan_meier(g.duration_h, g.event).assign(cohort=f"v{fw} rev {rev}", devices=len(g))
                   for (fw, rev), g in ttf.groupby(["firmware_version", "hw_revision"])], ignore_index=True)
    km.to_csv(TABLEAU / "kaplan_meier_reboot.csv", index=False)
    degradation_summary(health).to_csv(TABLEAU / "bearing_degradation.csv", index=False)
    health[["test_id", "bearing", "timestamp", "hours", "health_index", "health_index_smooth",
            "rms", "kurtosis", "failure_mode"]].to_csv(TABLEAU / "bearing_health_index.csv", index=False)
    nab_eval.to_csv(TABLEAU / "nab_detector_evaluation.csv", index=False)
    print(f"[{time.time() - t0:5.0f}s] Tableau extracts written to {TABLEAU.relative_to(ROOT)}")

    summary = {
        "headline": decision["headline"],
        "decisions": decision["decisions"],
        "logrank_time_to_first_reboot": lr,
        "change_points": cp_df.to_dict(orient="records"),
        "flagged_devices": int(decision["device_scores"].regression_flag.sum()),
        "treated_devices": int(len(decision["device_scores"])),
        "nab_detector_mean_recall": float(nab_eval.recall.mean()),
    }
    (pp.PROCESSED / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\nRECOMMENDATION:", decision["headline"])
    for seg, d in decision["decisions"].items():
        print(f"  {seg:15s} {d['decision']:12s} " + "; ".join(d["reasons"]))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-features", action="store_true")
    main(ap.parse_args().rebuild_features)

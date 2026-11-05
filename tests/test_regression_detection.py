"""End-to-end check: the analysis must recover the injected v2.0 effects on the synthetic fleet."""
import numpy as np
import pandas as pd

from src.experiment_design import ROLLOUT_AT
from src.regression_detection import (cusum, device_regression_scores, evaluate_detector_on_nab,
                                      fleet_change_point, rollout_decision)
from src.reliability_analysis import kaplan_meier, logrank_test, mtbf_table
from src.statistical_tests import compare_by_segment


def test_segment_results_recover_ground_truth(kpis):
    res = compare_by_segment(kpis, "hw_revision").set_index(["segment", "metric"])
    assert res.loc[("hw_revision=B", "fw_errors_per_100"), "verdict"] == "regressed"
    assert res.loc[("hw_revision=B", "p50_response_ms"), "verdict"] == "regressed"
    assert res.loc[("hw_revision=A", "p50_response_ms"), "verdict"] == "improved"
    assert res.loc[("hw_revision=A", "sensor_jitter"), "verdict"] == "improved"
    assert res.loc[("hw_revision=A", "fw_errors_per_100"), "verdict"] != "regressed"


def test_decision_targets_rev_b(kpis):
    d = rollout_decision(kpis)["decisions"]
    assert d["hw_revision=B"]["decision"] == "ROLLBACK"
    assert d["hw_revision=A"]["decision"] == "CONTINUE"


def test_device_flags_concentrate_on_rev_b(kpis):
    s = device_regression_scores(kpis)
    flagged = s[s.regression_flag]
    assert len(flagged) > 0
    assert (flagged.hw_revision == "B").mean() > 0.8


def test_change_point_near_rollout(telemetry):
    cp = fleet_change_point(telemetry, "response_time_ms", "B")
    assert abs(cp["change_point_interval"] - ROLLOUT_AT) <= 12


def test_cusum_no_alarm_on_stationary_series():
    x = np.random.default_rng(0).normal(0, 1, 500)
    assert cusum(x, 100)["first_alarm"] is None
    y = np.r_[x[:250], x[250:] + 2]
    assert 250 <= cusum(y, 100)["first_alarm"] < 270


def test_nab_evaluation_counts_windows():
    ts = pd.date_range("2020-01-01", periods=1000, freq="5min")
    v = np.random.default_rng(0).normal(0, 1, 1000)
    v[700:705] += 15
    win = np.zeros(1000, bool)
    win[680:720] = True
    nab = pd.DataFrame({"timestamp": ts, "value": v, "in_anomaly_window": win, "series": "toy"})
    r = evaluate_detector_on_nab(nab).iloc[0]
    assert r.anomaly_windows == 1 and r.windows_detected == 1 and r.recall == 1.0


def test_kaplan_meier_and_logrank():
    km = kaplan_meier([1, 2, 3, 4, 5], [1, 1, 0, 1, 0])
    assert km.survival.iloc[-1] == np.float64(0.8 * 0.75 * 0.5)
    rng = np.random.default_rng(0)
    a, b = rng.exponential(10, 200), rng.exponential(3, 200)
    assert logrank_test(a, np.ones(200), b, np.ones(200))["p_value"] < 1e-6


def test_mtbf_lower_for_rev_b_on_v2(telemetry):
    m = mtbf_table(telemetry).set_index(["firmware_version", "hw_revision"])
    assert m.loc[("2.0.0", "B"), "mtbf_reboot_hours"] < m.loc[("1.0.0", "B"), "mtbf_reboot_hours"]

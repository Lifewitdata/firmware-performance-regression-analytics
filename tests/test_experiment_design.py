import numpy as np
import pandas as pd
import pytest

from src.experiment_design import BASELINE_FW, NEW_FW, ROLLOUT_AT, WINDOW, build_fleet
from src.preprocessing import parse_snapshot_time, signal_features


def test_signal_features_on_sine():
    t = np.arange(20480) / 20000
    f = signal_features(np.sin(2 * np.pi * 100 * t))
    assert f["rms"] == pytest.approx(1 / np.sqrt(2), rel=1e-3)
    assert f["crest_factor"] == pytest.approx(np.sqrt(2), rel=1e-2)
    assert f["kurtosis"] == pytest.approx(1.5, rel=1e-2)
    assert f["hf_energy_ratio"] < 0.01


def test_parse_snapshot_time():
    assert parse_snapshot_time("2004.02.12.10.32.39") == pd.Timestamp("2004-02-12 10:32:39")


def test_fleet_is_reproducible_and_balanced(features):
    a, b = build_fleet(features, seed=5), build_fleet(features, seed=5)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 12 * (1440 // WINDOW)
    share = (a.group == "treatment").mean()
    assert 0.5 < share < 0.7
    for _, g in a.groupby("hw_revision"):
        assert g.group.nunique() == 2


def test_firmware_assignment(telemetry):
    ctrl = telemetry[telemetry.group == "control"]
    assert (ctrl.firmware_version == BASELINE_FW).all()
    tr = telemetry[telemetry.group == "treatment"]
    assert (tr.loc[tr.interval_idx < ROLLOUT_AT, "firmware_version"] == BASELINE_FW).all()
    assert (tr.loc[tr.interval_idx >= ROLLOUT_AT, "firmware_version"] == NEW_FW).all()


def test_real_columns_untouched_and_pre_period_identical(telemetry):
    pre = telemetry[telemetry.period == "pre"]
    np.testing.assert_allclose(pre.vib_rms, pre.real_rms)
    ctrl = telemetry[telemetry.group == "control"]
    np.testing.assert_allclose(ctrl.vib_rms, ctrl.real_rms)


def test_uptime_bounds(telemetry):
    assert telemetry.uptime_pct.between(0, 100).all()
    assert (telemetry.downtime_min <= 10).all()

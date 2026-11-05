"""Experimental firmware-release layer on top of real IMS bearing and NAB telemetry.

Everything that is *not* a real measurement is created here, from a fixed seed, and the injected
effects are written to `data/firmware_metadata/firmware_effects_ground_truth.json` so the analysis
can be checked against a known answer. Real measurements are kept in `real_*` columns unchanged.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing import DATA, PROCESSED, load_nab_series

FW_META = DATA / "firmware_metadata"

WINDOW = 240            # telemetry intervals per device (40 h at a 10-minute cadence)
ROLLOUT_AT = 120        # interval index at which treatment devices receive v2.0
INTERVAL_MIN = 10
CALENDAR_START = pd.Timestamp("2026-01-05 00:00")
TREATMENT_SHARE = 0.6
HW_REV_B_SHARE = 0.3
REGIONS = ("NA-East", "EU-West", "APAC")

BASELINE_FW = "1.0.0"
NEW_FW = "2.0.0"

FIRMWARE_RELEASES = pd.DataFrame([
    {"firmware_version": BASELINE_FW, "release_date": "2025-06-02", "status": "stable",
     "notes": "Baseline production firmware."},
    {"firmware_version": NEW_FW, "release_date": str((CALENDAR_START + pd.Timedelta(minutes=ROLLOUT_AT * INTERVAL_MIN)).date()),
     "status": "staged rollout",
     "notes": "New DSP filter on vibration channel, async telemetry stack, new I2C driver."},
])


@dataclass(frozen=True)
class FirmwareEffects:
    """Ground-truth effects of v2.0 relative to v1.0. Values are multipliers unless stated."""
    sensor_filter_alpha: float = 0.5          # EMA weight on new reading (v2.0 DSP filter, all revisions)
    rms_gain: dict = field(default_factory=lambda: {"A": 1.00, "B": 1.06})   # calibration-table bug on rev B
    response_time: dict = field(default_factory=lambda: {"A": 0.88, "B": 1.22})
    fw_error_rate: dict = field(default_factory=lambda: {"A": 0.75, "B": 2.20})
    reboot_prob: dict = field(default_factory=lambda: {"A": 0.001, "B": 0.008})  # absolute per interval
    baseline_reboot_prob: float = 0.001
    baseline_fw_error_rate: float = 0.04       # Poisson mean per interval
    fault_downtime_min: float = 5.0            # protective shutdown when the bearing is in a fault state


EFFECTS = FirmwareEffects()


def _healthy_baselines(features: pd.DataFrame, healthy_share: float = 0.2) -> pd.DataFrame:
    def per_bearing(g: pd.DataFrame) -> pd.Series:
        head = g.nsmallest(max(1, int(len(g) * healthy_share)), "snapshot_idx")
        return pd.Series({"healthy_rms": head["rms"].median(), "healthy_kurtosis": head["kurtosis"].median()})
    return features.groupby(["test_id", "bearing"]).apply(per_bearing, include_groups=False).reset_index()


def build_fleet(features: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Slice every real bearing history into non-overlapping windows; each window is one device."""
    rng = np.random.default_rng(seed)
    rows = []
    for (test_id, bearing), g in features.groupby(["test_id", "bearing"], sort=True):
        n_windows = len(g) // WINDOW
        for w in range(n_windows):
            start = w * WINDOW
            rows.append({
                "source_test": test_id, "source_bearing": int(bearing), "window_index": w,
                "source_start_idx": start,
                "source_start_time": g["timestamp"].iloc[start],
                "failure_mode": g["failure_mode"].iloc[0],
                "contains_end_of_life": bool(g["bearing_failed"].iloc[0] and w == n_windows - 1),
            })
    fleet = pd.DataFrame(rows)
    fleet.insert(0, "device_id", [f"DEV-{i:04d}" for i in range(1, len(fleet) + 1)])
    fleet["hw_revision"] = np.where(rng.random(len(fleet)) < HW_REV_B_SHARE, "B", "A")
    fleet["region"] = rng.choice(REGIONS, size=len(fleet))

    # Stratified randomisation so end-of-life hardware and board revisions are balanced across arms.
    fleet["group"] = "control"
    for _, idx in fleet.groupby(["hw_revision", "contains_end_of_life", "source_test"]).groups.items():
        idx = rng.permutation(np.asarray(idx))
        n_treat = int(round(len(idx) * TREATMENT_SHARE))
        fleet.loc[idx[:n_treat], "group"] = "treatment"
    return fleet


def _ema(values: np.ndarray, alpha: float, start: int) -> np.ndarray:
    out = values.copy()
    for t in range(max(start, 1), len(out)):
        out[t] = alpha * values[t] + (1 - alpha) * out[t - 1]
    return out


def generate_telemetry(fleet: pd.DataFrame, features: pd.DataFrame, latency_source: np.ndarray,
                       effects: FirmwareEffects = EFFECTS, seed: int = 7) -> pd.DataFrame:
    """One row per device per 10-minute interval, combining real sensor data with the experimental layer."""
    rng = np.random.default_rng(seed)
    base = _healthy_baselines(features).set_index(["test_id", "bearing"])
    by_bearing = {k: g.sort_values("snapshot_idx").reset_index(drop=True)
                  for k, g in features.groupby(["test_id", "bearing"])}
    t_idx = np.arange(WINDOW)
    timestamps = CALENDAR_START + pd.to_timedelta(t_idx * INTERVAL_MIN, unit="min")
    frames = []

    for dev in fleet.itertuples(index=False):
        src = by_bearing[(dev.source_test, dev.source_bearing)].iloc[dev.source_start_idx:dev.source_start_idx + WINDOW]
        hb = base.loc[(dev.source_test, dev.source_bearing)]
        treated = dev.group == "treatment"
        on_v2 = treated & (t_idx >= ROLLOUT_AT)
        rev = dev.hw_revision

        real_rms = src["rms"].to_numpy()
        real_kurt = src["kurtosis"].to_numpy()
        real_peak = src["peak"].to_numpy()

        obs_rms, obs_kurt, obs_peak = real_rms.copy(), real_kurt.copy(), real_peak.copy()
        if treated:
            a = effects.sensor_filter_alpha
            obs_rms = _ema(real_rms, a, ROLLOUT_AT)
            obs_kurt = _ema(real_kurt, a, ROLLOUT_AT)
            obs_peak = _ema(real_peak, a, ROLLOUT_AT)
            obs_rms[on_v2] *= effects.rms_gain[rev]
            obs_peak[on_v2] *= effects.rms_gain[rev]

        # Real request-latency profile from NAB, block-bootstrapped so its temporal structure is kept.
        start = rng.integers(0, len(latency_source) - WINDOW)
        resp = latency_source[start:start + WINDOW] * rng.lognormal(0, 0.10)
        resp = np.where(on_v2, resp * effects.response_time[rev], resp)

        lam_dev = effects.baseline_fw_error_rate * rng.lognormal(0, 0.3)
        lam = np.where(on_v2, lam_dev * effects.fw_error_rate[rev], lam_dev)
        fw_errors = rng.poisson(lam)

        p_reboot = np.where(on_v2, effects.reboot_prob[rev], effects.baseline_reboot_prob)
        reboot = rng.random(WINDOW) < p_reboot
        reboot_down = np.where(reboot, rng.uniform(3, 9, WINDOW), 0.0)

        # Hardware state is derived from the *real* signal, so it is independent of firmware.
        hw_fault = (real_rms > 2.5 * hb["healthy_rms"]) | (real_kurt > 8)
        # Monitoring alarm uses the *observed* reading, so the v2.0 filter can change alarm behaviour.
        alarm = (obs_rms > 1.8 * hb["healthy_rms"]) | (obs_kurt > 5)

        downtime = np.minimum(reboot_down + np.where(hw_fault, effects.fault_downtime_min, 0.0), INTERVAL_MIN)

        frames.append(pd.DataFrame({
            "device_id": dev.device_id, "interval_idx": t_idx, "timestamp": timestamps,
            "firmware_version": np.where(on_v2, NEW_FW, BASELINE_FW),
            "period": np.where(t_idx >= ROLLOUT_AT, "post", "pre"),
            "group": dev.group, "hw_revision": rev, "region": dev.region,
            "source_timestamp": src["timestamp"].to_numpy(),
            "real_rms": real_rms, "real_kurtosis": real_kurt, "real_peak": real_peak,
            "vib_rms": obs_rms, "vib_kurtosis": obs_kurt, "vib_peak": obs_peak,
            "response_time_ms": resp, "fw_error_count": fw_errors,
            "vibration_alarm": alarm, "hardware_fault": hw_fault,
            "reboot": reboot, "downtime_min": downtime,
            "uptime_pct": 100 * (1 - downtime / INTERVAL_MIN),
        }))
    return pd.concat(frames, ignore_index=True)


COLUMN_ORIGIN = {
    "real (NASA IMS)": ["source_timestamp", "real_rms", "real_kurtosis", "real_peak", "hardware_fault"],
    "real (NAB ec2_request_latency) + experimental firmware effect": ["response_time_ms"],
    "real IMS signal + experimental firmware filter/calibration": ["vib_rms", "vib_kurtosis", "vib_peak", "vibration_alarm"],
    "experimental": ["device_id", "timestamp", "firmware_version", "period", "group", "hw_revision", "region",
                     "fw_error_count", "reboot", "downtime_min", "uptime_pct"],
}


def build_experiment(features: pd.DataFrame, seed: int = 42, write: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    latency = load_nab_series("ec2_request_latency_system_failure")["value"].to_numpy()
    fleet = build_fleet(features, seed=seed)
    telemetry = generate_telemetry(fleet, features, latency, seed=seed + 1)
    if write:
        FW_META.mkdir(parents=True, exist_ok=True)
        PROCESSED.mkdir(parents=True, exist_ok=True)
        fleet.to_csv(FW_META / "devices.csv", index=False)
        FIRMWARE_RELEASES.to_csv(FW_META / "firmware_releases.csv", index=False)
        (FW_META / "firmware_effects_ground_truth.json").write_text(json.dumps({
            "WARNING": "Experimental, injected effects. Used only to validate the analysis, never as an input to it.",
            "seed": seed, "window_intervals": WINDOW, "rollout_interval": ROLLOUT_AT,
            "interval_minutes": INTERVAL_MIN, "effects_v2_vs_v1": asdict(EFFECTS),
            "column_origin": COLUMN_ORIGIN,
        }, indent=2))
        telemetry.to_parquet(PROCESSED / "telemetry.parquet", index=False)
    return fleet, telemetry

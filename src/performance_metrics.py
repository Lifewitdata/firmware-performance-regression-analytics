"""Device- and fleet-level KPIs for firmware comparison."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Metric:
    name: str
    label: str
    higher_is_better: bool
    guardrail: bool      # a significant regression here can justify a rollback
    question: str


METRICS = (
    Metric("fw_errors_per_100", "Firmware errors / 100 intervals", False, True, "Have hardware error rates increased?"),
    Metric("uptime_pct", "Uptime %", True, True, "Has device uptime improved or declined?"),
    Metric("reboots_per_100", "Reboots / 100 intervals", False, True, "Has device uptime improved or declined?"),
    Metric("sensor_jitter", "Sensor jitter (rel. step-to-step change)", False, False, "Are sensor readings more stable?"),
    Metric("mean_rms", "Mean vibration RMS (g)", False, False, "Are sensor readings consistent / calibrated?"),
    Metric("alarm_rate_pct", "Vibration alarm rate %", False, False, "Are sensor readings more stable?"),
    Metric("p50_response_ms", "Median response time (ms)", False, False, "Has response time changed?"),
    Metric("p95_response_ms", "P95 response time (ms)", False, True, "Has response time changed?"),
)
METRIC_BY_NAME = {m.name: m for m in METRICS}


def _jitter(x: pd.Series) -> float:
    x = x.to_numpy()
    return float(np.mean(np.abs(np.diff(x))) / np.mean(x)) if len(x) > 1 and np.mean(x) > 0 else np.nan


def device_period_kpis(tel: pd.DataFrame) -> pd.DataFrame:
    """One row per device x period (pre/post rollout)."""
    g = tel.groupby(["device_id", "period"], sort=True)
    k = g.agg(
        group=("group", "first"), hw_revision=("hw_revision", "first"), region=("region", "first"),
        firmware_version=("firmware_version", "last"), n_intervals=("interval_idx", "size"),
        fw_errors=("fw_error_count", "sum"), reboots=("reboot", "sum"),
        uptime_pct=("uptime_pct", "mean"), mean_rms=("vib_rms", "mean"),
        rms_cv=("vib_rms", lambda s: s.std() / s.mean()),
        alarm_rate_pct=("vibration_alarm", lambda s: 100 * s.mean()),
        hw_fault_rate_pct=("hardware_fault", lambda s: 100 * s.mean()),
        p50_response_ms=("response_time_ms", "median"),
        p95_response_ms=("response_time_ms", lambda s: s.quantile(0.95)),
        mean_response_ms=("response_time_ms", "mean"),
    ).reset_index()
    k["sensor_jitter"] = g["vib_rms"].apply(_jitter).to_numpy()
    k["fw_errors_per_100"] = 100 * k["fw_errors"] / k["n_intervals"]
    k["reboots_per_100"] = 100 * k["reboots"] / k["n_intervals"]
    return k


def fleet_kpis_by_version(tel: pd.DataFrame) -> pd.DataFrame:
    """Headline numbers per firmware version (interval-weighted)."""
    g = tel.groupby("firmware_version")
    return pd.DataFrame({
        "devices": g["device_id"].nunique(),
        "device_hours": g.size() * 10 / 60,
        "fw_errors_per_100": 100 * g["fw_error_count"].mean(),
        "reboots_per_100": 100 * g["reboot"].mean(),
        "uptime_pct": g["uptime_pct"].mean(),
        "alarm_rate_pct": 100 * g["vibration_alarm"].mean(),
        "hw_fault_rate_pct": 100 * g["hardware_fault"].mean(),
        "p50_response_ms": g["response_time_ms"].median(),
        "p95_response_ms": g["response_time_ms"].quantile(0.95),
    }).reset_index()


def fleet_timeseries(tel: pd.DataFrame, bucket_intervals: int = 6) -> pd.DataFrame:
    """Fleet KPIs per time bucket and arm (default bucket = 1 hour), for trend charts."""
    t = tel.assign(bucket=tel["interval_idx"] // bucket_intervals * bucket_intervals)
    g = t.groupby(["bucket", "group", "hw_revision"])
    out = g.agg(timestamp=("timestamp", "min"), fw_errors_per_100=("fw_error_count", lambda s: 100 * s.mean()),
                uptime_pct=("uptime_pct", "mean"), p50_response_ms=("response_time_ms", "median"),
                mean_rms=("vib_rms", "mean"), alarm_rate_pct=("vibration_alarm", lambda s: 100 * s.mean()),
                reboots=("reboot", "sum")).reset_index()
    return out

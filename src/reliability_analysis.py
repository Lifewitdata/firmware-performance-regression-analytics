"""Reliability: MTBF, availability, survival to first reboot, and real bearing degradation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.experiment_design import INTERVAL_MIN, ROLLOUT_AT


def mtbf_table(tel: pd.DataFrame, by=("firmware_version", "hw_revision")) -> pd.DataFrame:
    """Firmware-attributable failures (reboots) vs hardware faults (from the real bearing signal), kept separate."""
    g = tel.groupby(list(by))
    out = pd.DataFrame({
        "devices": g["device_id"].nunique(),
        "operating_hours": g["uptime_pct"].sum() / 100 * INTERVAL_MIN / 60,
        "reboots": g["reboot"].sum(),
        "hw_fault_intervals": g["hardware_fault"].sum(),
        "availability_pct": g["uptime_pct"].mean(),
    }).reset_index()
    out["mtbf_reboot_hours"] = out["operating_hours"] / out["reboots"].replace(0, np.nan)
    # Exact Poisson (chi-square) 95% CI on the reboot rate, inverted to MTBF.
    lo_rate = stats.chi2.ppf(0.025, 2 * out["reboots"]) / 2 / out["operating_hours"]
    hi_rate = stats.chi2.ppf(0.975, 2 * (out["reboots"] + 1)) / 2 / out["operating_hours"]
    out["mtbf_ci_low"] = 1 / hi_rate
    out["mtbf_ci_high"] = 1 / lo_rate.replace(0, np.nan)
    return out


def kaplan_meier(durations, events) -> pd.DataFrame:
    d, e = np.asarray(durations, float), np.asarray(events, bool)
    times = np.unique(d[e])
    surv, var_acc, rows = 1.0, 0.0, [{"time": 0.0, "at_risk": len(d), "events": 0, "survival": 1.0,
                                       "ci_low": 1.0, "ci_high": 1.0}]
    for t in times:
        n = int((d >= t).sum())
        k = int(((d == t) & e).sum())
        surv *= 1 - k / n
        var_acc += k / (n * (n - k)) if n > k else 0.0
        se = surv * np.sqrt(var_acc)
        rows.append({"time": t, "at_risk": n, "events": k, "survival": surv,
                     "ci_low": max(0.0, surv - 1.96 * se), "ci_high": min(1.0, surv + 1.96 * se)})
    return pd.DataFrame(rows)


def logrank_test(d1, e1, d2, e2) -> dict:
    d = np.r_[d1, d2].astype(float)
    e = np.r_[e1, e2].astype(bool)
    grp = np.r_[np.zeros(len(d1)), np.ones(len(d2))]
    o_minus_e, var = 0.0, 0.0
    for t in np.unique(d[e]):
        at_risk = d >= t
        n, n1 = at_risk.sum(), (at_risk & (grp == 0)).sum()
        k = ((d == t) & e).sum()
        k1 = ((d == t) & e & (grp == 0)).sum()
        o_minus_e += k1 - k * n1 / n
        if n > 1:
            var += k * (n1 / n) * (1 - n1 / n) * (n - k) / (n - 1)
    chi2 = o_minus_e**2 / var if var > 0 else 0.0
    return {"chi2": float(chi2), "p_value": float(stats.chi2.sf(chi2, 1))}


def time_to_first_reboot(tel: pd.DataFrame) -> pd.DataFrame:
    """Hours from rollout to first reboot per device (censored at end of observation)."""
    post = tel[tel.interval_idx >= ROLLOUT_AT]
    horizon = post.interval_idx.max() - ROLLOUT_AT + 1
    first = post[post.reboot].groupby("device_id").interval_idx.min() - ROLLOUT_AT
    dev = post.groupby("device_id").agg(group=("group", "first"), hw_revision=("hw_revision", "first"),
                                        firmware_version=("firmware_version", "last"))
    dev["event"] = dev.index.isin(first.index)
    dev["duration_h"] = first.reindex(dev.index).fillna(horizon).to_numpy() * INTERVAL_MIN / 60
    return dev.reset_index()


def survival_by(ttf: pd.DataFrame, by: str = "firmware_version") -> dict[str, pd.DataFrame]:
    return {k: kaplan_meier(g.duration_h, g.event) for k, g in ttf.groupby(by)}


# ------------------------------------------------------------------ real IMS degradation

def bearing_health(features: pd.DataFrame, healthy_share: float = 0.2, threshold: float = 2.5) -> pd.DataFrame:
    """Health index = RMS / healthy-period median RMS, per bearing, with hours to threshold crossing."""
    out = []
    for (test, bearing), g in features.groupby(["test_id", "bearing"]):
        g = g.sort_values("timestamp").copy()
        base = g["rms"].iloc[: max(1, int(len(g) * healthy_share))].median()
        g["health_index"] = g["rms"] / base
        g["hours"] = (g["timestamp"] - g["timestamp"].iloc[0]).dt.total_seconds() / 3600
        g["health_index_smooth"] = g["health_index"].rolling(12, min_periods=1).median()
        out.append(g)
    return pd.concat(out, ignore_index=True)


def degradation_summary(health: pd.DataFrame, threshold: float = 2.5, kurtosis_threshold: float = 6.0) -> pd.DataFrame:
    """Two early-warning signals: RMS growth (energy) and kurtosis (impulsiveness, catches early spalls)."""
    rows = []
    for (test, bearing), g in health.groupby(["test_id", "bearing"]):
        crossed = g[g.health_index_smooth > threshold]
        kurt = g["kurtosis"].rolling(12, min_periods=1).median()
        k_crossed = g[kurt > kurtosis_threshold]
        rows.append({"test_id": test, "bearing": bearing, "failure_mode": g.failure_mode.iloc[0],
                     "run_hours": g.hours.max(),
                     "hours_to_rms_threshold": crossed.hours.iloc[0] if len(crossed) else np.nan,
                     "hours_to_kurtosis_threshold": k_crossed.hours.iloc[0] if len(k_crossed) else np.nan,
                     "final_health_index": g.health_index_smooth.iloc[-1],
                     "max_kurtosis": g["kurtosis"].max()})
    s = pd.DataFrame(rows)
    first = s[["hours_to_rms_threshold", "hours_to_kurtosis_threshold"]].min(axis=1)
    s["warning_lead_hours"] = s.run_hours - first
    return s

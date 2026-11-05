"""Device-level regression flags, change-point detection and the rollout decision."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.performance_metrics import METRIC_BY_NAME, METRICS
from src.statistical_tests import compare_by_segment, compare_firmware, device_deltas

DEVICE_Z_THRESHOLD = 3.0


def robust_z(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    med = np.median(ref)
    mad = 1.4826 * np.median(np.abs(ref - med))
    if mad == 0:
        mad = np.std(ref) or 1.0
    return (x - med) / mad


def device_regression_scores(kpis: pd.DataFrame, metrics=None, threshold: float = DEVICE_Z_THRESHOLD) -> pd.DataFrame:
    """For each treated device, how unusual is its pre->post change relative to control devices?

    Scores are signed so that positive = worse, regardless of metric direction.
    """
    metrics = metrics or [m.name for m in METRICS]
    out = None
    for m in metrics:
        w = device_deltas(kpis, m).dropna(subset=["delta"])
        ref = w.loc[w.group == "control", "delta"].to_numpy()
        t = w[w.group == "treatment"].copy()
        sign = -1 if METRIC_BY_NAME[m].higher_is_better else 1
        t[f"z_{m}"] = sign * robust_z(t["delta"].to_numpy(), ref)
        t = t[["device_id", "hw_revision", "region", f"z_{m}"]]
        out = t if out is None else out.merge(t, on=["device_id", "hw_revision", "region"])
    zcols = [c for c in out.columns if c.startswith("z_")]
    guard = [f"z_{m}" for m in metrics if METRIC_BY_NAME[m].guardrail]
    out["worst_metric"] = out[zcols].idxmax(axis=1).str.removeprefix("z_")
    out["worst_z"] = out[zcols].max(axis=1)
    out["n_metrics_flagged"] = (out[zcols] > threshold).sum(axis=1)
    out["regression_flag"] = (out[guard] > threshold).any(axis=1)
    return out.sort_values("worst_z", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ change-point / anomaly detection

def cusum(values: np.ndarray, baseline_n: int, k: float = 0.5, h: float = 8.0) -> dict:
    """Two-sided tabular CUSUM in baseline-sigma units. Returns first alarm index and both statistics."""
    x = np.asarray(values, float)
    mu, sd = np.mean(x[:baseline_n]), np.std(x[:baseline_n]) or 1.0
    z = (x - mu) / sd
    hi, lo = np.zeros_like(z), np.zeros_like(z)
    for i in range(1, len(z)):
        hi[i] = max(0.0, hi[i - 1] + z[i] - k)
        lo[i] = max(0.0, lo[i - 1] - z[i] - k)
    alarms = np.flatnonzero((hi > h) | (lo > h))
    first = int(alarms[0]) if len(alarms) else None
    # Change-point estimate: last time the triggering statistic was zero before the alarm.
    cp = None
    if first is not None:
        stat = hi if hi[first] > h else lo
        zeros = np.flatnonzero(stat[:first] == 0)
        cp = int(zeros[-1]) + 1 if len(zeros) else 0
    return {"first_alarm": first, "change_point": cp, "cusum_hi": hi, "cusum_lo": lo}


def rolling_zscore_alarms(values: np.ndarray, window: int = 288, threshold: float = 4.0) -> np.ndarray:
    s = pd.Series(values, dtype=float)
    mu = s.rolling(window, min_periods=window // 2).mean().shift(1)
    sd = s.rolling(window, min_periods=window // 2).std().shift(1)
    return ((s - mu).abs() / sd > threshold).fillna(False).to_numpy()


def evaluate_detector_on_nab(nab: pd.DataFrame, window: int = 288, threshold: float = 4.0,
                             debounce: int = 12) -> pd.DataFrame:
    """Window-level precision/recall of the rolling z-score detector against NAB's labelled anomaly windows."""
    rows = []
    for name, s in nab.groupby("series", sort=False):
        s = s.sort_values("timestamp").reset_index(drop=True)
        raw = rolling_zscore_alarms(s["value"].to_numpy(), window, threshold)
        idx = np.flatnonzero(raw)
        events = [i for j, i in enumerate(idx) if j == 0 or i - idx[j - 1] > debounce]
        in_win = s["in_anomaly_window"].to_numpy()
        win_id = (in_win & ~np.r_[False, in_win[:-1]]).cumsum() * in_win
        n_windows = int(win_id.max())
        detected = {int(win_id[i]) for i in events if in_win[i]}
        fp = sum(1 for i in events if not in_win[i])
        tp = len(detected)
        rows.append({"series": name, "anomaly_windows": n_windows, "windows_detected": tp,
                     "false_alarm_events": fp, "recall": tp / n_windows if n_windows else np.nan,
                     "precision": tp / (tp + fp) if tp + fp else np.nan})
    return pd.DataFrame(rows)


def fleet_change_point(tel: pd.DataFrame, metric: str, hw_revision: str | None = None,
                       baseline_n: int = 100, bucket: int = 3) -> dict:
    """CUSUM on the treatment-minus-control gap; returns the detected change interval."""
    t = tel if hw_revision is None else tel[tel.hw_revision == hw_revision]
    s = (t.assign(b=t.interval_idx // bucket).groupby(["b", "group"])[metric].mean().unstack())
    gap = (s["treatment"] - s["control"]).to_numpy()
    r = cusum(gap, baseline_n // bucket)
    to_interval = lambda v: None if v is None else int(v * bucket)
    return {"metric": metric, "hw_revision": hw_revision or "all", "gap": gap, "bucket": bucket,
            "first_alarm_interval": to_interval(r["first_alarm"]), "change_point_interval": to_interval(r["change_point"])}


# ------------------------------------------------------------------ decision

ROLLBACK_REL_CHANGE_PCT = 20.0
INVESTIGATE_FLAGGED_SHARE = 0.10


def segment_decision(results: pd.DataFrame, flagged_share: float) -> tuple[str, list[str]]:
    reasons = []
    regressed = results[results.verdict == "regressed"]
    severe = regressed[regressed.guardrail & (regressed.relative_change_pct.abs() >= ROLLBACK_REL_CHANGE_PCT)]
    uptime = regressed[regressed.metric == "uptime_pct"]
    for _, r in severe.iterrows():
        reasons.append(f"{r.label}: {r.relative_change_pct:+.1f}% vs control (adj. p={r.p_value_adj:.3g})")
    for _, r in uptime.iterrows():
        if r.metric not in set(severe.metric):
            reasons.append(f"{r.label}: {r.did:+.2f} pp vs control (adj. p={r.p_value_adj:.3g})")
    if reasons:
        return "ROLLBACK", reasons
    if len(regressed):
        reasons += [f"{r.label} regressed {r.relative_change_pct:+.1f}% (adj. p={r.p_value_adj:.3g})"
                    for _, r in regressed.iterrows()]
    if flagged_share > INVESTIGATE_FLAGGED_SHARE:
        reasons.append(f"{flagged_share:.0%} of updated devices individually flagged")
    if reasons:
        return "INVESTIGATE", reasons
    improved = results[results.verdict == "improved"]
    return "CONTINUE", [f"{r.label} improved {r.relative_change_pct:+.1f}%" for _, r in improved.iterrows()] or \
        ["No significant regressions detected"]


def rollout_decision(kpis: pd.DataFrame, segment: str = "hw_revision") -> dict:
    overall = compare_firmware(kpis)
    by_seg = compare_by_segment(kpis, segment)
    scores = device_regression_scores(kpis)

    decisions = {}
    d, why = segment_decision(overall, scores.regression_flag.mean())
    decisions["fleet"] = {"decision": d, "reasons": why}
    for seg, res in by_seg.groupby("segment"):
        val = seg.split("=", 1)[1]
        share = scores.loc[scores[segment] == val, "regression_flag"].mean()
        d, why = segment_decision(res, share)
        decisions[seg] = {"decision": d, "reasons": why}

    seg_decisions = {k: v["decision"] for k, v in decisions.items() if k != "fleet"}
    bad = sorted(k for k, v in seg_decisions.items() if v != "CONTINUE")
    if not bad:
        headline = "CONTINUE deployment to the full fleet."
    elif len(bad) == len(seg_decisions):
        headline = f"{decisions['fleet']['decision']} v2.0 fleet-wide."
    else:
        good = sorted(k for k, v in seg_decisions.items() if v == "CONTINUE")
        headline = (f"Segment-specific: {', '.join(f'{k} -> {seg_decisions[k]}' for k in bad)}; "
                    f"CONTINUE for {', '.join(good)}.")
    return {"headline": headline, "decisions": decisions, "overall": overall, "by_segment": by_seg,
            "device_scores": scores}

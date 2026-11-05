"""Statistical comparison of firmware versions.

The primary estimator is difference-in-differences (DiD) on device-level KPIs: each device is its own
baseline (pre vs post rollout) and the control arm absorbs fleet-wide drift such as bearing wear.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from src.performance_metrics import METRIC_BY_NAME, METRICS


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((b.mean() - a.mean()) / pooled) if pooled > 0 else 0.0


def welch_ttest(a, b) -> tuple[float, float]:
    res = stats.ttest_ind(b, a, equal_var=False)
    return float(res.statistic), float(res.pvalue)


def mann_whitney(a, b) -> float:
    return float(stats.mannwhitneyu(b, a, alternative="two-sided").pvalue)


def bootstrap_diff_ci(a, b, stat=np.mean, n_boot: int = 4000, ci: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Percentile CI for stat(b) - stat(a)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    ia = rng.integers(0, len(a), (n_boot, len(a)))
    ib = rng.integers(0, len(b), (n_boot, len(b)))
    diffs = stat(b[ib], axis=1) - stat(a[ia], axis=1)
    lo, hi = np.quantile(diffs, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(lo), float(hi)


def poisson_rate_test(count_a: int, exposure_a: float, count_b: int, exposure_b: float) -> dict:
    """Exact conditional test of rate_b / rate_a = 1 (binomial on the split of total events)."""
    total = count_a + count_b
    if total == 0:
        return {"rate_ratio": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": 1.0}
    p0 = exposure_b / (exposure_a + exposure_b)
    test = stats.binomtest(count_b, total, p0)
    lo, hi = test.proportion_ci(0.95, method="exact")
    to_rr = lambda p: (p / (1 - p)) * (exposure_a / exposure_b) if p < 1 else np.inf
    rr = (count_b / exposure_b) / (count_a / exposure_a) if count_a else np.inf
    return {"rate_ratio": rr, "ci_low": to_rr(lo), "ci_high": to_rr(hi), "p_value": float(test.pvalue)}


def two_proportion_ztest(x_a: int, n_a: int, x_b: int, n_b: int) -> tuple[float, float]:
    p = (x_a + x_b) / (n_a + n_b)
    se = np.sqrt(p * (1 - p) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 0.0, 1.0
    z = (x_b / n_b - x_a / n_a) / se
    return float(z), float(2 * stats.norm.sf(abs(z)))


def benjamini_hochberg(pvals) -> np.ndarray:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.clip(adj, 0, 1)
    return out


def device_deltas(kpis: pd.DataFrame, metric: str) -> pd.DataFrame:
    w = kpis.pivot_table(index=["device_id", "group", "hw_revision", "region"], columns="period",
                         values=metric).reset_index()
    w["delta"] = w["post"] - w["pre"]
    return w


def did_estimate(kpis: pd.DataFrame, metric: str, seed: int = 0) -> dict:
    """DiD = mean(post-pre | treatment) - mean(post-pre | control), with Welch test, bootstrap CI and OLS check."""
    w = device_deltas(kpis, metric).dropna(subset=["delta"])
    t, c = w.loc[w.group == "treatment", "delta"].to_numpy(), w.loc[w.group == "control", "delta"].to_numpy()
    did = t.mean() - c.mean()
    _, p = welch_ttest(c, t)
    lo, hi = bootstrap_diff_ci(c, t, seed=seed)

    long = kpis[["device_id", "group", "period", metric]].dropna()
    long = long.assign(treated=(long.group == "treatment").astype(int), post=(long.period == "post").astype(int))
    ols = smf.ols(f"{metric} ~ treated * post", data=long).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(long["device_id"])[0]})

    pre_t = w.loc[w.group == "treatment", "pre"].mean()
    return {
        "metric": metric, "n_treatment": len(t), "n_control": len(c),
        "treatment_pre": pre_t, "treatment_post": w.loc[w.group == "treatment", "post"].mean(),
        "control_pre": w.loc[w.group == "control", "pre"].mean(),
        "control_post": w.loc[w.group == "control", "post"].mean(),
        "did": did, "ci_low": lo, "ci_high": hi, "relative_change_pct": 100 * did / pre_t if pre_t else np.nan,
        "p_value": p, "p_value_mwu": mann_whitney(c, t), "p_value_ols_cluster": float(ols.pvalues["treated:post"]),
        "cohens_d": cohens_d(c, t),
    }


def naive_comparison(kpis: pd.DataFrame, metric: str) -> dict:
    """What you'd conclude without a control arm: treated devices before vs after the update."""
    t = device_deltas(kpis, metric).query("group == 'treatment'")
    _, p = stats.ttest_rel(t["post"], t["pre"])
    return {"metric": metric, "pre": t["pre"].mean(), "post": t["post"].mean(),
            "naive_diff": (t["post"] - t["pre"]).mean(), "p_value": float(p)}


def verdict(row: pd.Series, alpha: float = 0.05) -> str:
    m = METRIC_BY_NAME[row["metric"]]
    if row["p_value_adj"] >= alpha:
        return "no significant change"
    better = (row["did"] > 0) == m.higher_is_better
    return "improved" if better else "regressed"


def compare_firmware(kpis: pd.DataFrame, metrics=None, alpha: float = 0.05) -> pd.DataFrame:
    metrics = metrics or [m.name for m in METRICS]
    res = pd.DataFrame([did_estimate(kpis, m) for m in metrics])
    res["p_value_adj"] = benjamini_hochberg(res["p_value"])
    res["verdict"] = res.apply(verdict, axis=1, alpha=alpha)
    res["label"] = res["metric"].map(lambda n: METRIC_BY_NAME[n].label)
    res["guardrail"] = res["metric"].map(lambda n: METRIC_BY_NAME[n].guardrail)
    return res


def compare_by_segment(kpis: pd.DataFrame, segment: str, metrics=None) -> pd.DataFrame:
    out = []
    for value, sub in kpis.groupby(segment):
        if sub.group.nunique() < 2:
            continue
        r = compare_firmware(sub, metrics)
        r.insert(0, "segment", f"{segment}={value}")
        out.append(r)
    return pd.concat(out, ignore_index=True)


def event_rate_tests(tel: pd.DataFrame) -> pd.DataFrame:
    """Interval-level exact tests for count events in the post period: v2.0 (treated) vs v1.0 (control)."""
    post = tel[tel.period == "post"]
    rows = []
    for seg, sub in [("all", post), *[(f"hw_revision={r}", s) for r, s in post.groupby("hw_revision")]]:
        a, b = sub[sub.group == "control"], sub[sub.group == "treatment"]
        for col in ("fw_error_count", "reboot"):
            r = poisson_rate_test(int(a[col].sum()), len(a), int(b[col].sum()), len(b))
            rows.append({"segment": seg, "event": col, "v1_events": int(a[col].sum()), "v1_intervals": len(a),
                         "v2_events": int(b[col].sum()), "v2_intervals": len(b), **r})
    return pd.DataFrame(rows)

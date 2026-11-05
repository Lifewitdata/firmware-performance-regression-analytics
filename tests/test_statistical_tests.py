import numpy as np
import pandas as pd
import pytest

from src.statistical_tests import (benjamini_hochberg, bootstrap_diff_ci, cohens_d, did_estimate,
                                   poisson_rate_test, two_proportion_ztest)


def test_benjamini_hochberg_matches_reference():
    p = [0.01, 0.04, 0.03, 0.005]
    np.testing.assert_allclose(benjamini_hochberg(p), [0.02, 0.04, 0.04, 0.02])


def test_poisson_rate_test_detects_doubling():
    r = poisson_rate_test(100, 1000, 200, 1000)
    assert r["rate_ratio"] == pytest.approx(2.0)
    assert r["ci_low"] < 2.0 < r["ci_high"]
    assert r["p_value"] < 1e-6
    assert poisson_rate_test(50, 1000, 52, 1000)["p_value"] > 0.5


def test_two_proportion_null():
    z, p = two_proportion_ztest(50, 1000, 50, 1000)
    assert z == 0 and p == pytest.approx(1.0)


def test_bootstrap_ci_covers_true_difference():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, 400), rng.normal(0.5, 1, 400)
    lo, hi = bootstrap_diff_ci(a, b)
    assert lo < 0.5 < hi
    assert cohens_d(a, b) == pytest.approx(0.5, abs=0.15)


def _toy_kpis(effect: float, drift: float, n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        arm = "treatment" if i < n // 2 else "control"
        base = rng.normal(10, 1)
        post = base + drift + (effect if arm == "treatment" else 0) + rng.normal(0, 0.3)
        for period, v in (("pre", base), ("post", post)):
            rows.append({"device_id": f"d{i}", "group": arm, "hw_revision": "A", "region": "x",
                         "period": period, "m": v})
    return pd.DataFrame(rows)


def test_did_removes_common_drift():
    r = did_estimate(_toy_kpis(effect=0.0, drift=3.0), "m")
    assert abs(r["did"]) < 0.3 and r["p_value"] > 0.01
    r = did_estimate(_toy_kpis(effect=1.0, drift=3.0), "m")
    assert r["did"] == pytest.approx(1.0, abs=0.3) and r["p_value"] < 1e-6
    assert r["p_value_ols_cluster"] < 1e-6

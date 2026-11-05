import numpy as np
import pandas as pd
import pytest

from src.experiment_design import build_fleet, generate_telemetry
from src.performance_metrics import device_period_kpis


@pytest.fixture(scope="session")
def features() -> pd.DataFrame:
    """Synthetic stand-in for IMS features: 12 bearings x 1440 snapshots, two of them degrading."""
    rng = np.random.default_rng(0)
    rows = []
    for test in ("test1", "test2", "test3"):
        for b in range(1, 5):
            n = 1440
            failing = (test, b) in {("test1", 3), ("test2", 1)}
            trend = np.where(failing, np.linspace(1, 4, n) ** 2, 1.0)
            rms = 0.08 * trend * rng.lognormal(0, 0.05, n)
            rows.append(pd.DataFrame({
                "test_id": test, "bearing": b,
                "timestamp": pd.date_range("2004-01-01", periods=n, freq="10min"),
                "snapshot_idx": np.arange(n), "rms": rms, "peak": rms * 5,
                "kurtosis": 3 + rng.normal(0, 0.2, n) + np.where(failing, np.linspace(0, 6, n), 0),
                "failure_mode": "outer race failure" if failing else "none", "bearing_failed": failing,
            }))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="session")
def fleet(features):
    return build_fleet(features, seed=1)


@pytest.fixture(scope="session")
def telemetry(fleet, features):
    latency = 45 + np.random.default_rng(2).normal(0, 2, 4000)
    return generate_telemetry(fleet, features, latency, seed=3)


@pytest.fixture(scope="session")
def kpis(telemetry):
    return device_period_kpis(telemetry)

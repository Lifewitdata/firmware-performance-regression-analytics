"""Load raw NASA IMS bearing snapshots and NAB series into tidy, analysis-ready tables."""
from __future__ import annotations

import json
import os

# One BLAS thread per worker process; the default pool per process exhausts memory under multiprocessing.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
IMS_DIR = DATA / "nasa_ims"
NAB_DIR = DATA / "nab"
PROCESSED = DATA / "processed"

SAMPLE_RATE_HZ = 20_000


@dataclass(frozen=True)
class IMSTest:
    test_id: str
    folder: str
    n_channels: int
    # Column index used for each bearing. Test 1 records x and y per bearing; we use x.
    bearing_channels: tuple[int, int, int, int]
    failed_bearings: dict[int, str]


# Source: "Readme Document for IMS Bearing Data.pdf" (NASA PCoE / Univ. of Cincinnati IMS).
IMS_TESTS = (
    IMSTest("test1", "1st_test", 8, (0, 2, 4, 6),
            {3: "inner race defect", 4: "roller element defect"}),
    IMSTest("test2", "2nd_test", 4, (0, 1, 2, 3), {1: "outer race failure"}),
    IMSTest("test3", "4th_test/txt", 4, (0, 1, 2, 3), {3: "outer race failure"}),
)


def parse_snapshot_time(name: str) -> datetime:
    return datetime.strptime(name, "%Y.%m.%d.%H.%M.%S")


def read_snapshot(path: Path) -> np.ndarray:
    return pd.read_csv(path, sep="\t", header=None, dtype=np.float64, engine="c").to_numpy()


def signal_features(x: np.ndarray) -> dict[str, float]:
    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(np.abs(x)))
    spectrum = np.abs(np.fft.rfft(x - x.mean())) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1 / SAMPLE_RATE_HZ)
    total = spectrum.sum()
    return {
        "rms": rms,
        "peak": peak,
        "peak_to_peak": float(x.max() - x.min()),
        "std": float(x.std()),
        "kurtosis": float(stats.kurtosis(x, fisher=False)),
        "skewness": float(stats.skew(x)),
        "crest_factor": peak / rms if rms > 0 else np.nan,
        # Share of energy above 2 kHz, where bearing defect harmonics and resonances show up early.
        "hf_energy_ratio": float(spectrum[freqs >= 2000].sum() / total) if total > 0 else np.nan,
    }


def _features_for_file(args: tuple[str, str, tuple[int, ...]]) -> list[dict]:
    path_str, test_id, channels = args
    path = Path(path_str)
    data = read_snapshot(path)
    ts = parse_snapshot_time(path.name)
    rows = []
    for bearing, ch in enumerate(channels, start=1):
        rows.append({"test_id": test_id, "bearing": bearing, "timestamp": ts,
                     **signal_features(data[:, ch])})
    return rows


def list_snapshots(test: IMSTest, ims_dir: Path = IMS_DIR) -> list[Path]:
    folder = ims_dir / test.folder
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.name[:4].isdigit())


def build_ims_features(ims_dir: Path = IMS_DIR, workers: int = min(4, os.cpu_count() or 1),
                       limit_per_test: int | None = None) -> pd.DataFrame:
    """Compute per-snapshot, per-bearing vibration features for every IMS test run."""
    jobs = []
    for test in IMS_TESTS:
        files = list_snapshots(test, ims_dir)
        if limit_per_test:
            files = files[:limit_per_test]
        jobs += [(str(p), test.test_id, test.bearing_channels) for p in files]
    if not jobs:
        raise FileNotFoundError(f"No IMS snapshot files found under {ims_dir}. Extract the .rar archives first.")

    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [r for chunk in pool.map(_features_for_file, jobs, chunksize=32) for r in chunk]

    df = pd.DataFrame(rows).sort_values(["test_id", "bearing", "timestamp"]).reset_index(drop=True)
    df["snapshot_idx"] = df.groupby(["test_id", "bearing"]).cumcount()
    failed = {(t.test_id, b): mode for t in IMS_TESTS for b, mode in t.failed_bearings.items()}
    df["failure_mode"] = [failed.get(k, "none") for k in zip(df["test_id"], df["bearing"])]
    df["bearing_failed"] = df["failure_mode"] != "none"
    return df


def load_ims_features(path: Path = PROCESSED / "ims_features.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)


# ---------------------------------------------------------------- NAB

NAB_SERIES = {
    "machine_temperature_system_failure": "realKnownCause",
    "ec2_request_latency_system_failure": "realKnownCause",
    "ambient_temperature_system_failure": "realKnownCause",
    "cpu_utilization_asg_misconfiguration": "realKnownCause",
    "ec2_cpu_utilization_825cc2": "realAWSCloudwatch",
}


def load_nab_series(name: str, nab_dir: Path = NAB_DIR) -> pd.DataFrame:
    """Return a NAB series with `in_anomaly_window` and `is_labelled_anomaly` flags from the official labels."""
    df = pd.read_csv(nab_dir / f"{name}.csv", parse_dates=["timestamp"])
    key = f"{NAB_SERIES[name]}/{name}.csv"
    windows = json.loads((nab_dir / "labels" / "combined_windows.json").read_text())[key]
    labels = json.loads((nab_dir / "labels" / "combined_labels.json").read_text())[key]
    df["in_anomaly_window"] = False
    for start, end in windows:
        df.loc[df["timestamp"].between(pd.Timestamp(start), pd.Timestamp(end)), "in_anomaly_window"] = True
    df["is_labelled_anomaly"] = df["timestamp"].isin(pd.to_datetime(labels))
    df["series"] = name
    return df


def load_all_nab(nab_dir: Path = NAB_DIR) -> pd.DataFrame:
    return pd.concat([load_nab_series(n, nab_dir) for n in NAB_SERIES], ignore_index=True)


if __name__ == "__main__":
    import time

    PROCESSED.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    features = build_ims_features()
    features.to_parquet(PROCESSED / "ims_features.parquet", index=False)
    print(f"IMS features: {len(features):,} rows in {time.time() - t0:.0f}s")
    print(features.groupby("test_id").agg(snapshots=("snapshot_idx", "max"), start=("timestamp", "min"),
                                          end=("timestamp", "max")))

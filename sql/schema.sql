-- SQLite schema for firmware performance & regression analytics.
-- Load with: python -m src.pipeline (creates data/processed/firmware_analytics.db)

DROP VIEW IF EXISTS v_device_period_kpis;
DROP TABLE IF EXISTS telemetry;
DROP TABLE IF EXISTS devices;
DROP TABLE IF EXISTS firmware_releases;
DROP TABLE IF EXISTS ims_features;

CREATE TABLE firmware_releases (
    firmware_version TEXT PRIMARY KEY,
    release_date     TEXT NOT NULL,
    status           TEXT NOT NULL,
    notes            TEXT
);

-- Experimental device fleet. Each device is a 40-hour window of one real IMS bearing.
CREATE TABLE devices (
    device_id            TEXT PRIMARY KEY,
    source_test          TEXT NOT NULL,      -- real: IMS test run
    source_bearing       INTEGER NOT NULL,   -- real: bearing 1-4
    window_index         INTEGER NOT NULL,
    source_start_idx     INTEGER NOT NULL,
    source_start_time    TEXT NOT NULL,
    failure_mode         TEXT NOT NULL,      -- real: from IMS readme
    contains_end_of_life INTEGER NOT NULL,
    hw_revision          TEXT NOT NULL CHECK (hw_revision IN ('A', 'B')),   -- experimental
    region               TEXT NOT NULL,                                       -- experimental
    arm                  TEXT NOT NULL CHECK (arm IN ('treatment', 'control')) -- experimental
);

CREATE TABLE telemetry (
    device_id         TEXT NOT NULL REFERENCES devices(device_id),
    interval_idx      INTEGER NOT NULL,
    timestamp         TEXT NOT NULL,
    firmware_version  TEXT NOT NULL REFERENCES firmware_releases(firmware_version),
    period            TEXT NOT NULL CHECK (period IN ('pre', 'post')),
    arm               TEXT NOT NULL,
    hw_revision       TEXT NOT NULL,
    region            TEXT NOT NULL,
    source_timestamp  TEXT NOT NULL,
    real_rms          REAL NOT NULL,
    real_kurtosis     REAL NOT NULL,
    real_peak         REAL NOT NULL,
    vib_rms           REAL NOT NULL,
    vib_kurtosis      REAL NOT NULL,
    vib_peak          REAL NOT NULL,
    response_time_ms  REAL NOT NULL,
    fw_error_count    INTEGER NOT NULL,
    vibration_alarm   INTEGER NOT NULL,
    hardware_fault    INTEGER NOT NULL,
    reboot            INTEGER NOT NULL,
    downtime_min      REAL NOT NULL,
    uptime_pct        REAL NOT NULL,
    PRIMARY KEY (device_id, interval_idx)
);
CREATE INDEX ix_telemetry_fw ON telemetry (firmware_version, hw_revision);
CREATE INDEX ix_telemetry_period ON telemetry (arm, period);

-- Real per-snapshot vibration features from the NASA IMS bearing run-to-failure tests.
CREATE TABLE ims_features (
    test_id         TEXT NOT NULL,
    bearing         INTEGER NOT NULL,
    timestamp       TEXT NOT NULL,
    snapshot_idx    INTEGER NOT NULL,
    rms             REAL, peak REAL, peak_to_peak REAL, std REAL,
    kurtosis        REAL, skewness REAL, crest_factor REAL, hf_energy_ratio REAL,
    failure_mode    TEXT,
    bearing_failed  INTEGER,
    PRIMARY KEY (test_id, bearing, snapshot_idx)
);

CREATE VIEW v_device_period_kpis AS
SELECT device_id, period, arm, hw_revision, region,
       MAX(firmware_version)                       AS firmware_version,
       COUNT(*)                                    AS n_intervals,
       100.0 * SUM(fw_error_count) / COUNT(*)      AS fw_errors_per_100,
       100.0 * SUM(reboot) / COUNT(*)              AS reboots_per_100,
       AVG(uptime_pct)                             AS uptime_pct,
       AVG(vib_rms)                                AS mean_rms,
       100.0 * AVG(vibration_alarm)                AS alarm_rate_pct,
       100.0 * AVG(hardware_fault)                 AS hw_fault_rate_pct,
       AVG(response_time_ms)                       AS mean_response_ms
FROM telemetry
GROUP BY device_id, period;

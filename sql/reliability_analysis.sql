-- Reliability: firmware-attributable failures vs real hardware degradation.

-- name: mtbf_by_firmware
SELECT firmware_version, hw_revision,
       ROUND(SUM(uptime_pct) / 100.0 * 10 / 60, 1)  AS operating_hours,
       SUM(reboot)                                   AS reboots,
       ROUND(SUM(uptime_pct) / 100.0 * 10 / 60 / NULLIF(SUM(reboot), 0), 1) AS mtbf_reboot_hours,
       ROUND(AVG(uptime_pct), 3)                     AS availability_pct
FROM telemetry
GROUP BY firmware_version, hw_revision
ORDER BY hw_revision, firmware_version;

-- name: time_to_first_reboot_after_rollout
SELECT d.device_id, d.arm, d.hw_revision,
       ROUND((MIN(CASE WHEN t.reboot = 1 THEN t.interval_idx END) - 120) * 10 / 60.0, 2) AS hours_to_first_reboot
FROM devices d
JOIN telemetry t ON t.device_id = d.device_id AND t.interval_idx >= 120
GROUP BY d.device_id
ORDER BY hours_to_first_reboot IS NULL, hours_to_first_reboot;

-- name: hardware_fault_attribution
-- Hardware faults come from the real bearing signal; they should NOT differ by firmware arm.
SELECT d.source_test, d.failure_mode, t.arm, t.period,
       COUNT(DISTINCT t.device_id)               AS devices,
       ROUND(100.0 * AVG(t.hardware_fault), 2)   AS hw_fault_rate_pct,
       ROUND(100.0 * AVG(t.reboot), 3)           AS reboot_rate_pct
FROM telemetry t JOIN devices d ON d.device_id = t.device_id
GROUP BY d.source_test, d.failure_mode, t.arm, t.period
ORDER BY d.source_test, d.failure_mode, t.arm, t.period;

-- name: bearing_degradation
-- Real IMS run-to-failure: RMS growth from the first to last 10% of each bearing's life.
WITH n AS (SELECT test_id, bearing, MAX(snapshot_idx) + 1 AS total FROM ims_features GROUP BY test_id, bearing)
SELECT f.test_id, f.bearing, MAX(f.failure_mode) AS failure_mode,
       ROUND(AVG(CASE WHEN f.snapshot_idx <  n.total * 0.1 THEN f.rms END), 4) AS early_rms,
       ROUND(AVG(CASE WHEN f.snapshot_idx >= n.total * 0.9 THEN f.rms END), 4) AS late_rms,
       ROUND(AVG(CASE WHEN f.snapshot_idx >= n.total * 0.9 THEN f.rms END)
           / AVG(CASE WHEN f.snapshot_idx <  n.total * 0.1 THEN f.rms END), 2) AS rms_growth_x,
       ROUND(MAX(f.kurtosis), 1) AS max_kurtosis
FROM ims_features f JOIN n ON n.test_id = f.test_id AND n.bearing = f.bearing
GROUP BY f.test_id, f.bearing
ORDER BY f.test_id, f.bearing;

-- Fleet performance KPIs.

-- name: kpis_by_firmware_and_revision
SELECT firmware_version, hw_revision,
       COUNT(DISTINCT device_id)                   AS devices,
       ROUND(COUNT(*) * 10.0 / 60, 1)              AS device_hours,
       ROUND(100.0 * SUM(fw_error_count) / COUNT(*), 3) AS fw_errors_per_100,
       ROUND(100.0 * SUM(reboot) / COUNT(*), 3)    AS reboots_per_100,
       ROUND(AVG(uptime_pct), 3)                   AS uptime_pct,
       ROUND(100.0 * AVG(vibration_alarm), 2)      AS alarm_rate_pct,
       ROUND(AVG(response_time_ms), 2)             AS mean_response_ms
FROM telemetry
GROUP BY firmware_version, hw_revision
ORDER BY hw_revision, firmware_version;

-- name: response_time_percentiles
WITH r AS (
    SELECT firmware_version, hw_revision, response_time_ms,
           PERCENT_RANK() OVER (PARTITION BY firmware_version, hw_revision ORDER BY response_time_ms) AS pr
    FROM telemetry
)
SELECT firmware_version, hw_revision,
       ROUND(MIN(CASE WHEN pr >= 0.50 THEN response_time_ms END), 2) AS p50_ms,
       ROUND(MIN(CASE WHEN pr >= 0.95 THEN response_time_ms END), 2) AS p95_ms,
       ROUND(MIN(CASE WHEN pr >= 0.99 THEN response_time_ms END), 2) AS p99_ms
FROM r
GROUP BY firmware_version, hw_revision
ORDER BY hw_revision, firmware_version;

-- name: hourly_fleet_trend
SELECT interval_idx / 6 AS hour, arm, hw_revision,
       ROUND(100.0 * AVG(fw_error_count), 3) AS fw_errors_per_100,
       ROUND(AVG(uptime_pct), 3)             AS uptime_pct,
       ROUND(AVG(response_time_ms), 2)       AS mean_response_ms
FROM telemetry
GROUP BY hour, arm, hw_revision
ORDER BY hour, arm, hw_revision;

-- name: sensor_stability
-- Mean absolute step-to-step change in RMS relative to its level (lower = steadier readings).
WITH s AS (
    SELECT device_id, period, arm, hw_revision, vib_rms,
           ABS(vib_rms - LAG(vib_rms) OVER (PARTITION BY device_id, period ORDER BY interval_idx)) AS step
    FROM telemetry
)
SELECT arm, period, hw_revision,
       ROUND(AVG(step) / AVG(vib_rms), 4) AS sensor_jitter
FROM s
GROUP BY arm, period, hw_revision
ORDER BY hw_revision, arm, period;

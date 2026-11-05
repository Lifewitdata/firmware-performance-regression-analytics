-- Firmware v2.0 vs v1.0: naive and difference-in-differences comparisons.
-- Each query is delimited by a "-- name:" marker so src/pipeline.py can run and export it.

-- name: arm_period_means
SELECT arm, period, hw_revision,
       COUNT(*)                     AS devices,
       ROUND(AVG(fw_errors_per_100), 3) AS fw_errors_per_100,
       ROUND(AVG(reboots_per_100), 3)   AS reboots_per_100,
       ROUND(AVG(uptime_pct), 3)        AS uptime_pct,
       ROUND(AVG(mean_rms), 4)          AS mean_rms,
       ROUND(AVG(mean_response_ms), 2)  AS mean_response_ms
FROM v_device_period_kpis
GROUP BY arm, period, hw_revision
ORDER BY hw_revision, arm, period;

-- name: did_by_hw_revision
-- DiD = (treatment post - treatment pre) - (control post - control pre)
WITH m AS (
    SELECT hw_revision, arm, period,
           AVG(fw_errors_per_100) AS err, AVG(reboots_per_100) AS reb,
           AVG(uptime_pct) AS up, AVG(mean_response_ms) AS resp, AVG(mean_rms) AS rms
    FROM v_device_period_kpis GROUP BY hw_revision, arm, period
)
SELECT t_post.hw_revision,
       ROUND((t_post.err - t_pre.err) - (c_post.err - c_pre.err), 3)   AS did_fw_errors_per_100,
       ROUND((t_post.reb - t_pre.reb) - (c_post.reb - c_pre.reb), 3)   AS did_reboots_per_100,
       ROUND((t_post.up - t_pre.up) - (c_post.up - c_pre.up), 3)       AS did_uptime_pct,
       ROUND((t_post.resp - t_pre.resp) - (c_post.resp - c_pre.resp), 2) AS did_response_ms,
       ROUND(100.0 * ((t_post.resp - t_pre.resp) - (c_post.resp - c_pre.resp)) / t_pre.resp, 1) AS did_response_pct,
       ROUND(100.0 * ((t_post.rms - t_pre.rms) - (c_post.rms - c_pre.rms)) / t_pre.rms, 1)      AS did_rms_pct
FROM m t_post
JOIN m t_pre  ON t_pre.hw_revision  = t_post.hw_revision AND t_pre.arm  = 'treatment' AND t_pre.period  = 'pre'
JOIN m c_post ON c_post.hw_revision = t_post.hw_revision AND c_post.arm = 'control'   AND c_post.period = 'post'
JOIN m c_pre  ON c_pre.hw_revision  = t_post.hw_revision AND c_pre.arm  = 'control'   AND c_pre.period  = 'pre'
WHERE t_post.arm = 'treatment' AND t_post.period = 'post'
ORDER BY t_post.hw_revision;

-- name: device_change_ranking
-- Treated devices ranked by their post-pre change in firmware errors relative to the control average.
WITH d AS (
    SELECT device_id, arm, hw_revision, region,
           SUM(CASE WHEN period = 'post' THEN fw_errors_per_100 ELSE -fw_errors_per_100 END) AS d_err,
           SUM(CASE WHEN period = 'post' THEN uptime_pct ELSE -uptime_pct END)               AS d_up,
           SUM(CASE WHEN period = 'post' THEN mean_response_ms ELSE -mean_response_ms END)   AS d_resp
    FROM v_device_period_kpis GROUP BY device_id
), c AS (
    SELECT AVG(d_err) AS c_err, AVG(d_up) AS c_up, AVG(d_resp) AS c_resp FROM d WHERE arm = 'control'
)
SELECT d.device_id, d.hw_revision, d.region,
       ROUND(d.d_err - c.c_err, 3)   AS excess_fw_errors_per_100,
       ROUND(d.d_up - c.c_up, 3)     AS excess_uptime_pct,
       ROUND(d.d_resp - c.c_resp, 2) AS excess_response_ms
FROM d CROSS JOIN c
WHERE d.arm = 'treatment'
ORDER BY excess_fw_errors_per_100 DESC
LIMIT 25;

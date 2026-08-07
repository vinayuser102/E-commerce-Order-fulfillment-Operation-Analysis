-- 2. Delivery Delay Analysis
-- Purpose: percentage of delayed vs on-time orders (overall), plus the delay
-- distribution per platform. Uses the derived boolean is_delayed.

WITH overall AS (
    SELECT
        COUNT(*)                                        AS n_orders,
        COUNT(*) FILTER (WHERE is_delayed)              AS n_delayed,
        ROUND(100.0 * COUNT(*) FILTER (WHERE is_delayed) / COUNT(*), 2) AS delay_pct
    FROM orders
)
SELECT
    'All platforms' AS platform,
    overall.delay_pct,
    overall.n_delayed,
    overall.n_orders
FROM overall

UNION ALL

SELECT
    "Platform" AS platform,
    ROUND(100.0 * COUNT(*) FILTER (WHERE is_delayed) / COUNT(*), 2) AS delay_pct,
    COUNT(*) FILTER (WHERE is_delayed)                              AS n_delayed,
    COUNT(*)                                                        AS n_orders
FROM orders
GROUP BY "Platform"

ORDER BY delay_pct DESC;

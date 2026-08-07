-- 4. Refund Behavior Analysis
-- Purpose: refund rate by service rating, and refund rate for delayed vs
-- on-time orders. Low ratings are the strongest driver; delay matters too.

-- 4a. Refund rate by service rating
SELECT
    "Service Rating"                                       AS rating,
    COUNT(*)                                               AS n_orders,
    COUNT(*) FILTER (WHERE "Refund Requested" = 'Yes')     AS n_refunds,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE "Refund Requested" = 'Yes') / COUNT(*),
        2
    )                                                      AS refund_rate_pct
FROM orders
GROUP BY "Service Rating"
ORDER BY "Service Rating";

-- 4b. Refund rate for delayed vs on-time orders
SELECT
    CASE WHEN is_delayed THEN 'Delayed' ELSE 'On time' END AS delivery_status,
    COUNT(*)                                               AS n_orders,
    COUNT(*) FILTER (WHERE "Refund Requested" = 'Yes')     AS n_refunds,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE "Refund Requested" = 'Yes') / COUNT(*),
        2
    )                                                      AS refund_rate_pct
FROM orders
GROUP BY is_delayed
ORDER BY refund_rate_pct DESC;

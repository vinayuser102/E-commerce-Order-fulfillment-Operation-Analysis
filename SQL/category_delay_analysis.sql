-- 3. Product Category Delay Analysis
-- Purpose: delay percentage per category, ranked from highest to lowest risk,
-- plus each category's share of ALL delayed orders.
-- Expected: Fruits & Vegetables and Personal Care lead; Grocery is lowest.

WITH category_stats AS (
    SELECT
        "Product Category"                                  AS category,
        COUNT(*)                                            AS n_orders,
        COUNT(*) FILTER (WHERE is_delayed)                  AS n_delayed,
        ROUND(100.0 * COUNT(*) FILTER (WHERE is_delayed) / COUNT(*), 2) AS delay_pct,
        SUM(COUNT(*) FILTER (WHERE is_delayed)) OVER ()     AS total_delayed
    FROM orders
    GROUP BY "Product Category"
)
SELECT
    category,
    n_orders,
    n_delayed,
    delay_pct,
    ROUND(100.0 * n_delayed / total_delayed, 2) AS share_of_all_delays_pct
FROM category_stats
ORDER BY delay_pct DESC;

-- Refund rate by service rating
SELECT
    service_rating,
    COUNT(*) AS order_count,
    ROUND(
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (),
        2
    ) AS percentage_of_orders,
    ROUND(
        SUM(CASE WHEN refund_requested = 'Yes' THEN 1 ELSE 0 END) * 100.0
        / COUNT(*),
        2
    ) AS refund_rate
FROM orders
GROUP BY service_rating
ORDER BY service_rating;
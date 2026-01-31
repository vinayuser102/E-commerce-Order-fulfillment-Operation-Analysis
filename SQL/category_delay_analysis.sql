-- Delivery delay by product category
SELECT
    product_category,
    delivery_delay,
    COUNT(*) AS order_count,
    ROUND(
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY product_category),
        2
    ) AS percentage
FROM orders
GROUP BY product_category, delivery_delay
ORDER BY product_category, percentage DESC;
-- 1. Data Overview
-- Purpose: validate the table before drawing conclusions.
-- Expected: total orders = 100,000; 3 platforms; 6 categories.

SELECT
    COUNT(*)                                    AS total_orders,
    COUNT(DISTINCT "Customer ID")               AS distinct_customers,
    COUNT(DISTINCT "Platform")                  AS distinct_platforms,
    COUNT(DISTINCT "Product Category")          AS distinct_categories,
    ROUND(AVG("Actual Delivery (Minutes)"), 2)  AS avg_delivery_minutes,
    ROUND(AVG("Order Value (INR)"), 2)          AS avg_order_value_inr
FROM orders;

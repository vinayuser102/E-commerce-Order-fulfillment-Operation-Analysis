-- Refund Risk Scoring Model
-- Weights:
-- Service Rating: 60%
-- Delivery Delay: 30%
-- Product Category: 10%

SELECT
    order_id,
    platform,
    product_category,
    service_rating,
    delivery_delay,
    refund_requested,

    ROUND(
        (
            0.6 *
            CASE
                WHEN service_rating = 1 THEN 1.0
                WHEN service_rating = 2 THEN 0.9
                WHEN service_rating = 3 THEN 0.4
                WHEN service_rating = 4 THEN 0.1
                ELSE 0.0
            END
        )
        +
        (
            0.3 *
            CASE
                WHEN delivery_delay = 'Yes' THEN 1.0
                ELSE 0.0
            END
        )
        +
        (
            0.1 *
            CASE
                WHEN product_category = 'Grocery' THEN 1.0
                WHEN product_category = 'Beverages' THEN 0.0
                ELSE 0.5
            END
        ),
        2
    ) AS refund_risk_score
FROM orders
ORDER BY refund_risk_score DESC;
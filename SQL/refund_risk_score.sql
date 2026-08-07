-- 5. Refund Risk Scoring (Rule-Based Baseline)
--
-- An explainable, hand-weighted refund-risk score per order, in [0, 1].
-- This is deliberately a *baseline*: the weights are simple business rules,
-- not learned. src/model.py trains a proper classifier on the same inputs and
-- reports how much it improves on this score (ROC-AUC comparison).
--
-- Signals and weights:
--   Service Rating   60%   (ratings 1-5 -> 1.0 .. 0.0)
--   Delivery Delay   30%   (derived is_delayed boolean)
--   Product Category 10%   (Fruits & Vegetables / Grocery higher risk)
--
-- Refunds also depend on delay and order value; the trained model captures
-- those interaction effects that this simple linear score cannot.

SELECT
    "Order ID",
    "Platform",
    "Product Category",
    "Service Rating",
    is_delayed,
    "Order Value (INR)",
    "Refund Requested",

    ROUND(
        (
            0.6 * CASE
                WHEN "Service Rating" = 1 THEN 1.0
                WHEN "Service Rating" = 2 THEN 0.8
                WHEN "Service Rating" = 3 THEN 0.4
                WHEN "Service Rating" = 4 THEN 0.15
                ELSE 0.05
            END
        )
        +
        (
            0.3 * CASE WHEN is_delayed THEN 1.0 ELSE 0.0 END
        )
        +
        (
            0.1 * CASE
                WHEN "Product Category" IN ('Fruits & Vegetables', 'Grocery') THEN 1.0
                WHEN "Product Category" = 'Beverages' THEN 0.4
                ELSE 0.7
            END
        ),
        3
    ) AS refund_risk_score

FROM orders
ORDER BY refund_risk_score DESC, "Order ID";

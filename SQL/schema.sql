-- ============================================================================
-- Schema for the e-commerce order fulfillment analysis
-- Target: PostgreSQL 15+
-- ============================================================================
--
-- The `orders` table mirrors the clean table produced by src/etl.py
-- (data/processed/orders_clean.csv). Identifiers contain spaces / ampersands,
-- so they are double-quoted throughout.
--
-- Load the data (run from repo root):
--     CREATE TABLE orders (...);        -- from this file
--     \copy orders FROM 'data/processed/orders_clean.csv' WITH (FORMAT csv, HEADER true)
--
-- The derived columns (is_delayed, order_weekday, order_hour) are produced by
-- the ETL pipeline, not stored raw, so the delay flag never drifts from the
-- promised/actual minutes.

DROP TABLE IF EXISTS orders;

CREATE TABLE orders (
    "Order ID"                    TEXT        PRIMARY KEY,
    "Customer ID"                 TEXT        NOT NULL,
    "Platform"                    TEXT        NOT NULL,
    "Order Date & Time"           TIMESTAMP   NOT NULL,
    "Promised Delivery (Minutes)" INTEGER     NOT NULL CHECK ("Promised Delivery (Minutes)" BETWEEN 8 AND 90),
    "Actual Delivery (Minutes)"   INTEGER     NOT NULL CHECK ("Actual Delivery (Minutes)" BETWEEN 3 AND 300),
    "Product Category"            TEXT        NOT NULL,
    "Order Value (INR)"           INTEGER     NOT NULL CHECK ("Order Value (INR)" BETWEEN 50 AND 2000),
    "Items Count"                 INTEGER     NOT NULL CHECK ("Items Count" BETWEEN 1 AND 20),
    "Customer Feedback"           TEXT        NOT NULL,
    "Service Rating"              INTEGER     NOT NULL CHECK ("Service Rating" BETWEEN 1 AND 5),
    "Delivery Delay"              TEXT        NOT NULL CHECK ("Delivery Delay" IN ('Yes', 'No')),
    "Refund Requested"            TEXT        NOT NULL CHECK ("Refund Requested" IN ('Yes', 'No')),
    is_delayed                    BOOLEAN     NOT NULL,
    order_weekday                 INTEGER     NOT NULL CHECK (order_weekday BETWEEN 0 AND 6),
    order_hour                    INTEGER     NOT NULL CHECK (order_hour BETWEEN 0 AND 23)
);

-- Convenience indexes for the analysis queries.
CREATE INDEX idx_orders_platform   ON orders ("Platform");
CREATE INDEX idx_orders_category   ON orders ("Product Category");
CREATE INDEX idx_orders_delayed    ON orders (is_delayed);
CREATE INDEX idx_orders_rating     ON orders ("Service Rating");
CREATE INDEX idx_orders_refund     ON orders ("Refund Requested");

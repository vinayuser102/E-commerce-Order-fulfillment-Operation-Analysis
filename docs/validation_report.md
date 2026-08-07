# Data Validation Report

- **File:** `V:\E-commerce Order Fulfillment Operations Analysis\data\raw\ecommerce_orders.csv`
- **Rows:** 100,000
- **Generated check timestamp:** n/a (reproducible, seeded)
- **Overall result:** **PASS**

| Check | Result | Detail |
|---|---|---|
| row count == 100000 | PASS | rows=100000 |
| no missing values | PASS | nulls across all columns |
| all dates parse | PASS | NaT count |
| date range sane (2024-11-01 to 2025-01-31) | PASS | min=2024-11-01 00:00:51 max=2025-01-31 23:57:02 |
| delay rate in [8%, 20%] | PASS | delay_rate=13.245% |
| refund rate in [5%, 25%] | PASS | refund_rate=18.751% |
| refund rate at rating 1 strictly in (0.5%, 99%) | PASS | rating=1 refund_rate=79.118% |
| refund rate at rating 2 strictly in (0.5%, 99%) | PASS | rating=2 refund_rate=58.384% |
| refund rate at rating 3 strictly in (0.5%, 99%) | PASS | rating=3 refund_rate=18.042% |
| refund rate at rating 4 strictly in (0.5%, 99%) | PASS | rating=4 refund_rate=6.708% |
| refund rate at rating 5 strictly in (0.5%, 99%) | PASS | rating=5 refund_rate=2.249% |
| |corr(actual_minutes, delay)| < 0.95 | PASS | corr=0.7238 |
| same delivery minute can be delayed and on-time (>= 3 overlaps) | PASS | overlapping minute values=26 |
| some rating-5 orders still have negative feedback | PASS | n=1313 |
| some rating-1 orders still have positive feedback | PASS | n=549 |
| delivery minutes within [3, 300] | PASS | range check |
| order value within [50, 2000] | PASS | range check |
| ratings within {1..5} | PASS | ratings=[1, 2, 3, 4, 5] |
| items count within [1, 20] | PASS | range check |
| all 3 platforms present | PASS | platforms=['Blinkit', 'JioMart', 'Swiggy Instamart'] |
| all 6 categories present | PASS | categories=['Beverages', 'Dairy', 'Fruits & Vegetables', 'Grocery', 'Personal Care', 'Snacks'] |

## Rates
- Delivery delay rate: **13.245%**
- Refund rate: **18.751%**
- Refund rate by rating: rating 1 = 79.1%, rating 2 = 58.4%, rating 3 = 18.0%, rating 4 = 6.7%, rating 5 = 2.2%

## Why these checks exist

The original dataset was fully deterministic: delay was a fixed threshold of delivery
time, and refund was a 1:1 function of service rating (100% at ratings 1-2, 0% at 3-5).
This validator fails if any of those relationships reappear, so the analysis and the
risk model must genuinely *learn* from the data instead of re-discovering pre-baked rules.
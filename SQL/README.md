# SQL Analysis Layer

SQL-first operational analysis on top of the `orders` table. Run `schema.sql`
first to create the table, then load the clean CSV (instructions in
`schema.sql`), then execute any query below.

| File | Answers |
|------|---------|
| `schema.sql` | DDL for the `orders` table (mirrors `data/processed/orders_clean.csv`) + load instructions |
| `data_overview.sql` | Sanity checks: total orders, distinct customers/platforms/categories, average delivery time and order value |
| `delivery_delay_analysis.sql` | Overall delay rate and delay distribution per platform |
| `category_delay_analysis.sql` | Delay percentage per category, ranked highest-to-lowest, plus share of all delays |
| `refund_analysis.sql` | Refund rate by service rating, and refund rate for delayed vs on-time orders |
| `refund_risk_score.sql` | Rule-based, explainable refund-risk score per order (baseline for the trained model in `src/model.py`) |

Notes:

- Identifiers with spaces/ampersands are double-quoted throughout.
- `is_delayed` is the derived boolean (actual > promised) produced by `src/etl.py`;
  the SQL uses it instead of the stored `Delivery Delay` text label.
- Query logic is cross-checked against the equivalent pandas computations in
  `notebooks/` — the SQL is written for PostgreSQL 15+.

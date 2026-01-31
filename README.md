This project analyzes e-commerce order fulfillment operations to identify operational inefficiencies, customer dissatisfaction drivers, and refund risk patterns.
The analysis is SQL-first and focuses on how delivery delays, service ratings, platforms, and product categories impact refunds.
The goal is operational decision support using clean, explainable SQL logic.

Business Problem

E-commerce platforms face frequent issues such as:

Late deliveries

Poor service experience

High refund rates

Inconsistent performance across platforms and categories

Refunds are costly consequences, not random events.

This project answers:

Where are delays happening?

Which platforms and categories are operationally weak?

How strongly do service ratings and delays relate to refunds?

Can we score refund risk at the order level using explainable rules?

Dataset

Source: Public e-commerce order fulfillment dataset

Records: ~100,000 orders

Each row represents a completed order

Key Fields Used

order_id

platform (Blinkit, JioMart, Swiggy Instamart)

product_category

service_rating (1–5)

delivery_delay (Yes / No)

refund_requested (Yes / No)

order_value_inr

Analysis Approach (SQL-First)

All analysis is performed using PostgreSQL.

The project is intentionally structured as incremental operational questions, not random queries.

Key Analyses Performed
1. Data Overview

Total number of orders

Distinct platforms and categories

Sanity checks for missing or inconsistent values

Purpose:
Ensure data validity before drawing conclusions.

2. Delivery Delay Analysis

Percentage of delayed vs on-time deliveries

Delay distribution across platforms

Key Insight:

A non-trivial share of orders are delayed

Delay rates vary significantly by platform

3. Product Category Delay Analysis

Delay percentage by product category

Categories ranked from highest to lowest delay rate

Key Insight:

Grocery orders show the highest delay risk

Beverages and certain categories are relatively stable

Operational Interpretation:
Perishable and fast-moving categories increase fulfillment complexity.

4. Refund Behavior Analysis

Refund rate comparison:

Delayed vs on-time orders

Across service ratings

Key Insight:

Refunds are more frequent when service ratings are low

Delivery delay alone does not fully explain refunds

Service experience acts as a strong amplifier

5. Refund Risk Scoring (Core Outcome)

A rule-based refund risk score is created at the order level.

Signals Used
Signal	Weight
Service Rating	60%
Delivery Delay	30%
Product Category	10%

Weights are:

Interpretable

Business-driven

Explicitly stated (not learned blindly)

Logic

Low service rating → high refund risk

Delivery delay → increased risk

Grocery category → higher operational fragility

Each order receives a refund risk score between 0 and 1.

Purpose:

Early identification of high-risk orders

Operational triage

Support team prioritization

Policy and process improvement

E-commerce-Order-Fulfillment-Operations-Analysis/
│
├── Data/
│   └── ecommerce_orders.csv
│
├── SQL/
│   ├── data_overview.sql
│   ├── delivery_delay_analysis.sql
│   ├── category_delay_analysis.sql
│   ├── refund_analysis.sql
│   └── refund_risk_score.sql
│
├── README.md

Tools Used

PostgreSQL

SQL (window functions, CASE logic, aggregation)

GitHub for version control

Future Improvements

Convert risk score into a logistic regression model

Add time-based analysis (peak hours, weekdays)

Incorporate order value into risk weighting

Platform-specific operational benchmarks

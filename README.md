



# E-Commerce Order Fulfillment Operations Analysis

## Project Overview

This project analyzes **e-commerce order fulfillment operations** to identify operational inefficiencies, customer dissatisfaction drivers, and refund risk patterns.

The analysis is **SQL-first** and focuses on how delivery delays, service ratings, platforms, and product categories impact refunds.
The goal is **operational decision support**, not dashboards or black-box models.

---

## Business Problem

E-commerce platforms frequently face operational issues such as:

* Late deliveries
* Poor service experience
* High refund rates
* Inconsistent performance across platforms and categories

Refunds are **costly consequences**, not random events.

This project answers:

* Where are delivery delays happening?
* Which platforms and product categories are operationally weak?
* How do service ratings and delivery delays relate to refunds?
* Can we score refund risk at the order level using **explainable rules**?

---

## Dataset

**Source:** Public e-commerce order fulfillment dataset(kaggle dataset ,Last updated fed, 2025)
**Size:** ~100,000 orders

Each row represents a completed order.

### Key Fields

* `order_id`
* `platform` (Blinkit, JioMart, Swiggy Instamart)
* `product_category`
* `service_rating` (1–5)
* `delivery_delay` (Yes / No)
* `refund_requested` (Yes / No)
* `order_value_inr`

---

## Analysis Approach

All analysis is performed using **PostgreSQL**.

The project is structured as a sequence of **operational questions**, not isolated queries.

---

## Key Analyses

### 1. Data Overview

* Total number of orders
* Distinct platforms and product categories
* Basic sanity checks

**Purpose:**
Validate data before drawing conclusions.

---

### 2. Delivery Delay Analysis

* Percentage of delayed vs on-time orders
* Delay distribution across platforms

**Insight:**
Delivery delay rates vary significantly by platform, indicating uneven operational performance.

---

### 3. Product Category Delay Analysis

* Delay percentage by product category
* Categories ranked from highest to lowest delay risk

**Insight:**
Grocery orders show the highest delay rates, suggesting higher fulfillment complexity.

---

### 4. Refund Behavior Analysis

* Refund rate for delayed vs on-time orders
* Refund rate by service rating

**Insight:**
Low service ratings strongly correlate with higher refund rates.
Delivery delay alone does not fully explain refunds.

---

### 5. Refund Risk Scoring (Core Outcome)

A **rule-based refund risk score** is calculated for each order.

#### Signals Used

| Signal           | Weight |
| ---------------- | ------ |
| Service Rating   | 60%    |
| Delivery Delay   | 30%    |
| Product Category | 10%    |

Weights are explicitly defined and business-driven.

#### Interpretation

* Low service rating → highest refund risk
* Delivery delay → secondary risk driver
* Grocery category → higher operational fragility

Each order receives a **refund risk score between 0 and 1**, enabling early identification of high-risk orders.

---

## Project Structure

```
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
```

---


## Limitations

* Dataset is historical and simulated
* External factors (weather, staffing, supply chain) are not available

---

## Future Improvements

* Convert risk score into a predictive model
* Add time-based analysis (hour, weekday, peak periods)
* Incorporate order value into risk weighting
* Platform-specific operational benchmarks

---

## Tools Used

* PostgreSQL
* SQL (CTEs, window functions, CASE logic)
* GitHub for version control

---




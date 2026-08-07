"""Generate a realistic, seeded e-commerce order fulfillment dataset.

This replaces the original synthetic CSV, whose columns were deterministic
functions of each other (delay == a fixed threshold of delivery time;
refund == service rating 1 or 2). Those pre-baked relationships made the
analysis circular. The new generator produces *probabilistic* relationships:

- Delivery Delay is DERIVED from promised-vs-actual delivery minutes
  (a per-row promised SLA plus random overrun) -- never a fixed threshold.
- Service Rating depends on delay / category / platform / order value but is
  never forced, so rating and refund correlate but are never 1:1.
- Refund depends on rating / delay / category / order value but is never
  deterministic.

All randomness is seeded, so re-running produces the identical dataset.

Usage:
    python scripts/generate_data.py                    # default: 100k rows
    python scripts/generate_data.py --rows 20000 --seed 7
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "ecommerce_orders.csv"

# -- Timestamp distribution --------------------------------------------------
# 92 days: 2024-11-01 .. 2025-01-31 (matches the README's "last updated Feb 2025").
START_DATE = datetime(2024, 11, 1)
N_DAYS = 92

# Hour-of-day demand profile (0..23). Evening peak, overnight trough.
HOUR_WEIGHTS = np.array(
    [
        0.35, 0.25, 0.20, 0.20, 0.25, 0.35,   # 00-05
        0.80, 1.30, 1.60, 1.70, 1.60, 1.40,   # 06-11
        1.30, 1.20, 1.20, 1.40, 1.60, 2.20,   # 12-17
        3.00, 3.40, 3.20, 2.40, 1.40, 0.70,   # 18-23
    ],
    dtype=float,
)
WEEKEND_WEIGHT = 1.35  # Sat/Sun generate more orders

# -- Platform ----------------------------------------------------------------
PLATFORMS = ["Blinkit", "JioMart", "Swiggy Instamart"]
PLATFORM_DELAY_MULT = {"JioMart": 1.05, "Blinkit": 0.95, "Swiggy Instamart": 1.00}
PLATFORM_RATING_SHIFT = {"JioMart": 0.05, "Blinkit": 0.00, "Swiggy Instamart": -0.05}

# -- Product categories ------------------------------------------------------
#   sla          : base promised delivery minutes (per-category SLA)
#   p_delay      : base probability the order runs over its promised SLA
#   value_mu     : log-scale mean of order value (INR)
#   items_mean   : mean item count (Poisson)
#   rating_shift : additive shift to the latent satisfaction score
CATEGORIES = {
    "Grocery":             {"sla": 22, "p_delay": 0.10, "value_mu": 6.50, "items_mean": 4.0, "rating_shift": 0.10},
    "Dairy":               {"sla": 24, "p_delay": 0.11, "value_mu": 5.90, "items_mean": 3.0, "rating_shift": 0.05},
    "Beverages":           {"sla": 26, "p_delay": 0.12, "value_mu": 6.10, "items_mean": 4.0, "rating_shift": 0.00},
    "Snacks":              {"sla": 28, "p_delay": 0.13, "value_mu": 5.80, "items_mean": 4.0, "rating_shift": -0.05},
    "Personal Care":       {"sla": 34, "p_delay": 0.16, "value_mu": 6.35, "items_mean": 2.0, "rating_shift": -0.10},
    "Fruits & Vegetables": {"sla": 36, "p_delay": 0.18, "value_mu": 5.70, "items_mean": 3.0, "rating_shift": -0.15},
}
CATEGORY_NAMES = list(CATEGORIES)

# Refund base probability per rating (multiplied by delay/category/value).
REFUND_P_BY_RATING = {1: 0.60, 2: 0.45, 3: 0.15, 4: 0.06, 5: 0.02}

# Feedback phrase pools -- probabilistically matched to rating, never 1:1.
POSITIVE_FEEDBACK = [
    "Fast delivery, great service!",
    "Excellent experience!",
    "Easy to order, loved it!",
    "Very satisfied with the service.",
    "Quick and reliable!",
    "Good quality products.",
    "Will definitely order again!",
    "Great value for money.",
    "Smooth ordering and quick delivery.",
]
NEUTRAL_FEEDBACK = [
    "Packaging could be better.",
    "Decent experience overall.",
    "Order was okay, nothing special.",
    "Product quality was average.",
    "Delivery was a little slow but acceptable.",
]
NEGATIVE_FEEDBACK = [
    "Items missing from order.",
    "Delivery person was rude.",
    "Not fresh, disappointed.",
    "Very late delivery, not happy.",
    "Wrong item delivered.",
    "Horrible experience, never ordering again.",
    "Delayed and poor communication.",
    "Product arrived damaged.",
    "Cold items arrived melted.",
]

# Pool selection probabilities by rating band: [POS, NEU, NEG]
FEEDBACK_POOL_P = {
    "high": np.array([0.75, 0.18, 0.07]),
    "mid": np.array([0.25, 0.50, 0.25]),
    "low": np.array([0.10, 0.20, 0.70]),
}


def sample_timestamps(rng: np.random.Generator, n: int) -> np.ndarray:
    """Sample n order timestamps with evening + weekend demand weighting."""
    days = np.arange(N_DAYS)
    weekdays = (START_DATE.weekday() + days) % 7
    day_weights = np.where(np.isin(weekdays, [5, 6]), WEEKEND_WEIGHT, 1.0)

    grid = np.outer(day_weights, HOUR_WEIGHTS).ravel()  # day x hour
    grid /= grid.sum()

    idx = rng.choice(grid.size, size=n, p=grid)
    day_idx, hour_idx = np.divmod(idx, 24)
    minute = rng.integers(0, 60, size=n)
    second = rng.integers(0, 60, size=n)

    base = np.datetime64(START_DATE.date(), "D").astype("datetime64[s]")
    offset = (
        day_idx.astype("timedelta64[D]")
        + hour_idx.astype("timedelta64[h]")
        + minute.astype("timedelta64[m]")
        + second.astype("timedelta64[s]")
    )
    return base + offset


def generate(rows: int, seed: int, output: Path) -> None:
    rng = np.random.default_rng(seed)

    # --- Platform & category -------------------------------------------------
    platforms = rng.choice(PLATFORMS, size=rows)
    categories = rng.choice(CATEGORY_NAMES, size=rows)

    # --- Delivery times (promised + actual) ----------------------------------
    cat_sla = np.array([CATEGORIES[c]["sla"] for c in categories], dtype=float)
    cat_p_delay = np.array([CATEGORIES[c]["p_delay"] for c in categories], dtype=float)
    p_mult = np.array([PLATFORM_DELAY_MULT[p] for p in platforms], dtype=float)

    promised = np.clip(cat_sla + rng.normal(0.0, 3.0, size=rows), 8.0, 90.0)
    promised = np.round(promised).astype(int)

    p_delay = np.clip(cat_p_delay * p_mult, 0.02, 0.50)
    delayed = rng.random(size=rows) < p_delay

    overrun = np.where(
        delayed,
        rng.gamma(shape=2.0, scale=6.0, size=rows) + 3.0,      # late: mean ~+15 min
        -(2.0 + np.abs(rng.normal(0.0, 5.0, size=rows))),       # on-time: ~-4 min early
    )
    actual = np.clip(promised + overrun, 3.0, 300.0)
    actual = np.round(actual).astype(int)

    # Delay is *derived* from the two noisy delivery columns, so no single
    # delivery-time value determines it.
    delay_flag = (actual > promised).astype(float)
    delivery_delay = np.where(delay_flag == 1.0, "Yes", "No")

    # --- Order value & items ---------------------------------------------------
    cat_value_mu = np.array([CATEGORIES[c]["value_mu"] for c in categories], dtype=float)
    order_value = np.exp(rng.normal(cat_value_mu, 0.5, size=rows))
    order_value = np.clip(np.round(order_value), 50, 2000).astype(int)

    cat_items = np.array([CATEGORIES[c]["items_mean"] for c in categories], dtype=float)
    items = rng.poisson(cat_items) + 1
    items = np.clip(items, 1, 20).astype(int)

    # --- Service rating (latent satisfaction, never forced) --------------------
    cat_shift = np.array([CATEGORIES[c]["rating_shift"] for c in categories], dtype=float)
    plat_shift = np.array([PLATFORM_RATING_SHIFT[p] for p in platforms], dtype=float)
    value_penalty = np.where(order_value >= 1000, -0.10, 0.0)

    latent = (
        3.7
        + cat_shift
        + plat_shift
        - 0.9 * delay_flag
        + value_penalty
        + rng.normal(0.0, 1.05, size=rows)
    )
    rating = np.clip(np.round(latent), 1, 5).astype(int)

    # --- Refund (probabilistic, never 1:1 with rating) --------------------------
    p_refund_base = np.array([REFUND_P_BY_RATING[r] for r in rating], dtype=float)
    cat_refund_mult = np.array(
        [
            1.2 if c == "Fruits & Vegetables" else 1.1 if c == "Grocery" else 1.0
            for c in categories
        ],
        dtype=float,
    )
    refund_mult = np.where(delay_flag == 1.0, 1.7, 1.0) * cat_refund_mult
    refund_mult = np.where(order_value >= 800, refund_mult * 1.15, refund_mult)
    p_refund = np.clip(p_refund_base * refund_mult, 0.0, 0.95)

    refund_flag = rng.random(size=rows) < p_refund
    refund_requested = np.where(refund_flag, "Yes", "No")

    # --- Customer feedback (probabilistically tied to rating) --------------------
    # Pick each row's phrase pool by probability, then a phrase from that pool.
    pool_probs = np.where(
        (rating >= 4)[:, None],
        FEEDBACK_POOL_P["high"][None, :],
        np.where(
            (rating == 3)[:, None],
            FEEDBACK_POOL_P["mid"][None, :],
            np.broadcast_to(FEEDBACK_POOL_P["low"], (rows, 3)),
        ),
    )
    cum = np.cumsum(pool_probs, axis=1)
    pool_idx = (rng.random(size=rows)[:, None] < cum).argmax(axis=1)

    feedback = np.empty(rows, dtype=object)
    for i, pool in enumerate((POSITIVE_FEEDBACK, NEUTRAL_FEEDBACK, NEGATIVE_FEEDBACK)):
        mask = pool_idx == i
        feedback[mask] = rng.choice(pool, size=int(mask.sum()))

    # --- Customer IDs (9000 customers, power-law repeat buying) -------------------
    k = np.arange(1, 9001)
    cust_weights = k ** -1.5
    cust_weights /= cust_weights.sum()
    customer_rank = rng.choice(9000, size=rows, p=cust_weights) + 1
    customer_id = np.array([f"CUST{c:04d}" for c in customer_rank])

    # --- Assemble, sort chronologically, assign sequential Order IDs ---------------
    timestamps = sample_timestamps(rng, rows)

    frame = pd.DataFrame(
        {
            "Customer ID": customer_id,
            "Platform": platforms,
            "Order Date & Time": timestamps,
            "Promised Delivery (Minutes)": promised,
            "Actual Delivery (Minutes)": actual,
            "Product Category": categories,
            "Order Value (INR)": order_value,
            "Items Count": items,
            "Customer Feedback": feedback,
            "Service Rating": rating,
            "Delivery Delay": delivery_delay,
            "Refund Requested": refund_requested,
        }
    )
    frame = frame.sort_values("Order Date & Time", kind="stable").reset_index(drop=True)
    frame.insert(0, "Order ID", [f"ORD{i + 1:06d}" for i in range(rows)])

    output.parent.mkdir(parents=True, exist_ok=True)
    frame["Order Date & Time"] = frame["Order Date & Time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    frame.to_csv(output, index=False, quoting=1)  # QUOTE_ALL keeps commas in feedback safe
    print(f"Wrote {rows:,} rows to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the e-commerce order fulfillment dataset.")
    parser.add_argument("--rows", type=int, default=100000, help="number of orders (default 100000)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output CSV path")
    args = parser.parse_args()

    generate(args.rows, args.seed, args.output)


if __name__ == "__main__":
    main()

"""Feature engineering: turn the clean orders table into a model-ready matrix.

Pipeline stage 2 of 3 (etl.py -> features.py -> model.py).

Engineered features (used by the refund-prediction model in model.py):

- minutes_late      : how late the order was (0 if on time / early)
- weekday, hour     : calendar context (categorical)
- value_bucket      : low / mid / high / premium order value
- delay_and_value   : interaction term (delayed AND high value -> risky)
- platform/category : one-hot encoded

The target is `refund_requested` (1 = Yes).

Usage (CLI):
    python -m src.features

Importable:
    from src.features import build_features
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_INPUT = PROJECT_ROOT / "data" / "processed" / "orders_clean.csv"
FEATURES_OUTPUT = PROJECT_ROOT / "data" / "processed" / "features.csv"


def build_features(clean: pd.DataFrame) -> pd.DataFrame:
    """Return a numeric feature matrix plus the target, order-preserving."""
    out = pd.DataFrame()
    out["order_id"] = clean["Order ID"]

    # -- Continuous ------------------------------------------------------------
    out["minutes_late"] = np.maximum(
        clean["Actual Delivery (Minutes)"] - clean["Promised Delivery (Minutes)"], 0
    )
    out["order_value"] = clean["Order Value (INR)"]
    out["items_count"] = clean["Items Count"]
    out["promised_minutes"] = clean["Promised Delivery (Minutes)"]

    # -- Calendar ---------------------------------------------------------------
    out["weekday"] = clean["order_weekday"]
    out["hour"] = clean["order_hour"]

    # -- Signals the rule-based baseline uses (rating + delay) -------------------
    # The SQL baseline weights Service Rating at 60% and Delivery Delay at 30%.
    # Exposing them to the trained models keeps the comparison apples-to-apples.
    out["service_rating"] = clean["Service Rating"]
    out["is_delayed"] = clean["is_delayed"].astype(int)

    # -- Binned / interaction ---------------------------------------------------
    out["value_bucket"] = pd.cut(
        clean["Order Value (INR)"],
        bins=[0, 300, 600, 1000, 5000],
        labels=["low", "mid", "high", "premium"],
    )
    out["delay_and_high_value"] = (
        clean["is_delayed"].astype(int) & (clean["Order Value (INR)"] >= 800)
    ).astype(int)

    # -- One-hot categoricals ----------------------------------------------------
    platform_dummies = pd.get_dummies(clean["Platform"], prefix="platform")
    category_dummies = pd.get_dummies(clean["Product Category"], prefix="category")
    out = pd.concat([out, platform_dummies.astype(int), category_dummies.astype(int)], axis=1)

    # -- Target ------------------------------------------------------------------
    out["refund_requested"] = clean["Refund Requested"].eq("Yes").astype(int)

    return out


def run_pipeline(clean_path: Path = CLEAN_INPUT, out_path: Path = FEATURES_OUTPUT) -> pd.DataFrame:
    clean = pd.read_csv(clean_path)
    features = build_features(clean)
    features.to_csv(out_path, index=False)
    print(f"FEATURES OK: {features.shape[0]:,} rows x {features.shape[1]} cols -> {out_path}")
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the feature-engineering stage.")
    parser.add_argument("--clean", type=Path, default=CLEAN_INPUT)
    parser.add_argument("--out", type=Path, default=FEATURES_OUTPUT)
    args = parser.parse_args()
    run_pipeline(args.clean, args.out)


if __name__ == "__main__":
    main()

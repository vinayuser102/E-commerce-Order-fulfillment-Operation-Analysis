"""ETL: load raw orders CSV, clean/validate, and write a model-ready clean table.

Pipeline stage 1 of 3 (etl.py -> features.py -> model.py).

What this stage guarantees:
- Order timestamps are parsed as real datetimes (the original dataset shipped a
  corrupted timestamp column; this module asserts none slips through).
- Delivery delay is *derived* from promised vs actual delivery minutes and then
  stored as an explicit boolean (is_delayed), keeping the raw rule in one place.
- No nulls, no duplicate Order IDs, value ranges validated.
- A data dictionary is emitted alongside the clean table.

Usage (CLI):
    python -m src.etl

Importable:
    from src.etl import run_pipeline
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "ecommerce_orders.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEAN_OUTPUT = PROCESSED_DIR / "orders_clean.csv"
DICT_OUTPUT = PROCESSED_DIR / "data_dictionary.csv"

PLATFORMS = {"Blinkit", "JioMart", "Swiggy Instamart"}
CATEGORIES = {"Grocery", "Dairy", "Beverages", "Snacks", "Personal Care", "Fruits & Vegetables"}


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Order Date & Time"])
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Type-coerce and derive the explicit delivery-delay flag."""
    out = df.copy()

    # -- Type coercion -------------------------------------------------------
    out["Order Date & Time"] = pd.to_datetime(out["Order Date & Time"], errors="raise")
    out["Promised Delivery (Minutes)"] = out["Promised Delivery (Minutes)"].astype(int)
    out["Actual Delivery (Minutes)"] = out["Actual Delivery (Minutes)"].astype(int)
    out["Order Value (INR)"] = out["Order Value (INR)"].astype(int)
    out["Items Count"] = out["Items Count"].astype(int)
    out["Service Rating"] = out["Service Rating"].astype(int)

    # -- Category / platform standardization -----------------------------------
    out["Product Category"] = out["Product Category"].str.strip()
    out["Platform"] = out["Platform"].str.strip()

    # -- Derive the explicit delay flag (single source of truth) ---------------
    # The generator's "Delivery Delay" is (actual > promised); recompute here so
    # the flag never drifts from the raw minutes.
    out["is_delayed"] = out["Actual Delivery (Minutes)"] > out["Promised Delivery (Minutes)"]

    # -- Calendar features (kept on the clean table for downstream EDA) --------
    out["order_weekday"] = out["Order Date & Time"].dt.dayofweek  # Mon=0 .. Sun=6
    out["order_hour"] = out["Order Date & Time"].dt.hour

    return out


def validate(out: pd.DataFrame) -> None:
    """Hard assertions on the clean table (fail fast if violated)."""
    assert out["Order ID"].is_unique, "duplicate Order IDs"
    assert out.isna().sum().sum() == 0, "nulls present in clean table"
    assert set(out["Platform"]) <= PLATFORMS, "unexpected platform value"
    assert set(out["Product Category"]) <= CATEGORIES, "unexpected category value"
    assert out["Service Rating"].between(1, 5).all(), "rating out of range"
    assert out["Order Value (INR)"].between(50, 2000).all(), "order value out of range"
    assert out["Items Count"].between(1, 20).all(), "items out of range"
    assert out["is_delayed"].isin([True, False]).all(), "is_delayed must be boolean"
    # Delay flag must be consistent with the raw minutes columns.
    assert (out["is_delayed"] == (out["Actual Delivery (Minutes)"] > out["Promised Delivery (Minutes)"])).all()
    # The derived flag must agree with the generator's stored label.
    assert (out["is_delayed"] == out["Delivery Delay"].eq("Yes")).all(), "derived flag disagrees with stored label"


def data_dictionary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "column": [
                "Order ID", "Customer ID", "Platform", "Order Date & Time",
                "Promised Delivery (Minutes)", "Actual Delivery (Minutes)",
                "Product Category", "Order Value (INR)", "Items Count",
                "Customer Feedback", "Service Rating", "Delivery Delay",
                "Refund Requested", "is_delayed", "order_weekday", "order_hour",
            ],
            "type": [
                "str", "str", "category", "datetime",
                "int", "int",
                "category", "int", "int",
                "str", "int 1-5", "Yes/No",
                "Yes/No", "bool", "int 0-6", "int 0-23",
            ],
            "description": [
                "Unique order identifier", "Anonymized customer id",
                "Delivery platform", "Order placement timestamp (real datetime)",
                "Promised delivery minutes (per-category SLA + jitter)",
                "Actual delivery minutes (promised + overrun/early)",
                "Product category (6)", "Order value in INR",
                "Number of items in order", "Free-text customer feedback",
                "Customer satisfaction rating", "Stored delay label (actual > promised)",
                "Refund requested flag", "Derived delay boolean",
                "Day of week, Monday=0", "Hour of day, 0-23",
            ],
        }
    )


def run_pipeline(raw: Path = RAW_DATA, out: Path = CLEAN_OUTPUT) -> pd.DataFrame:
    """Run the full ETL stage and return the clean table."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw(raw)
    clean_df = clean(df)
    validate(clean_df)

    clean_df.to_csv(out, index=False, quoting=1)
    data_dictionary().to_csv(DICT_OUTPUT, index=False, quoting=1)

    print(f"ETL OK: {len(clean_df):,} rows, {clean_df.shape[1]} cols -> {out}")
    print(f"Delay rate: {100 * clean_df['is_delayed'].mean():.2f}% | "
          f"Refund rate: {100 * clean_df['Refund Requested'].eq('Yes').mean():.2f}%")
    return clean_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ETL stage.")
    parser.add_argument("--raw", type=Path, default=RAW_DATA)
    parser.add_argument("--out", type=Path, default=CLEAN_OUTPUT)
    args = parser.parse_args()
    run_pipeline(args.raw, args.out)


if __name__ == "__main__":
    main()

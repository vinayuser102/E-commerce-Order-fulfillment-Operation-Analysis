"""Validate the generated order dataset and write a QA report.

This is the gate that guarantees the original dataset's determinism never
returns. The original data had three hard-coded relationships:

    delivery_delay == (delivery time > 40)          -> delay was a pure threshold
    refund_requested == (rating in {1, 2})          -> refund was 1:1 with rating
    feedback phrase -> rating mapping was 1:1

If any of those reappear, this script FAILS (non-zero exit) and the report
records exactly which check tripped.

Usage:
    python scripts/validate_data.py                  # validates the default CSV
    python scripts/validate_data.py --data path.csv  # validate a specific CSV
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_ROOT / "data" / "raw" / "ecommerce_orders.csv"
REPORT_PATH = PROJECT_ROOT / "docs" / "validation_report.md"

NEGATIVE_KEYWORDS = [
    "late", "rude", "missing", "wrong", "horrible", "damaged",
    "melted", "disappointed", "communication",
]
POSITIVE_KEYWORDS = [
    "fast", "excellent", "loved", "satisfied", "quick", "great",
    "again", "value", "smooth", "good quality",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the generated dataset.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="CSV to validate")
    args = parser.parse_args()

    df = pd.read_csv(args.data, parse_dates=["Order Date & Time"])
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append((name, bool(ok), detail))

    # -- Structure & completeness --------------------------------------------
    n = len(df)
    check("row count == 100000", n == 100000, f"rows={n}")
    check("no missing values", int(df.isna().sum().sum()) == 0, "nulls across all columns")

    # -- Timestamps ------------------------------------------------------------
    parsed = pd.to_datetime(df["Order Date & Time"], errors="coerce")
    check("all dates parse", int(parsed.isna().sum()) == 0, "NaT count")
    check(
        "date range sane (2024-11-01 to 2025-01-31)",
        bool(parsed.between("2024-11-01", "2025-02-01", inclusive="left").all()),
        f"min={parsed.min()} max={parsed.max()}",
    )

    # -- Rates within intended windows -----------------------------------------
    delay_rate = float(df["Delivery Delay"].eq("Yes").mean())
    check("delay rate in [8%, 20%]", 0.08 <= delay_rate <= 0.20, f"delay_rate={delay_rate:.3%}")

    refund_rate = float(df["Refund Requested"].eq("Yes").mean())
    check("refund rate in [5%, 25%]", 0.05 <= refund_rate <= 0.25, f"refund_rate={refund_rate:.3%}")

    # -- Determinism 1: refund must NOT be 1:1 with rating -----------------------
    refund_by_rating = (
        df.groupby("Service Rating")["Refund Requested"].apply(lambda s: float(s.eq("Yes").mean()))
    )
    for r, rate in refund_by_rating.items():
        check(
            f"refund rate at rating {int(r)} strictly in (0.5%, 99%)",
            0.005 < rate < 0.99,
            f"rating={int(r)} refund_rate={rate:.3%}",
        )

    # -- Determinism 2: delay must NOT be a threshold of a single time column -----
    minutes = df["Actual Delivery (Minutes)"].astype(int).to_numpy()
    delayed = df["Delivery Delay"].eq("Yes").to_numpy().astype(float)
    corr = float(np.corrcoef(minutes, delayed)[0, 1])
    check("|corr(actual_minutes, delay)| < 0.95", abs(corr) < 0.95, f"corr={corr:.4f}")

    # Same actual-minutes value must be able to be both delayed and on-time.
    delayed_mins = set(minutes[delayed == 1.0])
    ontime_mins = set(minutes[delayed == 0.0])
    overlap = len(delayed_mins & ontime_mins)
    check(
        "same delivery minute can be delayed and on-time (>= 3 overlaps)",
        overlap >= 3,
        f"overlapping minute values={overlap}",
    )

    # -- Determinism 3: feedback must NOT be 1:1 with rating ----------------------
    lower = df["Customer Feedback"].astype(str).str.lower()
    neg_mask = lower.str.contains("|".join(NEGATIVE_KEYWORDS), regex=True)
    pos_mask = lower.str.contains("|".join(POSITIVE_KEYWORDS), regex=True)

    r5_with_neg = int(df.loc[df["Service Rating"].eq(5) & neg_mask].shape[0])
    check("some rating-5 orders still have negative feedback", r5_with_neg > 0, f"n={r5_with_neg}")

    r1_with_pos = int(df.loc[df["Service Rating"].eq(1) & pos_mask].shape[0])
    check("some rating-1 orders still have positive feedback", r1_with_pos > 0, f"n={r1_with_pos}")

    # -- Value ranges -------------------------------------------------------------
    check(
        "delivery minutes within [3, 300]",
        bool(df["Actual Delivery (Minutes)"].between(3, 300).all())
        and bool(df["Promised Delivery (Minutes)"].between(8, 90).all()),
        "range check",
    )
    check(
        "order value within [50, 2000]",
        bool(df["Order Value (INR)"].between(50, 2000).all()),
        "range check",
    )
    check(
        "ratings within {1..5}",
        set(df["Service Rating"].unique()) <= {1, 2, 3, 4, 5},
        f"ratings={sorted(df['Service Rating'].unique())}",
    )
    check(
        "items count within [1, 20]",
        bool(df["Items Count"].between(1, 20).all()),
        "range check",
    )

    # -- Coverage -------------------------------------------------------------------
    check(
        "all 3 platforms present",
        set(df["Platform"].unique()) >= {"Blinkit", "JioMart", "Swiggy Instamart"},
        f"platforms={sorted(df['Platform'].unique())}",
    )
    check(
        "all 6 categories present",
        set(df["Product Category"].unique())
        >= {"Grocery", "Dairy", "Beverages", "Snacks", "Personal Care", "Fruits & Vegetables"},
        f"categories={sorted(df['Product Category'].unique())}",
    )

    # -- Write report ---------------------------------------------------------------
    all_ok = all(ok for _, ok, _ in checks)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Data Validation Report",
        "",
        f"- **File:** `{args.data}`",
        f"- **Rows:** {n:,}",
        f"- **Generated check timestamp:** n/a (reproducible, seeded)",
        f"- **Overall result:** {'**PASS**' if all_ok else '**FAIL**'}",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    for name, ok, detail in checks:
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")

    lines += [
        "",
        "## Rates",
        f"- Delivery delay rate: **{delay_rate:.3%}**",
        f"- Refund rate: **{refund_rate:.3%}**",
        f"- Refund rate by rating: "
        + ", ".join(f"rating {int(r)} = {rate:.1%}" for r, rate in refund_by_rating.items()),
        "",
        "## Why these checks exist",
        "",
        "The original dataset was fully deterministic: delay was a fixed threshold of delivery",
        "time, and refund was a 1:1 function of service rating (100% at ratings 1-2, 0% at 3-5).",
        "This validator fails if any of those relationships reappear, so the analysis and the",
        "risk model must genuinely *learn* from the data instead of re-discovering pre-baked rules.",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote validation report to {REPORT_PATH}")
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print(f"\nOVERALL: {'PASS' if all_ok else 'FAIL'} ({sum(ok for _, ok, _ in checks)}/{len(checks)} checks)")

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

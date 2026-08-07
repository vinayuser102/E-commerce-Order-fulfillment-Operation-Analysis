"""Train and evaluate refund-prediction models, and compare them to the
rule-based SQL baseline.

Pipeline stage 3 of 3 (etl.py -> features.py -> model.py).

What this stage delivers:
- A proper train/test split (stratified 80/20) and cross-validation -- the
  thing the original hand-weighted score never had.
- Two classifiers: logistic regression (interpretable baseline) and a random
  forest (captures interactions like delay x value).
- Class imbalance handled with class weighting.
- Real metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix,
  and feature importance.
- A direct comparison against the *rule-based* refund-risk score implemented
  in SQL/refund_risk_score.sql, so we can quantify how much the trained model
  improves on the explainable heuristic.

Artifacts written to models/:
    metrics.json        all metrics for logistic regression, random forest, baseline
    baseline_predictions.csv   order-level scores/predictions for the SQL baseline
    refund_model.joblib        the best model (used by the dashboard)

Usage (CLI):
    python -m src.model

Importable:
    from src.model import run_pipeline, rule_based_score
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_INPUT = PROJECT_ROOT / "data" / "processed" / "features.csv"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_OUTPUT = MODELS_DIR / "metrics.json"
BASELINE_PREDS = MODELS_DIR / "baseline_predictions.csv"
MODEL_OUTPUT = MODELS_DIR / "refund_model.joblib"

RANDOM_STATE = 42
TARGET = "refund_requested"

# Columns used as model inputs (everything except id / target / value_bucket label).
def _feature_cols(features: pd.DataFrame) -> list[str]:
    """Derive the model input columns from the feature matrix at runtime."""
    categorical_hot = [c for c in features.columns if c.startswith(("platform_", "category_"))]
    return [
        "minutes_late",
        "order_value",
        "items_count",
        "promised_minutes",
        "weekday",
        "hour",
        "delay_and_high_value",
        "service_rating",  # the baseline's dominant signal (60% weight)
        "is_delayed",      # the baseline's secondary signal (30% weight)
    ] + categorical_hot

# --- Rule-based baseline (mirror of SQL/refund_risk_score.sql) ---------------
# The SQL weights: rating 60%, delay 30%, category 10%. We approximate the
# score from the available features so the baseline is directly comparable.
RATING_COL = "Service Rating"  # not a model feature; used only for the baseline
BASELINE_WEIGHTS = {1: 1.0, 2: 0.8, 3: 0.4, 4: 0.15, 5: 0.05}
CATEGORY_WEIGHTS = {
    "Fruits & Vegetables": 1.0,
    "Grocery": 1.0,
    "Beverages": 0.4,
    "Dairy": 0.7,
    "Personal Care": 0.7,
    "Snacks": 0.7,
}


def rule_based_score(clean: pd.DataFrame) -> pd.Series:
    """Reimplement the SQL rule-based risk score in [0, 1]."""
    rating_w = clean["Service Rating"].map(BASELINE_WEIGHTS).fillna(0.0)
    delay_w = clean["is_delayed"].astype(float)
    cat_w = clean["Product Category"].map(CATEGORY_WEIGHTS).fillna(0.0)
    return (0.6 * rating_w + 0.3 * delay_w + 0.1 * cat_w).round(3)


def _evaluate(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_score), 4),
    }


def train_and_evaluate(features: pd.DataFrame, clean: pd.DataFrame) -> dict:
    """Train both classifiers on the split, cross-validate, and return metrics."""
    feature_cols = _feature_cols(features)
    X = features[feature_cols]
    y = features[TARGET].astype(int).to_numpy()

    # Index-based split so the rule-based baseline is scored on the SAME rows.
    idx = np.arange(len(X))
    idx_tr, idx_te = train_test_split(
        idx, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_tr, X_te = X.iloc[idx_tr], X.iloc[idx_te]
    y_tr, y_te = y[idx_tr], y[idx_te]

    # --- Models (class_weight handles the ~19% positive imbalance) -----------
    # n_jobs kept low (and depth capped) so training fits comfortably in memory
    # even across the 5-fold CV runs.
    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=14, class_weight="balanced_subsample",
            random_state=RANDOM_STATE, n_jobs=2,
        ),
    }

    results: dict[str, dict] = {}
    cv_scores: dict[str, list[float]] = {}

    for name, model in models.items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]

        metrics = _evaluate(y_te, y_proba, y_pred)
        metrics["confusion_matrix"] = confusion_matrix(y_te, y_pred).tolist()
        metrics["n_test"] = int(len(y_te))

        cv = cross_val_score(model, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
        cv_scores[name] = [round(v, 4) for v in cv.tolist()]
        metrics["cv_auc_mean"] = round(float(np.mean(cv)), 4)
        metrics["cv_auc_std"] = round(float(np.std(cv)), 4)

        if name == "random_forest":
            metrics["feature_importance"] = dict(
                sorted(
                    zip(feature_cols, model.feature_importances_),
                    key=lambda kv: -kv[1],
                )
            )

        results[name] = metrics

    # --- Rule-based baseline (score > 0.5 predicts refund) ----------------------
    # Evaluated on the SAME held-out test rows as the trained models, so the
    # comparison is fair. baseline_proba[i] aligns 1:1 with features rows.
    baseline_proba = rule_based_score(clean).to_numpy()
    baseline_proba_te = baseline_proba[idx_te]
    baseline_pred_te = (baseline_proba_te > 0.5).astype(int)
    results["rule_based_baseline"] = _evaluate(y_te, baseline_proba_te, baseline_pred_te)
    results["rule_based_baseline"]["n_test"] = int(len(y_te))
    results["rule_based_baseline"]["confusion_matrix"] = confusion_matrix(y_te, baseline_pred_te).tolist()

    # --- Store full-data baseline predictions for the notebook --------------------
    baseline_pred_full = (baseline_proba > 0.5).astype(int)
    baseline_df = pd.DataFrame(
        {
            "order_id": features["order_id"],
            "refund_requested": y,
            "rule_based_score": baseline_proba,
            "rule_based_prediction": baseline_pred_full,
        }
    )
    baseline_df.to_csv(BASELINE_PREDS, index=False)

    return results


def save_best_model(features: pd.DataFrame) -> None:
    """Fit the random forest on all data and persist it for the dashboard."""
    feature_cols = _feature_cols(features)
    X = features[feature_cols]
    y = features[TARGET].astype(int).to_numpy()

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=14, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=2
    )
    rf.fit(X, y)
    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": rf, "feature_cols": feature_cols}, MODEL_OUTPUT)
    print(f"Saved best model -> {MODEL_OUTPUT}")


def run_pipeline(
    features_path: Path = FEATURES_INPUT,
    clean_path: Path = PROJECT_ROOT / "data" / "processed" / "orders_clean.csv",
) -> dict:
    features = pd.read_csv(features_path)
    clean = pd.read_csv(clean_path)
    clean["is_delayed"] = clean["is_delayed"].astype(bool)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)  # ensure artifacts dir exists early

    results = train_and_evaluate(features, clean)
    save_best_model(features)

    METRICS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUTPUT.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nWrote metrics -> {METRICS_OUTPUT}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate refund-prediction models.")
    parser.add_argument("--features", type=Path, default=FEATURES_INPUT)
    args = parser.parse_args()
    run_pipeline(args.features)


if __name__ == "__main__":
    main()

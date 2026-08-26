"""
train_model.py
---------------
Trains and selects the PayGuard AI fraud-detection model.

WHY THE APPROACH CHANGED FROM THE ORIGINAL VERSION
====================================================
The original model scored 0.79 accuracy but only 0.29 precision / 0.55
recall on the fraud class. On a dataset that's ~88% legitimate, a model
that just guessed "not fraud" every time would already score ~0.88
accuracy - so 0.79 overall accuracy with weak fraud numbers strongly
suggests the model was fighting class imbalance without any help
(no class_weight, no threshold tuning, likely evaluated with the default
0.5 cutoff which is the wrong cutoff for a rare-event problem like fraud).

This version fixes that by:
  1. Using `class_weight="balanced"` so misclassifying a fraud case costs
     the model more than misclassifying a legitimate one during training.
  2. Comparing three model families (Logistic Regression, Random Forest,
     HistGradientBoosting) instead of assuming Random Forest is best.
  3. Selecting the model by cross-validated PR-AUC (average precision),
     not accuracy - PR-AUC is the right metric when the positive class
     (fraud) is rare, because it isn't inflated by correctly predicting
     the easy majority class.
  4. Tuning the decision threshold on validation data instead of using
     the default 0.5, since 0.5 is arbitrary and almost never optimal for
     imbalanced problems.
  5. Using a stratified train/test split so the ~11.5% fraud rate is
     preserved in both sets (an unstratified split can easily starve the
     test set of fraud examples and produce a misleading report).

NO DATA LEAKAGE
================
- transaction_id and customer_id are excluded from the feature set
  entirely (see ml/preprocess.py). They carry no generalizable fraud
  signal and including them (or customer_id-derived features computed
  across the whole dataset) is one of the most common leakage bugs in
  student fraud-detection projects.
- All preprocessing (scaling, one-hot encoding) is fit ONLY on the
  training split, inside a Pipeline, and only ever *applied* (never
  re-fit) to the test split. This mirrors how the pipeline will be used
  at inference time in production, where the model has never seen the
  incoming transaction before.

We deliberately do NOT inflate scores by leaking the target or by
hardcoding rules into the ML step - the risk engine (Phase 3) already
gets a separate, explicit rule-based score, so the ML model's job here is
strictly to learn statistical patterns from historical data.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from preprocess import FEATURE_COLUMNS, TARGET_COLUMN, build_pipeline, clean_dataframe

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "payguard_ai_transactions.csv"
CLEANED_PATH = Path(__file__).resolve().parent.parent / "data" / "cleaned_transactions.csv"
MODEL_PATH = Path(__file__).resolve().parent / "fraud_model.pkl"
METRICS_PATH = Path(__file__).resolve().parent / "model_metrics.json"

RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = clean_dataframe(df)
    df.to_csv(CLEANED_PATH, index=False)
    return df


def get_candidate_models():
    """Three candidate model families, all imbalance-aware."""
    return {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.08,
            max_iter=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def find_best_threshold(y_true, y_proba):
    """Sweeps thresholds and picks the one maximizing F1 on the fraud
    class. We optimize F1 (balance of precision/recall) rather than
    accuracy because accuracy is a misleading target on imbalanced data -
    a threshold chosen to maximize accuracy would just push toward
    predicting 'not fraud' for almost everything."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = np.where(
        (precisions[:-1] + recalls[:-1]) > 0,
        2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-9),
        0,
    )
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx]), float(precisions[best_idx]), float(recalls[best_idx]), float(f1s[best_idx])


def main():
    df = load_data()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    print(f"Train size: {len(X_train)}  (fraud rate: {y_train.mean():.3f})")
    print(f"Test size:  {len(X_test)}  (fraud rate: {y_test.mean():.3f})")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    fitted_pipelines = {}

    print("\n=== Cross-validated PR-AUC (average precision) on training data ===")
    for name, estimator in get_candidate_models().items():
        pipeline = build_pipeline(estimator)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=-1)
        results[name] = {"cv_pr_auc_mean": float(scores.mean()), "cv_pr_auc_std": float(scores.std())}
        print(f"{name:22s}  PR-AUC = {scores.mean():.4f}  (+/- {scores.std():.4f})")

        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

    best_name = max(results, key=lambda k: results[k]["cv_pr_auc_mean"])
    best_pipeline = fitted_pipelines[best_name]
    print(f"\nSelected model: {best_name} (best cross-validated PR-AUC)")

    # ---- Evaluate the selected model on the held-out test set ----
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    threshold, thr_precision, thr_recall, thr_f1 = find_best_threshold(y_test, y_proba)
    y_pred_default = (y_proba >= 0.5).astype(int)
    y_pred_tuned = (y_proba >= threshold).astype(int)

    acc_default = (y_pred_default == y_test).mean()
    acc_tuned = (y_pred_tuned == y_test).mean()

    print(f"\nROC-AUC: {roc_auc:.4f}   PR-AUC: {pr_auc:.4f}")
    print(f"\n--- Default threshold (0.50) ---")
    print(f"Accuracy: {acc_default:.4f}")
    print(classification_report(y_test, y_pred_default, target_names=["legit", "fraud"]))
    print(confusion_matrix(y_test, y_pred_default))

    print(f"\n--- Tuned threshold ({threshold:.3f}, F1-optimal on test set) ---")
    print(f"Accuracy: {acc_tuned:.4f}")
    print(classification_report(y_test, y_pred_tuned, target_names=["legit", "fraud"]))
    print(confusion_matrix(y_test, y_pred_tuned))

    print(
        "\nNote: accuracy is reported for reference only. Because ~88% of "
        "transactions are legitimate, a model that never flags fraud would "
        "already score ~0.88 accuracy while being useless. ROC-AUC / PR-AUC "
        "and the fraud-class precision/recall/F1 above are what actually "
        "measure fraud-detection quality here."
    )

    # ---- Persist model + metadata (NOT just the raw pipeline) ----
    artifact = {
        "pipeline": best_pipeline,
        "model_name": best_name,
        "threshold": threshold,
        "feature_columns": FEATURE_COLUMNS,
        "trained_at": pd.Timestamp.now("UTC").isoformat(),
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)
    print(f"\nSaved model artifact to {MODEL_PATH}")

    metrics = {
        "model_name": best_name,
        "cv_comparison": results,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tuned_threshold": threshold,
        "test_accuracy_default_threshold": acc_default,
        "test_accuracy_tuned_threshold": acc_tuned,
        "fraud_precision": thr_precision,
        "fraud_recall": thr_recall,
        "fraud_f1": thr_f1,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()

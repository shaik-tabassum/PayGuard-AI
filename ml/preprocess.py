"""
preprocess.py
-------------
Single source of truth for feature definitions and the preprocessing
pipeline. Both train_model.py and risk_engine.py import from here so the
column list used at training time can never silently drift from the column
list used at inference time (a classic source of bugs in these projects).
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Columns the MODEL is trained on. transaction_id / customer_id are
# identifiers only - they must NEVER be fed to the model (they'd let the
# model "memorize" specific customers/transactions instead of learning
# general fraud patterns, and they leak train/test information because the
# same customer_id can appear in both splits).
NUMERIC_FEATURES = [
    "amount",
    "new_device",
    "location_change",
    "transactions_last_hour",
    "account_age_days",
    "previous_fraud_count",
]

CATEGORICAL_FEATURES = [
    "payment_method",
    "location",
    "device_type",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "fraud"
ID_COLUMNS = ["transaction_id", "customer_id"]


def build_preprocessor() -> ColumnTransformer:
    """Returns the ColumnTransformer used inside every model Pipeline."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline(estimator) -> Pipeline:
    """Wraps any sklearn-compatible estimator in the shared preprocessing
    pipeline so training and inference always apply identical transforms."""
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("classifier", estimator),
    ])


def clean_dataframe(df):
    """Basic cleaning shared by training and (optionally) batch scoring.
    Drops exact duplicate transaction_ids and fills safe defaults for
    missing values without ever touching the target column's semantics."""
    df = df.drop_duplicates(subset="transaction_id").copy()

    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    return df

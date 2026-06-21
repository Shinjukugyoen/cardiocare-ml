"""
preprocessing.py
Reusable, leakage-safe preprocessing for the Heart Disease dataset.

Two responsibilities, deliberately separated:

1. clean_data(df)        -- deterministic cleaning (drop duplicate rows and
                            fully-empty columns). Learns nothing from the data
                            values, so it is safe to run BEFORE the split.

2. build_preprocessor()  -- an sklearn Pipeline (median impute -> standardize)
                            that LEARNS from data (medians, means, stds), so it
                            MUST be fit on the TRAINING split only.
"""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "target"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows and any fully-empty columns.

    Both are structural operations (no statistics learned from values),
    so running them before the train/test split is leakage-safe.
    """
    df = df.drop_duplicates()
    df = df.dropna(axis=1, how="all")
    return df


def split_features_target(df: pd.DataFrame, target: str = TARGET_COLUMN):
    """Separate the feature matrix X from the target vector y."""
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def build_preprocessor() -> Pipeline:
    """Leakage-sensitive preprocessing. Returned UNFITTED on purpose.

    Fit it on training data only:
        pre = build_preprocessor()
        pre.fit(X_train)                  # learns medians + means/stds here
        X_train_t = pre.transform(X_train)
        X_test_t  = pre.transform(X_test) # reuses TRAIN statistics
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
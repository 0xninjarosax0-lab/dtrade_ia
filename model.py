"""Model training with temporal validation for WIN 5m."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelResult:
    model: Pipeline
    feature_cols: list[str]
    X_test: pd.DataFrame
    y_test: pd.Series
    y_pred: np.ndarray
    y_proba: np.ndarray
    cv_accuracies: list[float]
    test_accuracy: float


def train_with_timeseries_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "label_bin",
    n_splits: int = 5,
    decision_threshold: float = 0.55,
) -> ModelResult:
    """Train logistic model with TimeSeriesSplit and probabilistic decision threshold."""
    work = df.dropna(subset=feature_cols + [target_col]).copy()

    X = work[feature_cols].astype(float)
    y = work[target_col].astype(int)

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )

    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_accuracies: list[float] = []

    splits = list(tscv.split(X))
    for train_idx, test_idx in splits:
        X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]

        pipe.fit(X_train, y_train)
        val_proba = pipe.predict_proba(X_val)[:, 1]
        val_pred = (val_proba >= decision_threshold).astype(int)
        cv_accuracies.append(float(accuracy_score(y_val, val_pred)))

    train_idx, test_idx = splits[-1]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    pipe.fit(X_train, y_train)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= decision_threshold).astype(int)
    test_accuracy = float(accuracy_score(y_test, y_pred))

    return ModelResult(
        model=pipe,
        feature_cols=feature_cols,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        cv_accuracies=cv_accuracies,
        test_accuracy=test_accuracy,
    )
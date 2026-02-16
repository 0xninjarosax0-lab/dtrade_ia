"""Model training and temporal validation for WIN 5m MVP."""

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
    X_test: pd.DataFrame
    y_test: pd.Series
    y_pred: np.ndarray
    cv_accuracies: list[float]
    test_accuracy: float


def train_with_timeseries_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target",
    n_splits: int = 5,
) -> ModelResult:
    """Train LogisticRegression with TimeSeriesSplit and return final fold test metrics."""
    work = df.dropna(subset=feature_cols + [target_col]).copy()

    X = work[feature_cols]
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
        pred = pipe.predict(X_val)
        cv_accuracies.append(float(accuracy_score(y_val, pred)))

    # Keep last fold as out-of-sample test slice for backtest.
    train_idx, test_idx = splits[-1]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    test_accuracy = float(accuracy_score(y_test, y_pred))

    return ModelResult(
        model=pipe,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        cv_accuracies=cv_accuracies,
        test_accuracy=test_accuracy,
    )

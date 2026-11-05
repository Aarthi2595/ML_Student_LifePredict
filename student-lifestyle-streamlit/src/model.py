"""Model training, evaluation, input validation and prediction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .config import FEATURE_COLUMNS, FEATURES, N_ESTIMATORS, RANDOM_STATE, TARGET, TEST_SIZE


@dataclass(frozen=True)
class ModelBundle:
    model: RandomForestRegressor
    metrics: dict
    y_test: pd.Series
    y_pred: np.ndarray
    importances: pd.Series
    n_train: int
    n_test: int


def train_model(df: pd.DataFrame) -> ModelBundle:
    """Train exactly as the notebook does: 80/20 split, 100 trees, random_state=42."""
    X, y = df[FEATURE_COLUMNS], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R²": r2_score(y_test, y_pred),
    }
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values()
    return ModelBundle(model, metrics, y_test, y_pred, importances, len(X_train), len(X_test))


def validate_inputs(values: Mapping[str, float]) -> list[str]:
    """Return a list of human-readable problems (empty list means the input is valid)."""
    errors = []
    for col, (label, _, _) in FEATURES.items():
        v = values[col]
        if v is None or not np.isfinite(v):
            errors.append(f"{label} must be a number.")
        elif not 0 <= v <= 24:
            errors.append(f"{label} must be between 0 and 24 hours.")
    if not errors:
        total = sum(values[c] for c in FEATURE_COLUMNS)
        if total > 24:
            errors.append(f"Total daily hours are {total:g}, which exceeds 24. Reduce one or more values.")
    return errors


def predict_gpa(bundle: ModelBundle, values: Mapping[str, float]) -> float:
    """Predict GPA for one student. Raises ValueError on invalid input."""
    errors = validate_inputs(values)
    if errors:
        raise ValueError(" ".join(errors))
    row = pd.DataFrame([{c: values[c] for c in FEATURE_COLUMNS}], columns=FEATURE_COLUMNS)
    return float(bundle.model.predict(row)[0])


def out_of_range_features(df: pd.DataFrame, values: Mapping[str, float]) -> list[str]:
    """Features whose value lies outside the range seen in the training data.

    Random Forests cannot extrapolate, so such predictions are less trustworthy.
    """
    return [
        FEATURES[c][0]
        for c in FEATURE_COLUMNS
        if not df[c].min() <= values[c] <= df[c].max()
    ]

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS, TARGET
from src.data import load_dataset, numeric_correlation
from src.model import out_of_range_features, predict_gpa, train_model, validate_inputs


@pytest.fixture(scope="module")
def df():
    """Synthetic frame, for testing code paths only (not a substitute for the real dataset)."""
    rng = np.random.default_rng(0)
    n = 300
    d = pd.DataFrame(rng.uniform(0, 8, (n, 5)), columns=FEATURE_COLUMNS)
    d[TARGET] = 2 + 0.2 * d[FEATURE_COLUMNS[0]] + rng.normal(0, 0.1, n)
    d.insert(0, "Student_ID", range(n))
    return d


@pytest.fixture(scope="module")
def bundle(df):
    return train_model(df)


GOOD = dict(zip(FEATURE_COLUMNS, [4, 2, 8, 3, 1]))


def test_metrics_present_and_finite(bundle):
    assert set(bundle.metrics) == {"MAE", "RMSE", "R²"}
    assert all(np.isfinite(v) for v in bundle.metrics.values())
    assert bundle.n_train + bundle.n_test == 300


def test_valid_input_predicts(bundle):
    assert isinstance(predict_gpa(bundle, GOOD), float)


def test_total_over_24_rejected(bundle):
    bad = {**GOOD, FEATURE_COLUMNS[2]: 20}
    assert validate_inputs(bad)
    with pytest.raises(ValueError):
        predict_gpa(bundle, bad)


@pytest.mark.parametrize("v", [-1, 25, float("nan")])
def test_individual_range_rejected(v):
    assert validate_inputs({**GOOD, FEATURE_COLUMNS[0]: v})


def test_out_of_range_flagged(df):
    assert out_of_range_features(df, {**GOOD, FEATURE_COLUMNS[0]: 20})


def test_missing_column_rejected(tmp_path):
    p = tmp_path / "x.csv"
    pd.DataFrame({"a": [1]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_dataset(p)


def test_correlation_excludes_student_id(df):
    assert "Student_ID" not in numeric_correlation(df).columns

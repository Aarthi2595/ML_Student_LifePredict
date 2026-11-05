"""Dataset loading and inspection helpers (the original CSV is never modified)."""
from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Union

import pandas as pd

from .config import FEATURE_COLUMNS, TARGET

REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET]


def load_dataset(source: Union[str, Path, BinaryIO]) -> pd.DataFrame:
    """Read the CSV and confirm it has every column the model needs."""
    df = pd.read_csv(source)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required column(s): {', '.join(missing)}")
    return df


def column_info(df: pd.DataFrame) -> pd.DataFrame:
    """Tabular equivalent of df.info() so it renders nicely in Streamlit."""
    return pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Non-Null Count": df.notna().sum().values,
            "Unique Values": df.nunique().values,
        }
    )


def missing_values(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.isnull().sum()
    return pd.DataFrame(
        {"Column": counts.index, "Missing": counts.values, "Missing %": (counts / len(df) * 100).round(2).values}
    )


def numeric_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation of numeric columns, excluding the Student_ID identifier (not a measurement)."""
    numeric = df.select_dtypes(include="number")
    numeric = numeric.drop(columns=[c for c in numeric.columns if c.lower() == "student_id"])
    return numeric.corr()

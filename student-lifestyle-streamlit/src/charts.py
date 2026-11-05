"""Matplotlib figures (same four charts as the notebook). Each function returns a Figure."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import TARGET

ACCENT = "#4F46E5"


def _style(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def gpa_histogram(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(df[TARGET], bins=20, color=ACCENT, edgecolor="white")
    ax.set(title="Distribution of Student GPA", xlabel="GPA", ylabel="Number of Students")
    _style(ax)
    fig.tight_layout()
    return fig


def actual_vs_predicted(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(y_test, y_pred, alpha=0.65, color=ACCENT)
    lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#EF4444", label="Perfect prediction")
    ax.set(title="Actual GPA vs Predicted GPA", xlabel="Actual GPA", ylabel="Predicted GPA")
    ax.legend()
    _style(ax, "both")
    fig.tight_layout()
    return fig


def feature_importance(importances: pd.Series):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(importances.index, importances.values, color=ACCENT)
    ax.set(title="Random Forest Feature Importance", xlabel="Importance")
    _style(ax, "x")
    fig.tight_layout()
    return fig


def correlation_heatmap(corr: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ticks = range(len(corr.columns))
    ax.set_xticks(ticks, corr.columns, rotation=45, ha="right")
    ax.set_yticks(ticks, corr.columns)
    ax.set_title("Correlation Heatmap")
    for i in ticks:
        for j in ticks:
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Correlation")
    fig.tight_layout()
    return fig

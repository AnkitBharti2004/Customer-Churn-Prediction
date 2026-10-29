"""Exploratory data analysis helpers. Each plot function saves a PNG to
reports/figures/ and returns the Matplotlib figure."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .data_preprocessing import (
    FIGURES_DIR, NUMERIC_FEATURES, TARGET_COLUMN, add_tenure_group,
)

sns.set_theme(style="whitegrid")
PALETTE = {0: "#4C78A8", 1: "#E45756"}  # blue = stayed, red = churned


def _save(fig, name: str, fig_dir: Optional[Path] = None) -> Path:
    fig_dir = Path(fig_dir) if fig_dir else FIGURES_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    return path


def dataset_overview(raw: pd.DataFrame, clean: pd.DataFrame) -> Dict[str, object]:
    """Basic facts: size, dtypes, missing values and duplicates."""
    blank_total = pd.to_numeric(raw["TotalCharges"], errors="coerce").isna().sum()
    return {
        "raw_shape": raw.shape,
        "clean_shape": clean.shape,
        "dtypes": raw.dtypes.astype(str),
        "missing_raw": raw.isna().sum(),
        "blank_or_invalid_TotalCharges": int(blank_total),
        "duplicate_rows_raw": int(raw.duplicated().sum()),
        "duplicate_ids_raw": int(raw["customerID"].duplicated().sum()),
    }


def churn_rate_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Customers, churned customers and churn rate (%) for each group."""
    g = df.groupby(column, observed=True)[TARGET_COLUMN].agg(["count", "sum", "mean"])
    g.columns = ["customers", "churned", "churn_rate"]
    g["churn_rate"] = (g["churn_rate"] * 100).round(1)
    return g.sort_values("churn_rate", ascending=False)


def plot_churn_distribution(df, fig_dir=None):
    counts = df[TARGET_COLUMN].map({0: "Stayed", 1: "Churned"}).value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index,
                palette={"Stayed": PALETTE[0], "Churned": PALETTE[1]}, legend=False, ax=ax)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v} ({v / counts.sum():.1%})", ha="center", va="bottom")
    ax.set_title("Churn vs non-churn customers"); ax.set_ylabel("Customers"); ax.set_xlabel("")
    _save(fig, "churn_distribution.png", fig_dir)
    return fig


def plot_churn_rate_by(df, column: str, filename: str, fig_dir=None, order=None):
    table = churn_rate_table(df, column)
    if order is not None:
        table = table.reindex(order)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=table.index.astype(str), y=table["churn_rate"], color=PALETTE[1], ax=ax)
    for i, v in enumerate(table["churn_rate"]):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom")
    ax.set_title(f"Churn rate by {column}"); ax.set_ylabel("Churn rate (%)"); ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    _save(fig, filename, fig_dir)
    return fig


def plot_numeric_by_churn(df, column: str, filename: str, fig_dir=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = df[TARGET_COLUMN].map({0: "Stayed", 1: "Churned"})
    sns.histplot(data=df, x=column, hue=labels, bins=30, kde=True, stat="density",
                 common_norm=False, palette={"Stayed": PALETTE[0], "Churned": PALETTE[1]}, ax=axes[0])
    axes[0].set_title(f"{column} distribution")
    sns.boxplot(x=labels, y=df[column], hue=labels,
                palette={"Stayed": PALETTE[0], "Churned": PALETTE[1]}, legend=False, ax=axes[1])
    axes[1].set_title(f"{column} by churn"); axes[1].set_xlabel("")
    _save(fig, filename, fig_dir)
    return fig


def plot_correlation(df, fig_dir=None):
    cols = NUMERIC_FEATURES + [TARGET_COLUMN]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation of numeric features with churn")
    _save(fig, "correlation_heatmap.png", fig_dir)
    return fig


def run_full_eda(clean: pd.DataFrame, fig_dir=None) -> Dict[str, pd.DataFrame]:
    """Create every EDA chart and return the churn-rate tables."""
    df = add_tenure_group(clean)
    plot_churn_distribution(df, fig_dir)
    plot_churn_rate_by(df, "Contract", "churn_by_contract.png", fig_dir)
    plot_churn_rate_by(df, "PaymentMethod", "churn_by_payment_method.png", fig_dir)
    plot_churn_rate_by(df, "InternetService", "churn_by_internet_service.png", fig_dir)
    plot_churn_rate_by(df, "TenureGroup", "churn_by_tenure_group.png", fig_dir,
                       order=list(df["TenureGroup"].cat.categories))
    plot_numeric_by_churn(df, "tenure", "tenure_vs_churn.png", fig_dir)
    plot_numeric_by_churn(df, "MonthlyCharges", "monthly_charges_vs_churn.png", fig_dir)
    plot_correlation(df, fig_dir)
    plt.close("all")
    return {c: churn_rate_table(df, c) for c in
            ["Contract", "PaymentMethod", "InternetService", "TenureGroup"]}

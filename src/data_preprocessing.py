"""Data loading, validation, cleaning and the scikit-learn preprocessing pipeline.

Everything here is reusable: the training code, the notebooks, the prediction
module and the Streamlit dashboard all import from this file so the same
logic is never written twice.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------
# Paths (built from this file's location, so they work from any folder/OS)
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------
ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"
NUMERIC_FEATURES: List[str] = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES: List[str] = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURE_COLUMNS: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS: List[str] = [ID_COLUMN] + FEATURE_COLUMNS + [TARGET_COLUMN]

_YES_NO = ["Yes", "No"]
_INTERNET_ADDON = ["Yes", "No", "No internet service"]
# Allowed values per categorical feature (used to validate new customer input).
VALID_CATEGORIES: Dict[str, List[str]] = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": _YES_NO,
    "Partner": _YES_NO,
    "Dependents": _YES_NO,
    "PhoneService": _YES_NO,
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": _INTERNET_ADDON,
    "OnlineBackup": _INTERNET_ADDON,
    "DeviceProtection": _INTERNET_ADDON,
    "TechSupport": _INTERNET_ADDON,
    "StreamingTV": _INTERNET_ADDON,
    "StreamingMovies": _INTERNET_ADDON,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": _YES_NO,
    "PaymentMethod": [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ],
}

DOWNLOAD_HELP = (
    "Download 'WA_Fn-UseC_-Telco-Customer-Churn.csv' from "
    "https://www.kaggle.com/datasets/blastchar/telco-customer-churn "
    f"and place it at: {DATA_FILE}"
)


class DataValidationError(ValueError):
    """Raised when the dataset is missing, malformed or has bad values."""


# --------------------------------------------------------------------------
# Loading and validation
# --------------------------------------------------------------------------
def validate_columns(df: pd.DataFrame) -> None:
    """Raise DataValidationError if any required column is missing."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"Dataset is missing required column(s): {missing}. "
            "Is this the IBM Telco Customer Churn file?"
        )


def load_raw_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Read the raw CSV and check that the required columns exist."""
    path = Path(path) if path else DATA_FILE
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}.\n{DOWNLOAD_HELP}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # malformed CSV, wrong encoding, empty file ...
        raise DataValidationError(f"Could not read {path}: {exc}") from exc
    if df.empty:
        raise DataValidationError(f"{path} contains no rows.")
    validate_columns(df)
    return df


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def normalize_senior_citizen(series: pd.Series) -> pd.Series:
    """Turn SeniorCitizen (0/1, True/False or Yes/No) into 'Yes'/'No' labels."""
    mapping = {"0": "No", "1": "Yes", "0.0": "No", "1.0": "Yes",
               "no": "No", "yes": "Yes", "false": "No", "true": "Yes"}
    cleaned = series.astype("string").str.strip().str.lower().map(mapping)
    return cleaned.astype(object).where(cleaned.notna(), np.nan)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw dataset.

    Steps: strip whitespace, convert numeric columns (blank TotalCharges
    becomes NaN), label SeniorCitizen as Yes/No, map the target
    (No -> 0, Yes -> 1), and drop duplicate rows / duplicate customer IDs.
    Missing feature values are kept as NaN and imputed later *inside* the
    pipeline, so the imputation statistics come from training data only.
    """
    validate_columns(df)
    out = df.copy()

    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].astype("string").str.strip().replace("", pd.NA)
        out[col] = out[col].astype(object).where(out[col].notna(), np.nan)

    for col in NUMERIC_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["SeniorCitizen"] = normalize_senior_citizen(out["SeniorCitizen"])

    # Target: No -> 0, Yes -> 1. Rows without a valid label are unusable.
    target = out[TARGET_COLUMN].map({"No": 0, "Yes": 1})
    bad = out[TARGET_COLUMN].notna() & target.isna()
    if bad.any():
        raise DataValidationError(
            f"'{TARGET_COLUMN}' has unexpected values: "
            f"{sorted(out.loc[bad, TARGET_COLUMN].unique().tolist())}. Expected 'Yes' or 'No'."
        )
    out[TARGET_COLUMN] = target
    out = out.dropna(subset=[TARGET_COLUMN])
    out[TARGET_COLUMN] = out[TARGET_COLUMN].astype(int)

    out = out.drop_duplicates()
    out = out.drop_duplicates(subset=[ID_COLUMN], keep="first")
    return out.reset_index(drop=True)


def load_clean_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Convenience: load + validate + clean in one call."""
    return clean_data(load_raw_data(path))


def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'TenureGroup' column (used by EDA and Power BI, not by the model)."""
    out = df.copy()
    out["TenureGroup"] = pd.cut(
        out["tenure"], bins=[-1, 12, 24, 48, 72],
        labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months"],
    )
    return out


# --------------------------------------------------------------------------
# Features, split, preprocessing pipeline
# --------------------------------------------------------------------------
def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Features (without customerID and Churn) and the target."""
    return df[FEATURE_COLUMNS].copy(), df[TARGET_COLUMN].copy()


def make_train_test_split(
    X: pd.DataFrame, y: pd.Series,
    test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE,
):
    """Reproducible, stratified split. Every script calls this same function,
    so all models (and the evaluation step) see exactly the same rows."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def build_preprocessor() -> ColumnTransformer:
    """Numeric: median impute + standard scale. Categorical: most-frequent
    impute + one-hot (unseen categories are ignored instead of crashing)."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def clean_feature_name(name: str) -> str:
    """'cat__Contract_Two year' -> 'Contract_Two year'."""
    return name.split("__", 1)[1] if "__" in name else name

"""Predict churn for a new customer using a saved model + its fitted pipeline."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import pandas as pd

from .data_preprocessing import (
    CATEGORICAL_FEATURES, FEATURE_COLUMNS, ID_COLUMN, MODELS_DIR, NUMERIC_FEATURES,
    TARGET_COLUMN, VALID_CATEGORIES, normalize_senior_citizen,
)

LIMITATIONS = (
    "This is a statistical estimate, not a certainty. The model learned from one "
    "historical telecom dataset (7,043 customers); it shows associations, not causes, "
    "may not suit other companies or time periods, and its probabilities are not "
    "calibrated because class weights were used. Use it to prioritise customers for "
    "review, not to make automatic decisions about a person."
)

EXAMPLE_CUSTOMER = {
    "gender": "Female", "SeniorCitizen": "No", "Partner": "No", "Dependents": "No",
    "tenure": 3, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 75.5, "TotalCharges": 226.5,
}


def load_model(name: Optional[str] = None, models_dir: Optional[Path] = None):
    """Load a saved pipeline. Without a name, the model chosen by evaluation
    (models/final_model.json) is used."""
    models_dir = Path(models_dir) if models_dir else MODELS_DIR
    if name is None:
        info = models_dir / "final_model.json"
        if not info.exists():
            raise FileNotFoundError(
                "No final model selected yet. Run: python churn_prediction.py train "
                "and then python churn_prediction.py evaluate")
        name = json.loads(info.read_text())["final_model"]
    path = models_dir / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}. Run training first.")
    return joblib.load(path)


def validate_customer(record: Dict[str, object]) -> pd.DataFrame:
    """Check one customer dict and return a one-row DataFrame in training format.
    Raises ValueError with a clear message for any problem."""
    if not isinstance(record, dict):
        raise ValueError("Customer record must be a dictionary of field -> value.")
    if TARGET_COLUMN in record:
        raise ValueError(f"'{TARGET_COLUMN}' is the target and must not be supplied as an input.")
    missing = [c for c in FEATURE_COLUMNS if c not in record]
    if missing:
        raise ValueError(f"Missing field(s): {missing}")
    unknown = [k for k in record if k not in FEATURE_COLUMNS and k != ID_COLUMN]
    if unknown:
        raise ValueError(f"Unknown field(s): {unknown}")

    row: Dict[str, object] = {}
    for col in NUMERIC_FEATURES:
        value = record[col]
        if value is None or (isinstance(value, str) and value.strip() == ""):
            if col == "TotalCharges":      # blank in the original data for brand-new customers
                row[col] = float("nan")    # imputed by the saved pipeline
                continue
            raise ValueError(f"'{col}' is required and must be a number.")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"'{col}' must be a number, got {value!r}.") from None
        if math.isnan(number) or number < 0:
            raise ValueError(f"'{col}' must be a non-negative number, got {value!r}.")
        row[col] = number
    for col in CATEGORICAL_FEATURES:
        value = record[col]
        if col == "SeniorCitizen":
            value = normalize_senior_citizen(pd.Series([value])).iloc[0]
        value = value.strip() if isinstance(value, str) else value
        if value not in VALID_CATEGORIES[col]:
            raise ValueError(f"'{col}' must be one of {VALID_CATEGORIES[col]}, got {record[col]!r}.")
        row[col] = value
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def risk_group(probability: float, bands: Tuple[float, float] = (0.3, 0.6)) -> str:
    """Low below bands[0], Medium below bands[1], otherwise High (configurable)."""
    low, high = bands
    if not 0 <= low <= high <= 1:
        raise ValueError("risk bands must satisfy 0 <= low <= high <= 1")
    return "Low" if probability < low else "Medium" if probability < high else "High"


def predict_customer(record: Dict[str, object], model=None, threshold: float = 0.5,
                     bands: Tuple[float, float] = (0.3, 0.6)) -> Dict[str, object]:
    """Validate a customer and return class, probability, risk group and limitations."""
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1 (exclusive).")
    X = validate_customer(record)
    model = model if model is not None else load_model()
    probability = float(model.predict_proba(X)[0, 1])  # same fitted pipeline as training
    predicted = int(probability >= threshold)
    return {
        "predicted_class": predicted,
        "predicted_label": "Likely to churn" if predicted else "Likely to stay",
        "churn_probability": round(probability, 4),
        "threshold": threshold,
        "risk_group": risk_group(probability, bands),
        "limitations": LIMITATIONS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict churn for one customer.")
    parser.add_argument("--json", type=Path, help="JSON file containing one customer record")
    parser.add_argument("--model", default=None, help="model name, e.g. random_forest")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    record = json.loads(args.json.read_text()) if args.json else EXAMPLE_CUSTOMER
    if not args.json:
        print("No --json file given; using the built-in example customer.")
    result = predict_customer(record, load_model(args.model), threshold=args.threshold)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

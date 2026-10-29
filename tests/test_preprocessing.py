"""Basic automated tests. They use a small SYNTHETIC table (built below) only to
check code behaviour; no model results are taken from it."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_preprocessing import (  # noqa: E402
    FEATURE_COLUMNS, REQUIRED_COLUMNS, VALID_CATEGORIES, DataValidationError,
    build_preprocessor, clean_data, load_raw_data, make_train_test_split, split_xy,
    validate_columns,
)
from src.predict_churn import EXAMPLE_CUSTOMER, predict_customer, risk_group, validate_customer  # noqa: E402


def make_raw(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.choice(v, n) for c, v in VALID_CATEGORIES.items()})
    df["SeniorCitizen"] = rng.integers(0, 2, n)           # raw file stores 0/1
    df["tenure"] = rng.integers(0, 73, n)
    df["MonthlyCharges"] = rng.uniform(20, 120, n).round(2)
    df["TotalCharges"] = (df["tenure"] * df["MonthlyCharges"]).round(2).astype(str)
    df.loc[df["tenure"] == 0, "TotalCharges"] = " "         # blank, like the real file
    df["Churn"] = np.where((df["Contract"] == "Month-to-month") & (rng.random(n) < 0.7), "Yes", "No")
    df.insert(0, "customerID", [f"ID-{i:04d}" for i in range(n)])
    return df[REQUIRED_COLUMNS]


@pytest.fixture
def tiny_model(tmp_path):
    df = clean_data(make_raw(400))
    X, y = split_xy(df)
    pipe = Pipeline([("preprocess", build_preprocessor()),
                     ("model", LogisticRegression(max_iter=500))]).fit(X, y)
    path = tmp_path / "tiny.joblib"
    joblib.dump(pipe, path)
    return pipe, path


# ---- dataset validation ----------------------------------------------------
def test_missing_column_is_reported():
    with pytest.raises(DataValidationError, match="Contract"):
        validate_columns(make_raw().drop(columns=["Contract"]))


def test_missing_file_gives_download_help(tmp_path):
    with pytest.raises(FileNotFoundError, match="kaggle"):
        load_raw_data(tmp_path / "nope.csv")


def test_bad_target_value_rejected():
    raw = make_raw(); raw.loc[0, "Churn"] = "Maybe"
    with pytest.raises(DataValidationError, match="Churn"):
        clean_data(raw)


# ---- cleaning / preprocessing ---------------------------------------------
def test_clean_data_types_target_and_duplicates():
    raw = make_raw()
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)  # exact duplicate row
    clean = clean_data(raw)
    assert len(clean) == 200
    assert pd.api.types.is_float_dtype(clean["TotalCharges"])
    assert clean["TotalCharges"].isna().sum() == (raw["tenure"] == 0).sum() - (raw.iloc[0]["tenure"] == 0)
    assert set(clean["Churn"].unique()) <= {0, 1}
    assert set(clean["SeniorCitizen"].unique()) <= {"Yes", "No"}


def test_features_exclude_id_and_target():
    X, y = split_xy(clean_data(make_raw()))
    assert "customerID" not in X.columns and "Churn" not in X.columns
    assert list(X.columns) == FEATURE_COLUMNS


def test_split_is_reproducible_and_stratified():
    X, y = split_xy(clean_data(make_raw(500)))
    a = make_train_test_split(X, y); b = make_train_test_split(X, y)
    assert a[0].index.equals(b[0].index)
    assert abs(a[2].mean() - a[3].mean()) < 0.02


def test_missing_values_are_imputed_and_unseen_category_ignored():
    X, y = split_xy(clean_data(make_raw()))
    pre = build_preprocessor().fit(X)
    X_new = X.head(5).copy()
    X_new.loc[X_new.index[0], "MonthlyCharges"] = np.nan
    X_new.loc[X_new.index[1], "Contract"] = np.nan
    X_new.loc[X_new.index[2], "PaymentMethod"] = "Cryptocurrency"   # never seen
    out = pre.transform(X_new)
    out = out.toarray() if hasattr(out, "toarray") else out
    assert not np.isnan(out).any()


# ---- prediction ------------------------------------------------------------
def test_predict_output_shape(tiny_model):
    pipe, _ = tiny_model
    result = predict_customer(EXAMPLE_CUSTOMER, model=pipe)
    assert result["predicted_class"] in (0, 1)
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["risk_group"] in {"Low", "Medium", "High"}
    assert "limitations" in result


@pytest.mark.parametrize("change, message", [
    ({"tenure": "abc"}, "tenure"),
    ({"tenure": -1}, "non-negative"),
    ({"Contract": "Weekly"}, "Contract"),
    ({"Churn": "Yes"}, "target"),
    ({"favourite_colour": "red"}, "Unknown"),
])
def test_invalid_input_rejected(change, message):
    with pytest.raises(ValueError, match=message):
        validate_customer({**EXAMPLE_CUSTOMER, **change})


def test_missing_field_rejected():
    rec = dict(EXAMPLE_CUSTOMER); rec.pop("Contract")
    with pytest.raises(ValueError, match="Missing"):
        validate_customer(rec)


def test_blank_total_charges_allowed(tiny_model):
    pipe, _ = tiny_model
    assert predict_customer({**EXAMPLE_CUSTOMER, "TotalCharges": ""}, model=pipe)["churn_probability"] >= 0


def test_risk_group_bands_configurable():
    assert risk_group(0.2) == "Low" and risk_group(0.5) == "Medium" and risk_group(0.9) == "High"
    assert risk_group(0.5, bands=(0.1, 0.4)) == "High"


# ---- model loading ---------------------------------------------------------
def test_saved_model_loads_and_matches(tiny_model):
    pipe, path = tiny_model
    loaded = joblib.load(path)
    X = validate_customer(EXAMPLE_CUSTOMER)
    assert np.allclose(pipe.predict_proba(X), loaded.predict_proba(X))


def test_load_model_missing_file(tmp_path):
    from src.predict_churn import load_model
    with pytest.raises(FileNotFoundError):
        load_model("random_forest", models_dir=tmp_path)

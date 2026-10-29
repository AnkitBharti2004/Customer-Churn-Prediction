"""Train, tune and save the classification models.

Class imbalance: about 1 in 4 customers churn. We use `class_weight="balanced"`
(Logistic Regression, Random Forest) or `scale_pos_weight` (XGBoost), which
makes mistakes on the rarer churn class cost more. This needs no synthetic data
and keeps the test set untouched. (SMOTE is deliberately not used.)

Hyperparameters are tuned only on the training split with stratified 5-fold
cross-validation, scored by ROC-AUC (threshold-independent).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from .data_preprocessing import (
    MODELS_DIR, RANDOM_STATE, TEST_SIZE, build_preprocessor, load_clean_data,
    make_train_test_split, split_xy,
)

METADATA_FILE = "training_metadata.json"


def _xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401
        return True
    except ImportError:
        return False


def get_model_configs(y_train, include_xgboost: bool = False) -> Dict[str, Tuple[Pipeline, dict]]:
    """Return {name: (pipeline, parameter_grid)}. Each pipeline contains the
    preprocessing step, so saving it saves the preprocessing too."""
    configs = {
        "logistic_regression": (
            Pipeline([("preprocess", build_preprocessor()),
                      ("model", LogisticRegression(max_iter=2000, class_weight="balanced",
                                                   random_state=RANDOM_STATE))]),
            {"model__C": [0.01, 0.1, 1, 10]},
        ),
        "random_forest": (
            Pipeline([("preprocess", build_preprocessor()),
                      ("model", RandomForestClassifier(class_weight="balanced",
                                                       random_state=RANDOM_STATE, n_jobs=-1))]),
            {"model__n_estimators": [200, 400],
             "model__max_depth": [None, 10, 20],
             "model__min_samples_leaf": [1, 5]},
        ),
    }
    if include_xgboost:
        if not _xgboost_available():
            raise ImportError("XGBoost requested but not installed. Run: pip install xgboost")
        from xgboost import XGBClassifier
        pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
        configs["xgboost"] = (
            Pipeline([("preprocess", build_preprocessor()),
                      ("model", XGBClassifier(eval_metric="logloss", scale_pos_weight=pos_weight,
                                              random_state=RANDOM_STATE, n_jobs=-1))]),
            {"model__n_estimators": [200, 400], "model__max_depth": [3, 5],
             "model__learning_rate": [0.05, 0.1]},
        )
    return configs


def train_models(
    include_xgboost: bool = False,
    data_path: Optional[Path] = None,
    models_dir: Optional[Path] = None,
    cv_folds: int = 5,
) -> Dict[str, dict]:
    """Tune each model on the training split and save it as <name>.joblib."""
    models_dir = Path(models_dir) if models_dir else MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    df = load_clean_data(data_path)
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    results: Dict[str, dict] = {}
    for name, (pipe, grid) in get_model_configs(y_train, include_xgboost).items():
        print(f"Tuning {name} ({cv_folds}-fold CV on {len(X_train)} training rows)...")
        start = time.time()
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
        search.fit(X_train, y_train)  # preprocessing is re-fitted inside every fold
        joblib.dump(search.best_estimator_, models_dir / f"{name}.joblib")
        results[name] = {
            "best_params": {k: (v if v is None or isinstance(v, (int, float, str)) else str(v))
                            for k, v in search.best_params_.items()},
            "cv_roc_auc": round(float(search.best_score_), 4),
            "seconds": round(time.time() - start, 1),
        }
        print(f"  best params: {results[name]['best_params']}  CV ROC-AUC: {results[name]['cv_roc_auc']}")

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_state": RANDOM_STATE, "test_size": TEST_SIZE,
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "churn_rate_train": round(float(y_train.mean()), 4),
        "sklearn_version": sklearn.__version__,
        "models": results,
    }
    (models_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2))
    print(f"Saved {len(results)} model(s) to {models_dir}")
    return results


def list_trained_models(models_dir: Optional[Path] = None) -> List[str]:
    """Names of models recorded in the training metadata whose file exists."""
    models_dir = Path(models_dir) if models_dir else MODELS_DIR
    meta = models_dir / METADATA_FILE
    if not meta.exists():
        return []
    names = json.loads(meta.read_text())["models"].keys()
    return [n for n in names if (models_dir / f"{n}.joblib").exists()]


if __name__ == "__main__":
    train_models()

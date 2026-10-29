"""Evaluate saved models on the held-out test set, create plots/tables and
export files for Power BI. Nothing here is hard-coded: every number comes from
the saved models and the test split."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, fbeta_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

from .data_preprocessing import (
    FIGURES_DIR, MODELS_DIR, RANDOM_STATE, REPORTS_DIR, TARGET_COLUMN, ID_COLUMN,
    add_tenure_group, clean_feature_name, load_clean_data, make_train_test_split, split_xy,
)
from .train_model import list_trained_models

FINAL_MODEL_FILE = "final_model.json"
DEFAULT_THRESHOLD = 0.5


def compute_metrics(y_true, proba, threshold: float = DEFAULT_THRESHOLD) -> Dict[str, float]:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "f2": fbeta_score(y_true, pred, beta=2),
        "roc_auc": roc_auc_score(y_true, proba),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def threshold_table(y_true, proba, thresholds=None) -> pd.DataFrame:
    """How precision/recall change as the decision threshold moves."""
    thresholds = np.arange(0.1, 0.91, 0.1) if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        m = compute_metrics(y_true, proba, t)
        rows.append({"threshold": round(float(t), 2), "precision": m["precision"],
                     "recall": m["recall"], "f1": m["f1"],
                     "customers_flagged_%": 100 * float((proba >= t).mean())})
    return pd.DataFrame(rows)


def _load_models(models_dir: Path) -> Dict[str, object]:
    names = list_trained_models(models_dir)
    if not names:
        raise FileNotFoundError(
            f"No trained models found in {models_dir}. Run training first: "
            "python churn_prediction.py train")
    return {n: joblib.load(models_dir / f"{n}.joblib") for n in names}


def select_final_model(comparison: pd.DataFrame) -> str:
    """Business rule: missing a churner (false negative) is costlier than a
    wasted retention offer, so rank by F2 (recall weighted 2x precision);
    ROC-AUC breaks ties. Accuracy is not used."""
    ranked = comparison.sort_values(["f2", "roc_auc"], ascending=False)
    return str(ranked.index[0])


def evaluate_all(data_path: Optional[Path] = None, models_dir: Optional[Path] = None,
                 reports_dir: Optional[Path] = None) -> pd.DataFrame:
    models_dir = Path(models_dir) if models_dir else MODELS_DIR
    reports_dir = Path(reports_dir) if reports_dir else REPORTS_DIR
    fig_dir = reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    models = _load_models(models_dir)
    df = load_clean_data(data_path)
    X, y = split_xy(df)
    _, X_test, _, y_test = make_train_test_split(X, y)   # same split as training

    probas, rows = {}, {}
    for name, pipe in models.items():
        probas[name] = pipe.predict_proba(X_test)[:, 1]
        rows[name] = compute_metrics(y_test, probas[name])
    comparison = pd.DataFrame(rows).T
    comparison.index.name = "model"
    final = select_final_model(comparison)
    comparison["selected"] = comparison.index == final
    out = comparison.copy()
    for c in ["accuracy", "precision", "recall", "f1", "f2", "roc_auc"]:
        out[c] = out[c].astype(float).round(4)
    for c in ["tn", "fp", "fn", "tp"]:
        out[c] = out[c].astype(int)
    out.to_csv(reports_dir / "model_comparison.csv")
    (models_dir / FINAL_MODEL_FILE).write_text(json.dumps({
        "final_model": final,
        "selection_rule": "highest F2 on the held-out test set (ROC-AUC breaks ties)",
    }, indent=2))

    # ---- plots -------------------------------------------------------------
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4))
    for ax, (name, p) in zip(np.atleast_1d(axes), probas.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test, (p >= DEFAULT_THRESHOLD).astype(int), display_labels=["Stayed", "Churned"],
            cmap="Blues", colorbar=False, ax=ax)
        ax.set_title(name.replace("_", " ").title()); ax.grid(False)
    fig.suptitle("Confusion matrices on the test set (threshold = 0.5)")
    fig.savefig(fig_dir / "confusion_matrices.png", dpi=130, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {rows[name]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random guessing")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves (test set)")
    ax.legend(loc="lower right")
    fig.savefig(fig_dir / "roc_curves.png", dpi=130, bbox_inches="tight")

    thr = threshold_table(y_test, probas[final], np.linspace(0.05, 0.95, 19))
    thr.to_csv(reports_dir / "threshold_analysis.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(thr["threshold"], thr["precision"], label="Precision")
    ax.plot(thr["threshold"], thr["recall"], label="Recall")
    ax.plot(thr["threshold"], thr["f1"], label="F1", linestyle="--")
    ax.set(xlabel="Decision threshold", ylabel="Score", title=f"Threshold trade-off: {final}")
    ax.legend()
    fig.savefig(fig_dir / "threshold_tradeoff.png", dpi=130, bbox_inches="tight")

    _plot_importances(models, X_test, y_test, final, fig_dir, reports_dir)
    _export_for_power_bi(df, X_test, y_test, probas[final], final, reports_dir)
    plt.close("all")
    return out


def _plot_importances(models, X_test, y_test, final, fig_dir: Path, reports_dir: Path) -> None:
    def top_plot(series: pd.Series, title: str, filename: str, xlabel: str):
        s = series.sort_values().tail(15)
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.barh(s.index, s.values, color="#4C78A8")
        ax.set(title=title, xlabel=xlabel)
        fig.savefig(fig_dir / filename, dpi=130, bbox_inches="tight")

    if "random_forest" in models:
        pipe = models["random_forest"]
        names = [clean_feature_name(n) for n in pipe.named_steps["preprocess"].get_feature_names_out()]
        imp = pd.Series(pipe.named_steps["model"].feature_importances_, index=names)
        imp.rename_axis("feature").sort_values(ascending=False).to_csv(
            reports_dir / "rf_feature_importance.csv", header=["importance"])
        top_plot(imp, "Random Forest feature importance (top 15)", "rf_feature_importance.png",
                 "Mean decrease in impurity")
    if "logistic_regression" in models:
        pipe = models["logistic_regression"]
        names = [clean_feature_name(n) for n in pipe.named_steps["preprocess"].get_feature_names_out()]
        coef = pd.Series(pipe.named_steps["model"].coef_[0], index=names)
        top = coef.reindex(coef.abs().sort_values().tail(15).index).sort_values()
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.barh(top.index, top.values, color=["#E45756" if v > 0 else "#4C78A8" for v in top.values])
        ax.set(title="Logistic Regression coefficients (top 15 by size)",
               xlabel="Coefficient (red = higher churn odds, blue = lower)")
        fig.savefig(fig_dir / "lr_coefficients.png", dpi=130, bbox_inches="tight")

    # Permutation importance on held-out data: how much does ROC-AUC drop when
    # one original column is shuffled? Works on raw columns, so it is easy to read.
    perm = permutation_importance(models[final], X_test, y_test, scoring="roc_auc",
                                  n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
    pi = pd.DataFrame({"feature": X_test.columns, "mean_drop_in_auc": perm.importances_mean,
                       "std": perm.importances_std}).sort_values("mean_drop_in_auc", ascending=False)
    pi.to_csv(reports_dir / "permutation_importance.csv", index=False)
    top_plot(pi.set_index("feature")["mean_drop_in_auc"], f"Permutation importance on test set ({final})",
             "permutation_importance.png", "Drop in ROC-AUC when the feature is shuffled")


def _export_for_power_bi(df, X_test, y_test, proba, final, reports_dir: Path) -> None:
    """powerbi_customers.csv: all cleaned customers + helper columns.
    test_set_predictions.csv: model output for held-out customers only
    (predictions for training rows would look better than they really are)."""
    pbi = add_tenure_group(df)
    pbi["ChurnLabel"] = pbi[TARGET_COLUMN].map({0: "Active", 1: "Churned"})
    pbi.to_csv(reports_dir / "powerbi_customers.csv", index=False)
    pred = pd.DataFrame({
        ID_COLUMN: df.loc[X_test.index, ID_COLUMN].values,
        "MonthlyCharges": X_test["MonthlyCharges"].values,
        "ActualChurn": y_test.values,
        "ChurnProbability": np.round(proba, 4),
        "PredictedChurn": (proba >= DEFAULT_THRESHOLD).astype(int),
        "Model": final,
    })
    pred.to_csv(reports_dir / "test_set_predictions.csv", index=False)


if __name__ == "__main__":
    print(evaluate_all())

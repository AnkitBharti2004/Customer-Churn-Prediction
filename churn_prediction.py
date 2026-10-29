"""Single entry point for the whole workflow.

    python churn_prediction.py eda
    python churn_prediction.py train [--xgboost]
    python churn_prediction.py evaluate
    python churn_prediction.py predict [--json customer.json]
    python churn_prediction.py all
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # works from any folder

import matplotlib
matplotlib.use("Agg")  # save figures without opening windows

from src.data_preprocessing import DATA_FILE, DataValidationError, load_raw_data, clean_data
from src.eda import dataset_overview, run_full_eda
from src.evaluate_model import evaluate_all
from src.predict_churn import EXAMPLE_CUSTOMER, load_model, predict_customer
from src.train_model import train_models
import json


def run_eda() -> None:
    raw = load_raw_data()
    clean = clean_data(raw)
    info = dataset_overview(raw, clean)
    print(f"Raw shape: {info['raw_shape']}  Clean shape: {info['clean_shape']}")
    print(f"Blank/invalid TotalCharges: {info['blank_or_invalid_TotalCharges']}  "
          f"Duplicate rows: {info['duplicate_rows_raw']}")
    for name, table in run_full_eda(clean).items():
        print(f"\nChurn rate by {name}:\n{table}")
    print("\nCharts saved in reports/figures/")


def main() -> int:
    parser = argparse.ArgumentParser(description="Customer churn prediction workflow")
    parser.add_argument("step", choices=["eda", "train", "evaluate", "predict", "all"])
    parser.add_argument("--xgboost", action="store_true", help="also train XGBoost (needs pip install xgboost)")
    parser.add_argument("--json", type=Path, help="customer JSON file for the predict step")
    args = parser.parse_args()
    try:
        if args.step in ("eda", "all"):
            run_eda()
        if args.step in ("train", "all"):
            train_models(include_xgboost=args.xgboost)
        if args.step in ("evaluate", "all"):
            print(evaluate_all().to_string())
        if args.step in ("predict", "all"):
            record = json.loads(args.json.read_text()) if args.json else EXAMPLE_CUSTOMER
            print(json.dumps(predict_customer(record, load_model()), indent=2))
    except (FileNotFoundError, DataValidationError, ValueError, ImportError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

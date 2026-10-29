# Customer Churn Prediction (Telecom)

A beginner-friendly machine learning project that predicts whether a telecom customer is likely to leave (churn), explains which factors are associated with churn, and turns the findings into business suggestions.

## Problem statement
Keeping an existing customer is usually cheaper than winning a new one. Given a customer's demographics, services, contract, tenure and billing details, can we estimate the chance that they churn, so the company can review the riskiest customers first?

## Features
- Data validation and cleaning (clear errors for a missing or malformed file)
- EDA with saved charts and plain-English findings
- Leak-free scikit-learn pipeline (`ColumnTransformer`: impute, scale, one-hot encode)
- Logistic Regression and Random Forest (XGBoost optional), tuned with cross-validation on training data only
- Evaluation: accuracy, precision, recall, F1, F2, ROC-AUC, confusion matrix, ROC curve, threshold analysis
- Feature importance: Random Forest importance, permutation importance, Logistic Regression coefficients
- Reusable prediction module with input validation, probability and risk group
- Streamlit dashboard (optional) and a separate Power BI guide
- 18 automated tests

## Technologies
Python 3, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Joblib, Streamlit, Pytest, Power BI Desktop. No paid services or APIs.

## Folder structure
```
Customer-Churn-Prediction/
├── data/                 # put the dataset CSV here
├── notebooks/            # 01_eda.ipynb, 02_model_training.ipynb
├── src/                  # data_preprocessing, eda, train_model, evaluate_model, predict_churn
├── models/               # saved pipelines (*.joblib) + training metadata
├── reports/              # model_comparison.csv, other tables, figures/
├── dashboard/            # Streamlit app
├── powerbi/              # Power BI guide + DAX
├── tests/                # test_preprocessing.py
├── churn_prediction.py   # one command for the whole workflow
├── requirements.txt
└── .gitignore
```
`src/eda.py` is an extra module (not in the original outline) so EDA code is reused by the notebook, script and dashboard instead of duplicated.

## Dataset
IBM Telco Customer Churn (7,043 customers, 21 columns).
1. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Place it in `data/` so the path is `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

The results below were produced with IBM's own copy of the same dataset (`Telco-Customer-Churn.csv` from the `IBM/telco-customer-churn-on-icp4d` GitHub repository, renamed). It has the same 21 columns and 7,043 rows as the Kaggle file; if your copy differs, your numbers may differ slightly.

## Installation (Windows PowerShell)
```powershell
cd Customer-Churn-Prediction
python -m venv venv
.\venv\Scripts\Activate.ps1
# If activation is blocked:  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## How to run
```powershell
python churn_prediction.py eda          # EDA tables + charts -> reports/figures
python churn_prediction.py train        # tune + save models -> models/
python churn_prediction.py evaluate     # metrics, plots, importance -> reports/
python churn_prediction.py predict      # predict the built-in example customer
python churn_prediction.py predict --json my_customer.json
python churn_prediction.py all          # everything above in order

python -m pytest tests                  # run the tests
streamlit run dashboard/app.py          # dashboard (after 'all')
jupyter notebook notebooks              # needs: pip install jupyter
python churn_prediction.py train --xgboost   # optional, needs: pip install xgboost
```
Paths are built from the file locations, so commands work from any folder as long as you give the script path.

## Evaluation methodology
- 80/20 stratified train/test split with `random_state=42`; every model and the evaluation step use the same split.
- Preprocessing sits inside each pipeline, so imputation, scaling and encoding statistics are learned from training data only (also inside each CV fold). `customerID` and `Churn` are never inputs.
- Class imbalance (about 27% churn) is handled with `class_weight="balanced"` rather than copying or inventing rows.
- Hyperparameters: grid search, 5-fold stratified CV, scored by ROC-AUC, on the training set only.
- Final model: highest **F2** on the test set (recall counts twice as much as precision, because missing a churner usually costs more than an unnecessary retention offer); ROC-AUC breaks ties. Accuracy is not used. Choosing on the test set is a limitation; a separate validation set would be stricter.

Key ideas: a **false positive** is a customer flagged who would have stayed; a **false negative** is a churner we missed. **Recall** is the share of churners caught. **ROC-AUC** measures ranking quality across all thresholds, so it complements accuracy. Raising the threshold increases precision and lowers recall.

## Results (actual, from `reports/model_comparison.csv`; test set = 1,409 customers)
| Model | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC | TN / FP / FN / TP |
|---|---|---|---|---|---|---|---|
| Logistic Regression (selected) | 0.7388 | 0.5052 | 0.7834 | 0.6143 | 0.7057 | 0.8406 | 748 / 287 / 81 / 293 |
| Random Forest | 0.7594 | 0.5332 | 0.7513 | 0.6238 | 0.6945 | 0.8422 | 789 / 246 / 93 / 281 |

Best CV ROC-AUC on training data: Logistic Regression 0.8462 (C=10), Random Forest 0.8473 (max_depth=10, min_samples_leaf=5, n_estimators=200).

Reading the table honestly: the two models are close. Logistic Regression was selected because it catches more churners (recall 0.78 vs 0.75); Random Forest has slightly higher accuracy, precision and ROC-AUC. With a single split, differences this small should not be over-interpreted.

Threshold effect for the selected model (`reports/threshold_analysis.csv`): at 0.3 recall is 0.93 and precision 0.43; at 0.5 recall 0.78 and precision 0.51; at 0.7 recall 0.60 and precision 0.60.

### What the data shows (associations, not causes)
- Churn rate: month-to-month contracts 42.7%, one-year 11.3%, two-year 2.8%.
- Churn falls with tenure: 47.4% in months 0-12 vs 9.5% after 48 months.
- Electronic-check payers 45.3% vs 15-19% for other methods; fiber-optic 41.9%, DSL 19.0%, no internet 7.4%.
- Most influential features on held-out data (permutation importance): tenure, InternetService, MonthlyCharges, Contract.

### Suggested business actions
- Review retention offers for new month-to-month customers in their first year.
- Investigate why fiber-optic and electronic-check customers churn more (price, service quality, billing friction).
- Pilot retention offers on a small group first and measure the result; the model shows associations, so changing a feature may not prevent churn.

## Screenshots
Generated charts are in `reports/figures/`. After running the dashboard, add your own screenshots here, for example:
`![Dashboard](reports/screenshots/dashboard.png)` (create the folder yourself; none are included).

## Power BI
See `powerbi/POWERBI_GUIDE.md` for import steps, DAX measures and chart recommendations. Run `evaluate` first to create `reports/powerbi_customers.csv`.

## Testing status
`python -m pytest tests` ran with 18 tests passing in the build environment. The Streamlit pages were run headlessly with Streamlit's test runner without errors; the dashboard was not viewed in a real browser. The Power BI steps are written instructions and were not run (Power BI Desktop is Windows-only). The optional XGBoost training path ran once in a temporary folder; XGBoost evaluation and plots were not run.

## Limitations and future work
- One historical dataset; may not generalise to other companies or periods.
- Probabilities are not calibrated (class weights); add calibration if probabilities must be exact.
- Model chosen on the test set; use a validation set or nested CV.
- No cost-based threshold; add real retention costs and customer lifetime value.
- Future: SHAP explanations, XGBoost comparison on evaluation, MLflow tracking, automatic retraining.

"""Streamlit dashboard. Run from the project root:  streamlit run dashboard/app.py
Models are loaded once (cached); they are never retrained here."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_preprocessing import (  # noqa: E402
    DATA_FILE, FIGURES_DIR, REPORTS_DIR, VALID_CATEGORIES, add_tenure_group, load_clean_data,
)
from src.eda import churn_rate_table  # noqa: E402
from src.predict_churn import EXAMPLE_CUSTOMER, LIMITATIONS, load_model, predict_customer  # noqa: E402
from src.train_model import list_trained_models  # noqa: E402

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_clean_data()


@st.cache_resource
def get_model(name: str):
    return load_model(name)


def show_figure(filename: str, caption: str = "") -> None:
    path = FIGURES_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.info(f"'{filename}' not found. Run: python churn_prediction.py all")


st.title("Customer Churn Prediction")
st.caption("Telecom churn analysis and prediction. Predictions are estimates, not certainties. "
           "The data is the public IBM Telco sample; no real customer data is shown.")

if not DATA_FILE.exists():
    st.error(f"Dataset not found at {DATA_FILE}. See the README for the download link.")
    st.stop()

df = get_data()
page = st.sidebar.radio("Page", ["Dataset overview", "EDA", "Model performance",
                                 "Feature importance", "Predict a customer"])

if page == "Dataset overview":
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Churned", f"{int(df['Churn'].sum()):,}")
    c3.metric("Churn rate", f"{df['Churn'].mean():.1%}")
    show_figure("churn_distribution.png", "Churn vs non-churn")
    st.dataframe(df.drop(columns=["customerID"]).head(20))

elif page == "EDA":
    tdf = add_tenure_group(df)
    for col, fig in [("Contract", "churn_by_contract.png"), ("TenureGroup", "churn_by_tenure_group.png"),
                     ("PaymentMethod", "churn_by_payment_method.png"),
                     ("InternetService", "churn_by_internet_service.png")]:
        left, right = st.columns(2)
        with left:
            show_figure(fig)
        with right:
            st.dataframe(churn_rate_table(tdf, col))
    show_figure("tenure_vs_churn.png")
    show_figure("monthly_charges_vs_churn.png")
    show_figure("correlation_heatmap.png")
    st.caption("These are associations in historical data. They do not prove what causes churn.")

elif page == "Model performance":
    path = REPORTS_DIR / "model_comparison.csv"
    if path.exists():
        st.dataframe(pd.read_csv(path).set_index("model"))
        st.caption("The final model is chosen by F2 (recall-weighted), not accuracy alone.")
    else:
        st.info("Run: python churn_prediction.py all")
    show_figure("confusion_matrices.png")
    show_figure("roc_curves.png")
    show_figure("threshold_tradeoff.png")

elif page == "Feature importance":
    show_figure("rf_feature_importance.png")
    show_figure("permutation_importance.png")
    show_figure("lr_coefficients.png")
    st.caption("Important features are associated with churn; changing them may not change churn.")

else:
    names = list_trained_models()
    if not names:
        st.warning("No trained models found. Run: python churn_prediction.py all")
        st.stop()
    choice = st.selectbox("Model", names)
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.5, 0.05)
    with st.form("customer_form"):
        cols = st.columns(3)
        rec = {}
        rec["tenure"] = cols[0].number_input("Tenure (months)", 0, 120, int(EXAMPLE_CUSTOMER["tenure"]))
        rec["MonthlyCharges"] = cols[1].number_input("Monthly charges", 0.0, 500.0, float(EXAMPLE_CUSTOMER["MonthlyCharges"]))
        rec["TotalCharges"] = cols[2].number_input("Total charges", 0.0, 20000.0, float(EXAMPLE_CUSTOMER["TotalCharges"]))
        for i, (col, options) in enumerate(VALID_CATEGORIES.items()):
            default = options.index(EXAMPLE_CUSTOMER[col])
            rec[col] = cols[i % 3].selectbox(col, options, index=default)
        submitted = st.form_submit_button("Predict")
    if submitted:
        try:
            result = predict_customer(rec, get_model(choice), threshold=threshold)
            st.metric("Estimated churn probability", f"{result['churn_probability']:.1%}")
            st.write(f"**Predicted class:** {result['predicted_label']}  |  **Risk group:** {result['risk_group']}")
            st.progress(min(max(result["churn_probability"], 0.0), 1.0))
            st.caption(LIMITATIONS)
        except ValueError as exc:
            st.error(str(exc))

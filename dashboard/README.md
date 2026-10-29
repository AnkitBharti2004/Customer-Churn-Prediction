# Streamlit dashboard

From the project root (after `python churn_prediction.py all`):

```powershell
streamlit run dashboard/app.py
```

Pages: dataset overview, EDA, model performance, feature importance, and a customer prediction form.
Models are loaded once with `st.cache_resource` and are never retrained in the dashboard.
Predictions are estimates, not certainties. This dashboard is separate from the Power BI dashboard (see `powerbi/POWERBI_GUIDE.md`).

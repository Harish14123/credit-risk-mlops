"""
Streamlit Web Dashboard for Credit Risk / Loan Default Prediction
Connects to the FastAPI backend (or directly loads champion model) to provide an interactive UI.
"""
import streamlit as st
import requests
import json
import pandas as pd
from pathlib import Path

# Page Config
st.set_page_config(
    page_title="Credit Risk & Default Prediction",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL = "http://127.0.0.1:8000"

st.title("💳 Credit Risk / Loan Default Prediction System")
st.markdown("Production-grade MLOps loan risk scoring system powered by FastAPI & scikit-learn.")

# Check API status
api_online = False
health_data = {}
try:
    res = requests.get(f"{API_BASE_URL}/health", timeout=2)
    if res.status_code == 200:
        api_online = True
        health_data = res.json()
except Exception:
    api_online = False

# Sidebar status
st.sidebar.header("System Status")
if api_online:
    st.sidebar.success(f"🟢 API Online\nModel: `{health_data.get('model_name')}` ({health_data.get('model_version')})")
else:
    st.sidebar.warning("🟡 API Offline (Falling back to direct model loader)")

st.sidebar.markdown("---")
st.sidebar.header("📋 Applicant Details")

# Input Form in Sidebar
income = st.sidebar.number_input("Annual Income ($)", min_value=1000, max_value=1000000, value=65000, step=5000)
age = st.sidebar.slider("Applicant Age", min_value=18, max_value=100, value=32)
loan_amount = st.sidebar.number_input("Requested Loan Amount ($)", min_value=500, max_value=500000, value=25000, step=1000)
employment_years = st.sidebar.slider("Employment (Years)", min_value=0, max_value=50, value=6)
credit_score = st.sidebar.slider("Credit Score (FICO)", min_value=300, max_value=850, value=720)
existing_debt = st.sidebar.number_input("Existing Total Debt ($)", min_value=0, max_value=500000, value=8000, step=1000)
num_credit_lines = st.sidebar.slider("Number of Open Credit Lines", min_value=0, max_value=30, value=5)
num_delinquencies_2yr = st.sidebar.selectbox("Delinquencies (Past 2 Years)", options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=0)
loan_term_months = st.sidebar.selectbox("Loan Term (Months)", options=[12, 24, 36, 48, 60, 84], index=4)
interest_rate = st.sidebar.slider("Interest Rate (%)", min_value=1.0, max_value=36.0, value=9.5, step=0.1)

home_ownership = st.sidebar.selectbox("Home Ownership", options=["MORTGAGE", "RENT", "OWN", "OTHER"])
loan_purpose = st.sidebar.selectbox(
    "Loan Purpose",
    options=[
        "home_improvement", "debt_consolidation", "credit_card",
        "major_purchase", "medical", "small_business", "car", "other"
    ]
)
employment_type = st.sidebar.selectbox("Employment Type", options=["salaried", "self_employed", "unemployed", "retired"])

# Tabs layout
tab1, tab2, tab3 = st.tabs(["🚀 Predict Risk", "📊 Model Comparison & Registry", "⚡ API Diagnostics"])

with tab1:
    st.subheader("Loan Application Assessment")
    
    applicant_payload = {
        "income": income,
        "age": age,
        "loan_amount": loan_amount,
        "employment_years": employment_years,
        "credit_score": credit_score,
        "existing_debt": existing_debt,
        "num_credit_lines": num_credit_lines,
        "num_delinquencies_2yr": num_delinquencies_2yr,
        "loan_term_months": loan_term_months,
        "interest_rate": interest_rate,
        "home_ownership": home_ownership,
        "loan_purpose": loan_purpose,
        "employment_type": employment_type,
    }

    if st.button("Evaluate Credit Risk", type="primary", use_container_width=True):
        try:
            if api_online:
                response = requests.post(f"{API_BASE_URL}/predict", json=applicant_payload)
                result = response.json()
            else:
                from api.model_loader import model_service
                if model_service.pipeline is None:
                    model_service.load()
                proba = model_service.predict_one(applicant_payload)
                result = {
                    "default_probability": round(proba, 4),
                    "prediction": model_service.bucket(proba),
                    "model_version": model_service.version,
                    "model_name": model_service.model_name,
                    "threshold": model_service.threshold,
                }
            
            prob = result["default_probability"]
            risk_tier = result["prediction"]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Default Probability", f"{prob * 100:.2f}%")
            with col2:
                if risk_tier == "low_risk":
                    st.success("Risk Status: LOW RISK 🟢")
                elif risk_tier == "medium_risk":
                    st.warning("Risk Status: MEDIUM RISK 🟡")
                else:
                    st.error("Risk Status: HIGH RISK 🔴")
            with col3:
                st.metric("Served Model", f"{result['model_name']} ({result['model_version']})")
                
            st.markdown("### Risk Analysis Summary")
            st.progress(prob)
            
            st.json(result)
        except Exception as e:
            st.error(f"Error making prediction: {e}")

with tab2:
    st.subheader("Model Comparison Report")
    comparison_file = Path(__file__).resolve().parent / "artifacts" / "model_comparison.json"
    if comparison_file.exists():
        with open(comparison_file, "r") as f:
            data = json.load(f)
        df_comp = pd.DataFrame(data)
        st.dataframe(
            df_comp[["model_name", "version", "roc_auc", "pr_auc", "f1", "precision", "recall", "brier_score"]],
            use_container_width=True,
        )
        st.caption("Champion model is selected based on PR-AUC to account for default class imbalance (~17%).")
    else:
        st.info("Run `python -m src.train` to generate model comparison metrics.")

with tab3:
    st.subheader("API Live Endpoint Info")
    if api_online:
        info_res = requests.get(f"{API_BASE_URL}/model/info")
        if info_res.status_code == 200:
            st.json(info_res.json())
        else:
            st.error("Failed to fetch model info from API.")
    else:
        st.warning("FastAPI service is not reachable at http://127.0.0.1:8000.")

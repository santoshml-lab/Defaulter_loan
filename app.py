
import streamlit as st
import numpy as np
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Loan Risk AI", layout="centered")

# ---------------- LOAD MODEL ----------------
model = joblib.load("loan_model.pkl")

# ---------------- HEADER ----------------
st.title("🏦 Loan Defaulter Prediction System")
st.caption("AI Powered Credit Risk Analyzer")

st.markdown("---")

# ---------------- INPUT ----------------
st.subheader("📋 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    income = st.number_input("💰 Income", min_value=0)

with col2:
    loan_amount = st.number_input("🏦 Loan Amount", min_value=0)

employment = st.selectbox("👔 Employment Status", ["Employed", "Unemployed"])

st.markdown("---")

# ---------------- PREDICT ----------------
if st.button("🚀 Predict Risk"):

    emp = 1 if employment == "Employed" else 0

    input_data = np.array([[income, loan_amount, emp]])

    pred = model.predict(input_data)[0]

    try:
        prob = model.predict_proba(input_data)[0][1]
    except:
        prob = 0.5

    # ---------------- RISK METER ----------------
    st.subheader("📊 Risk Meter")

    risk_percent = int(prob * 100)

    st.progress(risk_percent)

    st.write(f"Risk Score: **{risk_percent}%**")

    # ---------------- RESULT ----------------
    if pred == 1:
        st.error("🔴 HIGH RISK CUSTOMER (Defaulter Likely)")
    else:
        st.success("🟢 LOW RISK CUSTOMER (Safe)")

    # ---------------- EXTRA INSIGHT ----------------
    if risk_percent < 30:
        st.info("Low risk profile — Good financial stability")
    elif risk_percent < 70:
        st.warning("Medium risk — Monitor closely")
    else:
        st.error("High risk — Loan not recommended")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit + ML Model")

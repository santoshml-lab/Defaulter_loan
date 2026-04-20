import streamlit as st
import numpy as np
import joblib

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Loan Risk AI",
    page_icon="🏦",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("loan_model.pkl")

# ---------------- HEADER ----------------
st.title("🏦 Smart Loan Decision Engine")
st.caption("AI-powered Credit Risk Analysis System")

st.markdown("---")

# ---------------- INPUT ----------------
st.subheader("📋 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    income = st.number_input("💰 Monthly Income", min_value=0)

with col2:
    loan_amount = st.number_input("🏦 Loan Amount", min_value=0)

employment = st.selectbox("👔 Employment Status", ["Employed", "Unemployed"])

st.markdown("---")

# ---------------- PREDICT ----------------
if st.button("🚀 Evaluate Application"):

    emp = 1 if employment == "Employed" else 0

    input_data = np.array([[income, loan_amount, emp]])

    pred = model.predict(input_data)[0]

    try:
        prob = model.predict_proba(input_data)[0][1]
    except:
        prob = 0.5

    risk = prob * 100

    # ---------------- RESULT SECTION ----------------
    st.subheader("📊 Decision Dashboard")

    st.progress(int(risk))
    st.metric("Risk Score", f"{risk:.2f}%")

    # ---------------- DECISION ENGINE ----------------
    if risk < 30:
        st.success("🟢 APPROVED — Low Risk Customer")
        st.write("✔ Loan can be safely approved.")

    elif risk < 70:
        st.warning("🟡 MANUAL REVIEW REQUIRED")
        st.write("⚠ Needs human verification before approval.")

    else:
        st.error("🔴 REJECTED — High Risk Customer")
        st.write("❌ High probability of default detected.")

    # ---------------- AI INSIGHTS ----------------
    st.markdown("---")
    st.subheader("🧠 AI Insight Engine")

    if loan_amount > income * 2:
        st.warning("⚠ Loan amount is too high compared to income.")

    if employment == "Unemployed":
        st.warning("⚠ Unemployment increases default risk.")

    if risk < 30:
        st.info("✔ Strong financial stability detected.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with ❤️ using ML + Streamlit | Final Production Version")

import streamlit as st
import numpy as np
import joblib

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Loan Decision Engine", layout="centered")

# ---------------- LOAD MODEL ----------------
model = joblib.load("loan_model.pkl")

# ---------------- UI ----------------
st.title("🏦 Smart Loan Decision Engine")
st.caption("AI-powered Credit Approval System")

st.markdown("---")

# ---------------- INPUT ----------------
col1, col2 = st.columns(2)

with col1:
    income = st.number_input("💰 Income", min_value=0)

with col2:
    loan_amount = st.number_input("🏦 Loan Amount", min_value=0)

employment = st.selectbox("👔 Employment Status", ["Employed", "Unemployed"])

st.markdown("---")

# ---------------- BUTTON ----------------
if st.button("🚀 Evaluate Application"):

    emp = 1 if employment == "Employed" else 0

    input_data = np.array([[income, loan_amount, emp]])

    pred = model.predict(input_data)[0]

    try:
        prob = model.predict_proba(input_data)[0][1]
    except:
        prob = 0.5

    risk = prob * 100

    # ---------------- DECISION LOGIC ----------------
    st.subheader("📊 Decision Result")

    st.progress(int(risk))

    st.write(f"🔢 Risk Score: **{risk:.2f}%**")

    # ---------------- SMART DECISION ENGINE ----------------
    if risk < 30:
        st.success("🟢 APPROVED — Low Risk Customer")
        st.info("Loan is SAFE to approve. Strong profile.")

    elif 30 <= risk < 70:
        st.warning("🟡 MANUAL REVIEW REQUIRED")
        st.info("Send to human verification team.")

    else:
        st.error("🔴 REJECTED — High Risk Customer")
        st.info("Loan not recommended due to high default probability.")

    # ---------------- EXTRA INSIGHT ----------------
    st.markdown("---")
    st.subheader("🧠 AI Insight")

    if loan_amount > income * 2:
        st.write("⚠ Loan amount is very high compared to income.")

    if employment == "Unemployed":
        st.write("⚠ Employment status increases risk factor.")

    if risk < 30:
        st.write("✔ Strong financial stability detected.")

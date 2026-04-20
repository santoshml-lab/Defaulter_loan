
import streamlit as st
import requests

st.set_page_config(page_title="Loan Risk AI", layout="centered")

# ---------------- HEADER ----------------
st.title("🏦 Loan Defaulter Prediction System")
st.caption("AI-powered Credit Risk Analyzer")

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

# ---------------- BUTTON ----------------
if st.button("🚀 Predict Risk"):

    emp = 1 if employment == "Employed" else 0

    input_data = {
        "income": income,
        "loan_amount": loan_amount,
        "employment_status": emp
    }

    try:
        res = requests.post("http://127.0.0.1:8000/predict", json=input_data)
        output = res.json()

        prob = output["probability"]
        pred = output["prediction"]

        st.markdown("## 📊 Result Dashboard")

        st.progress(int(prob * 100))
        st.metric("Risk Probability", f"{prob*100:.2f}%")

        if pred == 1:
            st.error("🔴 HIGH RISK CUSTOMER")
        else:
            st.success("🟢 LOW RISK CUSTOMER")

    except:
        st.warning("⚠ API not running. Start FastAPI first.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit + ML")

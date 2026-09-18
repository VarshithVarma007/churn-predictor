import streamlit as st
import pandas as pd
import joblib
import sqlite3

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("telco.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna().reset_index(drop=True)
    df['Churn_Flag'] = (df['Churn'] == 'Yes').astype(int)
    return df

bundle = load_model()
model, encoders, features = bundle["model"], bundle["encoders"], bundle["features"]
df = load_data()

st.title("📉 Customer Churn Predictor")
st.caption("Logistic Regression model trained on the Telco Customer Churn dataset (7,032 cleaned records)")

tab1, tab2 = st.tabs(["Predict Churn Risk", "SQL Insights"])

with tab1:
    st.subheader("Enter customer details")
    col1, col2 = st.columns(2)
    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
        total = st.number_input("Total Charges ($)", 0.0, 10000.0, 840.0)
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    with col2:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox("Payment Method", df['PaymentMethod'].unique())
        internet = st.selectbox("Internet Service", df['InternetService'].unique())
        security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    if st.button("Predict churn risk", type="primary"):
        row = {
            "tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": total,
            "Contract": contract, "PaymentMethod": payment, "InternetService": internet,
            "OnlineSecurity": security, "TechSupport": support,
            "PaperlessBilling": paperless, "SeniorCitizen": 1 if senior == "Yes" else 0
        }
        X = pd.DataFrame([row])[features]
        for col, le in encoders.items():
            X[col] = le.transform(X[col].astype(str))
        proba = model.predict_proba(X)[0][1]

        st.metric("Churn probability", f"{proba*100:.1f}%")
        if proba > 0.5:
            st.error("⚠️ High churn risk")
        else:
            st.success("✅ Low churn risk")

        coefs = pd.Series(model.coef_[0], index=features).sort_values(key=abs, ascending=False)
        st.write("**Top risk drivers (model coefficients):**")
        st.dataframe(coefs.head(3))

with tab2:
    st.subheader("SQL-derived churn insights")
    conn = sqlite3.connect(":memory:")
    df.to_sql("customers", conn, index=False)

    st.write("**Churn rate by contract type**")
    q1 = "SELECT Contract, COUNT(*) as n, ROUND(AVG(Churn_Flag)*100,2) as churn_rate_pct FROM customers GROUP BY Contract ORDER BY churn_rate_pct DESC"
    st.dataframe(pd.read_sql(q1, conn))

    st.write("**Churn rate by tenure bucket (window function: RANK)**")
    q2 = """
    SELECT CASE WHEN tenure < 12 THEN '0-12mo' WHEN tenure < 36 THEN '12-36mo' ELSE '36mo+' END as tenure_bucket,
           COUNT(*) as n, ROUND(AVG(Churn_Flag)*100,2) as churn_rate_pct,
           RANK() OVER (ORDER BY AVG(Churn_Flag) DESC) as risk_rank
    FROM customers GROUP BY tenure_bucket
    """
    st.dataframe(pd.read_sql(q2, conn))

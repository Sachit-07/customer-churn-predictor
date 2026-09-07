import streamlit as st
import joblib

st.title("📉 Customer Churn Predictor")

model = joblib.load("app/model/churn_model.pkl")
scaler = joblib.load("app/model/scaler.pkl")
feature_columns = joblib.load("app/model/feature_columns.pkl")
numerical_columns = joblib.load("app/model/numerical_columns.pkl")

st.write("Model loaded successfully!")
st.write(f"Expecting {len(feature_columns)} features")

st.header("Enter Customer Details")

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=1200.0)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])


col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    paperless = st.selectbox("Paperless Billing", ["No", "Yes"])

with col2:
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

total_charges = st.number_input("Total Charges", min_value=0.0, value=float(tenure) * monthly_charges)

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.preprocess import build_feature_vector

if st.button("Predict Churn"):
    raw_input = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    features = build_feature_vector(raw_input, feature_columns, scaler, numerical_columns)
    churn_proba = model.predict_proba(features)[0][1]

    if churn_proba >= 0.5:
        st.error(f"⚠️ Likely to Churn — Confidence: {churn_proba:.0%}")
    else:
        st.success(f"✅ Likely to Stay — Confidence: {1 - churn_proba:.0%}")
import streamlit as st
import pandas as pd
import numpy as np
from os import path
import joblib

# Title
st.title("Customer Churn Prediction")

st.write("This app predicts whether a customer will churn or stay, based on their details.")

# Load the trained model
model_path = path.join("Model", "customer_churn_pipeline.pkl")   # Ensure you save the model here
churn_predictor = joblib.load(model_path)


# User Inputs
st.subheader("Enter Customer Details:")

gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.radio("Senior Citizen", ["Yes", "No"])
Partner = st.radio("Partner", ["Yes", "No"])
Dependents = st.radio("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (Months)", min_value=0, max_value=100, step=1)
PhoneService = st.radio("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.radio("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
])
MonthlyCharges = st.slider("Monthly Charges($)", min_value=0, max_value=150, step=1, value=50)
TotalCharges = st.slider("Total Charges ($)", min_value=0, max_value=10000, step=50, value=1000)

# Convert SeniorCitizen Yes/No → 1/0
SeniorCitizen_val = 1 \
    if SeniorCitizen == "Yes" \
    else 0

# Prepare DataFrame for Prediction
user_input = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges
}])

st.write("Input Data Preview")
st.write(user_input)

# Prediction
if st.button("🔮 Predict Churn"):
    prediction = churn_predictor.predict(user_input)[0]
    if prediction == 1:
        st.error("The customer is likely to Churn")
    else:
        st.success("The customer is likely to Stay")
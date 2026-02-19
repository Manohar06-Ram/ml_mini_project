import streamlit as st
import pandas as pd
import joblib

model = joblib.load("xgb_trees_credit_model.pkl")
encoders = {col : joblib.load(f"{col}_encoder.pkl") for col in ["Sex", "Housing", "Saving accounts", "Checking_accounts"]}

st.title("Credit Risk Prediction")
st.write("Enter the details of the applicant to predict the credit risk.")

age = st.number_input("Age", min_value=18, max_value=80, value=30)
sex = st.selectbox("Sex", ["male", "female"])
job = st.number_input("Job(0-3)", min_value=0, max_value=3, value = 1)
housing = st.selectbox("Housing", ["own", "rent", "free"])
saving_accounts = st.selectbox("Saving accounts", ["little", "moderate", "rich", "quite rich"])
checking_accounts = st.selectbox("Checking accounts", ["little", "moderate", "rich"])
credit_amount = st.number_input("Credit amount", min_value=0, value=100)
duration = st.number_input("Duration (months)", min_value=1, value=12)

input_df = pd.DataFrame({
    "Age": [age],
    "Sex": [encoders["Sex"].transform([sex])[0]],
    "Job": [job],
    "Housing": [encoders["Housing"].transform([housing])[0]],
    "Saving accounts": [encoders["Saving accounts"].transform([saving_accounts])[0]],
    "Checking_accounts": [encoders["Checking_accounts"].transform([checking_accounts])[0]],
    "Credit amount": [credit_amount],
    "Duration": [duration]
})

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("THE PREDICTED CREDIT RISK IS: **GOOD**")
    else:
        st.error("THE PREDICTED CREDIT RISK IS: **BAD**")
"""
Streamlit App: Diabetes Prediction using Logistic Regression
==============================================================
Run locally with:
    streamlit run app.py

Requires diabetes_model.pkl, scaler.pkl and feature_names.pkl
(created by logistic_regression_diabetes.py) to be in the same folder.
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="centered")

# --------------------------------------------------------------------------
# Load trained model, scaler and feature order
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("diabetes_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, scaler, feature_names

model, scaler, feature_names = load_artifacts()

st.title("🩺 Diabetes Prediction App")
st.write(
    "This app uses a **Logistic Regression** model trained on the "
    "Pima Indians Diabetes dataset to estimate the probability that a "
    "patient has diabetes, based on diagnostic measurements."
)

st.header("Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
    glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin (mu U/mL)", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)

# Build the input row in the exact column order the model was trained on
input_dict = {
    "Pregnancies": pregnancies,
    "Glucose": glucose,
    "BloodPressure": blood_pressure,
    "SkinThickness": skin_thickness,
    "Insulin": insulin,
    "BMI": bmi,
    "DiabetesPedigreeFunction": dpf,
    "Age": age,
}
input_df = pd.DataFrame([input_dict])[feature_names]

if st.button("Predict"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Result")
    if prediction == 1:
        st.error(f"⚠️ The model predicts **Diabetic** "
                  f"(probability: {probability:.2%})")
    else:
        st.success(f"✅ The model predicts **Not Diabetic** "
                    f"(probability of diabetes: {probability:.2%})")

    st.progress(min(max(probability, 0.0), 1.0))

    with st.expander("See input summary"):
        st.dataframe(input_df)

st.markdown("---")
st.caption(
    "Note: This tool is for educational purposes only and is not a "
    "substitute for professional medical diagnosis."
)

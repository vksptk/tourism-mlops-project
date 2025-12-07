
import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import joblib

st.set_page_config(page_title="Tourism Wellness Package Predictor", layout="centered")

st.title("🧳 Wellness Tourism Package Prediction")
st.write("This app predicts whether a customer is likely to purchase the Wellness Tourism Package.")

# ------------------ Load Model from Hugging Face Model Hub ------------------
MODEL_REPO = "vksptk/tourism-wellness-package-model"  # change if your model repo id is different
MODEL_FILENAME = "best_model.joblib"

@st.cache_resource
def load_model():
    # Download model file from HF model hub
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        repo_type="model"
    )
    model = joblib.load(model_path)
    return model

model = load_model()
st.success("Model loaded successfully from Hugging Face Hub!")

# ------------------ Define Input Form ------------------
st.header("Customer Details")

with st.form("input_form"):
    # Numerical inputs
    customer_id = st.number_input("CustomerID", min_value=1, step=1, value=1)
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    city_tier = st.selectbox("CityTier", options=[1, 2, 3], index=0)
    number_of_person_visiting = st.number_input("NumberOfPersonVisiting", min_value=1, max_value=10, value=2)
    preferred_property_star = st.selectbox("PreferredPropertyStar", options=[1, 2, 3, 4, 5], index=2)
    number_of_trips = st.number_input("NumberOfTrips (per year)", min_value=0, max_value=50, value=2)
    passport = st.selectbox("Passport (1 = Yes, 0 = No)", options=[1, 0], index=0)
    own_car = st.selectbox("OwnCar (1 = Yes, 0 = No)", options=[1, 0], index=1)
    number_of_children_visiting = st.number_input("NumberOfChildrenVisiting", min_value=0, max_value=10, value=0)
    monthly_income = st.number_input("MonthlyIncome", min_value=0, value=50000)
    pitch_satisfaction_score = st.selectbox("PitchSatisfactionScore", options=[1, 2, 3, 4, 5], index=3)
    number_of_followups = st.number_input("NumberOfFollowups", min_value=0, max_value=20, value=3)
    duration_of_pitch = st.number_input("DurationOfPitch (minutes)", min_value=0, max_value=300, value=30)

    # Categorical inputs
    typeof_contact = st.selectbox("TypeofContact", options=["Company Invited", "Self Inquiry"])
    occupation = st.text_input("Occupation", value="Salaried")
    gender = st.selectbox("Gender", options=["Male", "Female"])
    marital_status = st.selectbox("MaritalStatus", options=["Single", "Married", "Divorced"])
    designation = st.text_input("Designation", value="Executive")
    product_pitched = st.text_input("ProductPitched", value="Basic")

    submitted = st.form_submit_button("Predict")

if submitted:
    # ------------------ Create Input DataFrame ------------------
    # IMPORTANT: Columns must match training features exactly
    feature_order = [
        "CustomerID",
        "Age",
        "TypeofContact",
        "CityTier",
        "Occupation",
        "Gender",
        "NumberOfPersonVisiting",
        "PreferredPropertyStar",
        "MaritalStatus",
        "NumberOfTrips",
        "Passport",
        "OwnCar",
        "NumberOfChildrenVisiting",
        "Designation",
        "MonthlyIncome",
        "PitchSatisfactionScore",
        "ProductPitched",
        "NumberOfFollowups",
        "DurationOfPitch"
    ]

    input_data = {
        "CustomerID": customer_id,
        "Age": age,
        "TypeofContact": typeof_contact,
        "CityTier": city_tier,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": number_of_person_visiting,
        "PreferredPropertyStar": preferred_property_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": number_of_trips,
        "Passport": passport,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": number_of_children_visiting,
        "Designation": designation,
        "MonthlyIncome": monthly_income,
        "PitchSatisfactionScore": pitch_satisfaction_score,
        "ProductPitched": product_pitched,
        "NumberOfFollowups": number_of_followups,
        "DurationOfPitch": duration_of_pitch
    }

    # Convert to DataFrame in correct column order
    input_df = pd.DataFrame([input_data], columns=feature_order)

    st.write("### Input Data")
    st.dataframe(input_df)

    # ------------------ Make Prediction ------------------
    prediction = model.predict(input_df)[0]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0][1]
    else:
        proba = None

    st.write("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ The model predicts that the customer is LIKELY to purchase the Wellness Tourism Package.")
    else:
        st.warning("❌ The model predicts that the customer is UNLIKELY to purchase the Wellness Tourism Package.")

    if proba is not None:
        st.write(f"**Probability of Purchase:** {proba:.2%}")

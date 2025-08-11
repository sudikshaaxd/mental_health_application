
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("model.pkl")

st.set_page_config(page_title="Mental Health Risk Predictor", page_icon="🧠")

# Simple login system
st.title("🔐 Login Page")
username = st.text_input("Username")
password = st.text_input("Password", type="password")

if username == "admin" and password == "1234":
    st.success("✅ Login successful!")

    st.title("🧠 Mental Health Risk Predictor")

    age = st.slider("Age", 18, 65, 25)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    family_history = st.selectbox("Family history of mental illness?", ["Yes", "No"])
    work_interfere = st.selectbox("Does your work interfere with mental health?", ["Never", "Rarely", "Sometimes", "Often"])
    no_employees = st.selectbox("Company Size", ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
    remote_work = st.selectbox("Remote work allowed?", ["Yes", "No"])
    tech_company = st.selectbox("Is it a tech company?", ["Yes", "No"])
    benefits = st.selectbox("Mental health benefits provided?", ["Yes", "No", "Don't know"])
    care_options = st.selectbox("Access to care options?", ["Yes", "No", "Not sure"])
    wellness_program = st.selectbox("Wellness program available?", ["Yes", "No", "Don't know"])
    seek_help = st.selectbox("Ease of seeking help?", ["Yes", "No", "Don't know"])

    # Convert inputs to DataFrame
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'family_history': [family_history],
        'work_interfere': [work_interfere],
        'no_employees': [no_employees],
        'remote_work': [remote_work],
        'tech_company': [tech_company],
        'benefits': [benefits],
        'care_options': [care_options],
        'wellness_program': [wellness_program],
        'seek_help': [seek_help]
    })

    # Predict button
    if st.button("Predict Risk"):
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            st.error("🔴 At Risk for Mental Health Issues")
        else:
            st.success("🟢 Not at Risk")
else:
    st.warning("Please log in to access the predictor.")

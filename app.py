import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# --- USER CREDENTIALS ---
USER_CREDENTIALS = {
    "admin": "password123",  # Change this for security
}

# --- LOGIN FUNCTION ---
def login():
    st.title("🔑 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.logged_in = True
            st.success("✅ Login successful!")
            st.rerun()
        else:
            st.error("❌ Invalid username or password")

# --- SESSION STATE ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# --- TRAIN MODEL FUNCTION ---
@st.cache_resource
def train_model():
    # Load dataset from file in app folder
    df = pd.read_csv("survey.csv")

    # Prepare target
    df = df.dropna(subset=['treatment'])
    df['treatment'] = df['treatment'].map({'Yes': 1, 'No': 0})

    # Define features
    features = ['Age', 'Gender', 'family_history', 'work_interfere', 'no_employees',
                'remote_work', 'tech_company', 'benefits', 'care_options',
                'wellness_program', 'seek_help']
    target = 'treatment'

    X = df[features]
    y = df[target]

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=['object']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('numeric', 'passthrough', numeric_cols)
        ]
    )

    model = RandomForestClassifier(class_weight='balanced', random_state=42)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    pipeline.fit(X, y)
    return pipeline

# --- MAIN APP ---
if not st.session_state.logged_in:
    login()
else:
    # Train model inside Streamlit Cloud
    model = train_model()

    st.title("🧠 Mental Health Risk Predictor")
    st.markdown("Answer the questions below to check your mental health risk.")

    # Form inputs
    age = st.number_input("Age", min_value=0, max_value=100, value=25)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    family_history = st.selectbox("Family History of Mental Illness?", ["Yes", "No"])
    work_interfere = st.selectbox("Does work interfere with your mental health?", ["Never", "Rarely", "Sometimes", "Often"])
    no_employees = st.selectbox("Number of Employees in Company", ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
    remote_work = st.selectbox("Do you work remotely?", ["Yes", "No"])
    tech_company = st.selectbox("Is it a tech company?", ["Yes", "No"])
    benefits = st.selectbox("Does your employer provide mental health benefits?", ["Yes", "No", "Don't know"])
    care_options = st.selectbox("Are care options available?", ["Yes", "No", "Not sure"])
    wellness_program = st.selectbox("Wellness program available?", ["Yes", "No", "Don't know"])
    seek_help = st.selectbox("Is help-seeking encouraged?", ["Yes", "No", "Don't know"])

    # Prediction
    if st.button("Predict Risk"):
        input_data = [[age, gender, family_history, work_interfere, no_employees,
                       remote_work, tech_company, benefits, care_options,
                       wellness_program, seek_help]]

        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]

        risk_label = "High Risk" if prediction == 1 else "Low Risk"
        confidence = round(proba[prediction] * 100, 2)

        st.subheader(f"🩺 Prediction: {risk_label}")
        st.write(f"Confidence: **{confidence}%**")

        if prediction == 1:
            st.warning("⚠️ Consider seeking professional advice or support.")
        else:
            st.success("✅ You appear at low risk based on the information provided.")

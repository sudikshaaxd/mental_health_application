import streamlit as st
import pandas as pd
import joblib
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# =========================
# Embedded CSV as string
# =========================
csv_data = """Age,Gender,family_history,work_interfere,no_employees,remote_work,tech_company,benefits,care_options,wellness_program,seek_help,risk
...PUT_YOUR_SURVEY_DATA_HERE...
"""

# Load the embedded dataset
df = pd.read_csv(StringIO(csv_data))

# =========================
# User login credentials
# =========================
USER_CREDENTIALS = {
    "admin": "password123"
}

# =========================
# Train model on the embedded data
# =========================
@st.cache_resource
def train_model():
    X = df.drop("risk", axis=1)
    y = df["risk"]

    categorical_features = X.select_dtypes(include=["object"]).columns
    numeric_features = X.select_dtypes(exclude=["object"]).columns

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ])

    clf = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42, class_weight="balanced"))
    ])

    clf.fit(X, y)
    return clf

model = train_model()

# =========================
# Login function
# =========================
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

# =========================
# Streamlit app main
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
else:
    st.title("🧠 Mental Health Risk Predictor")

    # Collect user input
    age = st.number_input("Age", min_value=10, max_value=100, step=1)
    gender = st.selectbox("Gender", df["Gender"].unique())
    family_history = st.selectbox("Family History", df["family_history"].unique())
    work_interfere = st.selectbox("Work Interfere", df["work_interfere"].unique())
    no_employees = st.selectbox("No. of Employees", df["no_employees"].unique())
    remote_work = st.selectbox("Remote Work", df["remote_work"].unique())
    tech_company = st.selectbox("Tech Company", df["tech_company"].unique())
    benefits = st.selectbox("Benefits", df["benefits"].unique())
    care_options = st.selectbox("Care Options", df["care_options"].unique())
    wellness_program = st.selectbox("Wellness Program", df["wellness_program"].unique())
    seek_help = st.selectbox("Seek Help", df["seek_help"].unique())

    if st.button("Predict Risk"):
        input_df = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "family_history": family_history,
            "work_interfere": work_interfere,
            "no_employees": no_employees,
            "remote_work": remote_work,
            "tech_company": tech_company,
            "benefits": benefits,
            "care_options": care_options,
            "wellness_program": wellness_program,
            "seek_help": seek_help
        }])

        prediction = model.predict(input_df)[0]
        if prediction == 1:
            st.error("🔴 High Risk of Mental Health Issues")
        else:
            st.success("🟢 Low Risk of Mental Health Issues")

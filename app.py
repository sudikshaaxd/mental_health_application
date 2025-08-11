# paste this whole file as app.py (replace existing)
import streamlit as st
import pandas as pd
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib
import tempfile

st.set_page_config(page_title="Mental Health Risk Predictor", layout="centered")

# -----------------------
# Credentials (change for production)
# -----------------------
USER_CREDENTIALS = {"admin": "password123", "user": "mypassword"}

# -----------------------
# Helper: load dataset
# -----------------------
def load_dataset():
    """
    Try loading survey.csv from the app folder. If not present, show a file uploader in the UI.
    Returns a DataFrame or None.
    """
    # 1) Try file in repo / app folder
    try:
        df = pd.read_csv("survey.csv")
        st.info("Loaded survey.csv from app folder.")
        return df
    except FileNotFoundError:
        # 2) fallback: ask user to upload
        st.warning("survey.csv not found in app folder. Please upload the CSV (one-time) below.")
        uploaded = st.file_uploader("Upload survey.csv", type=["csv"])
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                st.success("Dataset uploaded successfully.")
                return df
            except Exception as e:
                st.error(f"Uploaded file could not be parsed as CSV: {e}")
                return None
        return None
    except Exception as e:
        st.error(f"Error reading survey.csv from disk: {e}")
        return None

# -----------------------
# Model training (cached)
# -----------------------
@st.cache_resource
def train_pipeline(df, features, target_col):
    # drop rows with missing target
    df = df.dropna(subset=[target_col])
    X = df[features].copy()
    y = df[target_col].copy().astype(int)

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=['object']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_cols),
            ("num", "passthrough", numeric_cols)
        ]
    )

    clf = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(class_weight="balanced", random_state=42, n_estimators=150))
    ])

    clf.fit(X, y)
    return clf

# -----------------------
# Login UI
# -----------------------
def login_area():
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

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# -----------------------
# Main app logic
# -----------------------
if not st.session_state.logged_in:
    login_area()
else:
    st.title("🧠 Mental Health Risk Predictor")
    st.markdown("The app will train a model from the provided dataset (or `survey.csv` in app folder).")

    df = load_dataset()
    if df is None:
        st.info("Upload `survey.csv` above or add it to the app folder and rerun. The app needs the dataset to train.")
        st.stop()

    # Show a small preview
    if st.checkbox("Show dataset preview (first 5 rows)"):
        st.dataframe(df.head())

    # Detect / normalize target column
    target_col = None
    if "treatment" in df.columns:
        target_col = "treatment"
        df[target_col] = df[target_col].map({"Yes": 1, "No": 0})
    elif "risk" in df.columns:
        target_col = "risk"
    else:
        # try other likely names
        for cand in ["target", "label"]:
            if cand in df.columns:
                target_col = cand
                break

    if target_col is None:
        st.error("No target column found. Expected `treatment` (Yes/No) or `risk`. Columns: " + ", ".join(df.columns))
        st.stop()

    # Required features (must exist in dataset)
    FEATURES = ['Age', 'Gender', 'family_history', 'work_interfere', 'no_employees',
                'remote_work', 'tech_company', 'benefits', 'care_options',
                'wellness_program', 'seek_help']

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        st.error("Required feature columns missing: " + ", ".join(missing))
        st.write("Columns present:", ", ".join(df.columns))
        st.stop()

    # Train model
    with st.spinner("Training model (cached) — this runs once per deployment/session..."):
        try:
            pipeline = train_pipeline(df, FEATURES, target_col)
        except Exception as e:
            st.error("Training failed: " + str(e))
            st.stop()

    st.success("Model trained and ready.")

    # Build input form (use unique values from data for dropdowns so the preprocessor sees matching categories)
    st.header("Enter input values")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, value=25)
        gender = st.selectbox("Gender", sorted(df['Gender'].dropna().unique().tolist()))
        family_history = st.selectbox("Family history", sorted(df['family_history'].dropna().unique().tolist()))
        work_interfere = st.selectbox("Work interfere", sorted(df['work_interfere'].dropna().unique().tolist()))
        no_employees = st.selectbox("No. of employees", sorted(df['no_employees'].dropna().unique().tolist()))
    with col2:
        remote_work = st.selectbox("Remote work", sorted(df['remote_work'].dropna().unique().tolist()))
        tech_company = st.selectbox("Tech company", sorted(df['tech_company'].dropna().unique().tolist()))
        benefits = st.selectbox("Benefits", sorted(df['benefits'].dropna().unique().tolist()))
        care_options = st.selectbox("Care options", sorted(df['care_options'].dropna().unique().tolist()))
        wellness_program = st.selectbox("Wellness program", sorted(df['wellness_program'].dropna().unique().tolist()))
        seek_help = st.selectbox("Seek help", sorted(df['seek_help'].dropna().unique().tolist()))

    if st.button("Predict risk"):
        input_df = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'family_history': family_history,
            'work_interfere': work_interfere,
            'no_employees': no_employees,
            'remote_work': remote_work,
            'tech_company': tech_company,
            'benefits': benefits,
            'care_options': care_options,
            'wellness_program': wellness_program,
            'seek_help': seek_help
        }])

        try:
            pred = pipeline.predict(input_df)[0]
            proba = pipeline.predict_proba(input_df)[0]
        except Exception as e:
            st.error("Prediction error: " + str(e))
            st.stop()

        label = "High Risk" if pred == 1 else "Low Risk"
        confidence = round(proba[pred] * 100, 2)
        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: **{confidence}%**")
        if pred == 1:
            st.warning("Consider seeking professional help or talking to someone you trust.")
        else:
            st.success("Model indicates low risk for the provided inputs.")

    # Optional: allow download of trained model
    if st.button("Export trained model as .pkl"):
        try:
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
                joblib.dump(pipeline, tmp.name)
                tmp.seek(0)
                data = open(tmp.name, "rb").read()
                st.download_button("Download model.pkl", data, file_name="model.pkl")
        except Exception as e:
            st.error("Failed to export model: " + str(e))

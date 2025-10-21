# app.py
import os
import streamlit as st
import joblib
import pandas as pd
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

st.set_page_config(page_title="Titanic Survival Prediction", layout="centered")

# -------------------------
# Configuration / Settings
# -------------------------
MODEL_NAME = "tuned_random_forest_model.joblib"

# Common candidate paths to try (edit if your model is in a different folder)
CANDIDATE_PATHS = [
    MODEL_NAME,
    "./" + MODEL_NAME,
    "models/" + MODEL_NAME,
    "app/models/" + MODEL_NAME,
    "/app/" + MODEL_NAME,
]

# If your model is hosted publicly, you can set a URL here and the app will download it.
# MODEL_URL = "https://example.com/path/to/tuned_random_forest_model.joblib"
MODEL_URL = None  # set to a direct download URL if you want automatic download fallback


# -------------------------
# Helper functions
# -------------------------
def find_model_path():
    """Try candidate paths and return the first that exists, else None."""
    for p in CANDIDATE_PATHS:
        if os.path.exists(p):
            return p
    return None


@st.cache_resource(show_spinner=False)
def load_model(path):
    """Load a joblib model and return it. Accepts GridSearchCV objects too."""
    model = joblib.load(path)
    # If GridSearchCV-like object, prefer best_estimator_ if present
    if hasattr(model, "best_estimator_"):
        try:
            be = model.best_estimator_
            return be
        except Exception:
            # fallback to model itself
            return model
    return model


def safe_check_fitted(model):
    """Return (True, None) if fitted, else (False, error_message)."""
    try:
        check_is_fitted(model)
        return True, None
    except Exception as e:
        return False, str(e)




# -------------------------
# Ensure model exists (or download if configured)
# -------------------------
model_path = find_model_path()

if model_path is None and MODEL_URL:
    st.info("Model not found locally — attempting to download from MODEL_URL ...")
    try:
        from urllib.request import urlretrieve
        urlretrieve(MODEL_URL, MODEL_NAME)
        st.success("Downloaded model to: " + MODEL_NAME)
        model_path = MODEL_NAME
    except Exception as e:
        st.error("Automatic download failed: " + repr(e))
        model_path = None

if model_path is None:
    st.error(
        f"Model file '{MODEL_NAME}' not found. Place it in the repo root (or one of the candidate paths)."
    )
    st.info("If your file is large (>100 MB) GitHub may not accept it — use Git LFS or host the file and set MODEL_URL.")
    st.stop()

# -------------------------
# Load and validate model
# -------------------------
try:
    model = load_model(model_path)
except Exception as e:
    st.error("Failed to load model with joblib.load(): " + repr(e))
    st.stop()

# If someone accidentally saved a search but not best_estimator_, load_model tries to handle it.
fitted, fit_err = safe_check_fitted(model)
if not fitted:
    st.error("Loaded model is not fitted. Please save a fitted estimator or pipeline.")
    st.write("check_is_fitted() error:", fit_err)
    st.stop()

# st.success("Model loaded and verified as fitted.")

# -------------------------
# Application UI
# -------------------------
st.title("Titanic Survival Prediction")
st.write(
    "Enter the passenger details below and click **Predict Survival**. "
    "This app expects features similar to the Titanic dataset."
)

with st.form("input_form"):
    pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3], index=2)
    age = st.number_input("Age", min_value=0.1, max_value=120.0, value=30.0, step=0.5)
    sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, value=0)
    parch = st.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, value=0)
    fare = st.number_input("Fare", min_value=0.0, value=10.0, format="%.2f")
    sex = st.selectbox("Sex", ["male", "female"])
    embarked = st.selectbox("Port of Embarkation (Embarked)", ["C", "Q", "S"], index=2)

    submit = st.form_submit_button("Predict Survival")

# -------------------------
# Build inputs & attempt prediction
# -------------------------
def build_engineered_input():
    """Return DataFrame with engineered features matching your original app:
    ['Pclass','Age','SibSp','Parch','Fare','Sex_male','Embarked_Q','Embarked_S']
    """
    sex_male = 1 if sex == "male" else 0
    embarked_q = 1 if embarked == "Q" else 0
    embarked_s = 1 if embarked == "S" else 0
    df = pd.DataFrame(
        [[pclass, age, sibsp, parch, fare, sex_male, embarked_q, embarked_s]],
        columns=[
            "Pclass",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Sex_male",
            "Embarked_Q",
            "Embarked_S",
        ],
    )
    return df


def build_raw_input():
    """Return DataFrame with raw features often expected by a pipeline:
    ['Pclass','Age','SibSp','Parch','Fare','Sex','Embarked']
    """
    df = pd.DataFrame(
        [
            {
                "Pclass": pclass,
                "Age": age,
                "SibSp": sibsp,
                "Parch": parch,
                "Fare": fare,
                "Sex": sex,
                "Embarked": embarked,
            }
        ]
    )
    return df


if submit:
    # Try predicting using different reasonable input shapes
    input_options = []

    # 1) Try engineered features first (matches your original UI)
    eng_df = build_engineered_input()
    input_options.append(("engineered", eng_df))

    # 2) Also prepare raw input, in case you saved a pipeline that expects raw features
    raw_df = build_raw_input()
    input_options.append(("raw", raw_df))

    prediction = None
    proba = None
    last_error = None

    # If model exposes n_features_in_, use it to pick the right input quickly
    n_in = getattr(model, "n_features_in_", None)

    tried = []
    for name, df in input_options:
        tried.append(name)
        # quick length check
        if n_in is not None and df.shape[1] != n_in:
            # skip if mismatch
            st.write(f"Skipping '{name}' input because model expects {n_in} features but input has {df.shape[1]}.")
            continue

        try:
            pred = model.predict(df)
            prediction = int(pred[0])
            # try predict_proba if available
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(df)[0].tolist()
            break  # success
        except Exception as e:
            last_error = e
            st.write(f"Attempt with '{name}' input failed: {repr(e)}")
            continue

    if prediction is None:
        st.error(
            "Prediction failed. Tried input shapes: "
            + ", ".join(tried)
            + ". See debug messages above. "
            + ("Last error: " + repr(last_error) if last_error else "")
        )
    else:
        if prediction == 1:
            st.success("Prediction: Survived")
        else:
            st.info("Prediction: Did Not Survive")

        if proba is not None:
            # If binary, show prob for class 0 and 1
            st.write("Predicted class probabilities:", proba)
        else:
            st.write("Model does not provide predict_proba() output.")

# -------------------------
# Footer / Tips
# -------------------------
st.markdown("---")


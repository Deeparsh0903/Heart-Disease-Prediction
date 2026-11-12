from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "heart.csv"
TARGET = "target"
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
]
FEATURES = NUMERIC + CATEGORICAL

LABELS = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {
        0: "Typical angina",
        1: "Atypical angina",
        2: "Non-anginal pain",
        3: "Asymptomatic",
    },
    "fbs": {0: "No", 1: "Yes"},
    "restecg": {
        0: "Normal",
        1: "ST-T wave abnormality",
        2: "Left ventricular hypertrophy",
    },
    "exang": {0: "No", 1: "Yes"},
    "slope": {0: "Upsloping", 1: "Flat", 2: "Downsloping"},
    "ca": {0: "0 vessels", 1: "1 vessel", 2: "2 vessels", 3: "3 vessels", 4: "4 vessels"},
    "thal": {0: "Unknown (0)", 1: "Normal", 2: "Fixed defect", 3: "Reversible defect"},
}

st.set_page_config(
    page_title="Heart Disease Prediction Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    missing = sorted(set(FEATURES + [TARGET]) - set(data.columns))
    if missing:
        raise ValueError(f"heart.csv is missing required columns: {', '.join(missing)}")
    return data


@st.cache_resource
def train_model(data: pd.DataFrame) -> Pipeline:
    prep = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    model = Pipeline(
        steps=[("preprocess", prep), ("classifier", LogisticRegression(max_iter=1000))]
    )
    model.fit(data[FEATURES], data[TARGET])
    return model


def display_category(column: str, value: int) -> str:
    return LABELS[column].get(int(value), f"Category {value}")


def score_message(probability: float) -> tuple[str, str]:
    if probability >= 0.5:
        return "Model predicts target class 1", "The model estimated a higher probability for the dataset's positive class."
    return "Model predicts target class 0", "The model estimated a higher probability for the dataset's negative class."


try:
    df = load_data()
    model = train_model(df)
except Exception as exc:
    st.error(f"Could not load or train the model: {exc}")
    st.stop()

st.markdown(
    """
    <style>
    .stApp { background: #0d1018; }
    [data-testid="stHeader"] { background: rgba(13,16,24,0.9); }
    .hero { padding: 0.3rem 0 0.2rem; }
    .hero h1 { font-size: clamp(2rem, 4vw, 3.1rem); margin-bottom: .25rem; }
    .hero p { color: #c7cbd5; font-size: 1.05rem; }
    div.stButton > button[kind="primary"] { background: #ef3f4b; border-color: #ef3f4b; }
    div[data-testid="stMetric"] { background: #171b27; padding: 1rem; border: 1px solid #2a3040; border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero"><h1>❤️ Heart Disease Prediction Dashboard</h1>'
    '<p>Explore a machine learning estimate using the clinical fields in the supplied dataset.</p></div>',
    unsafe_allow_html=True,
)

simple_tab, batch_tab, analysis_tab = st.tabs(
    ["🧪 Simple Prediction", "📁 Batch Testing", "📊 Analysis"]
)

with simple_tab:
    st.header("Individual Prediction")
    left, middle, right = st.columns(3, gap="large")
    with left:
        age = st.slider("Age", 18, 100, 50, help="Age in years")
        sex = st.radio("Sex", [0, 1], format_func=lambda x: LABELS["sex"][x], horizontal=True)
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], format_func=lambda x: f"{LABELS['cp'][x]} ({x})")
        trestbps = st.number_input("Blood Pressure (mm Hg)", 70, 250, 120, step=1)
        chol = st.number_input("Cholesterol (mg/dl)", 80, 700, 240, step=1)
    with middle:
        fbs = st.radio("Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: LABELS["fbs"][x])
        restecg = st.selectbox("Resting ECG Results", [0, 1, 2], format_func=lambda x: f"{LABELS['restecg'][x]} ({x})")
        thalach = st.slider("Maximum Heart Rate", 60, 220, 150)
        exang = st.radio("Exercise-Induced Angina", [0, 1], format_func=lambda x: LABELS["exang"][x])
    with right:
        oldpeak = st.slider("ST Depression", 0.0, 7.0, 1.0, 0.1)
        slope = st.selectbox("ST Segment Slope", [0, 1, 2], format_func=lambda x: f"{LABELS['slope'][x]} ({x})")
        ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3, 4], format_func=lambda x: str(x))
        thal = st.selectbox("Thalassemia", [0, 1, 2, 3], format_func=lambda x: f"{LABELS['thal'][x]} ({x})")

    if st.button("🔎 Analyze Risk", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
            "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
            "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
        }], columns=FEATURES)
        probability = float(model.predict_proba(row)[0, list(model.classes_).index(1)])
        prediction = int(probability >= 0.5)
        title, explanation = score_message(probability)
        st.divider()
        a, b = st.columns([1, 2])
        with a:
            st.metric("Estimated probability of target class 1", f"{probability:.1%}")
        with b:
            st.subheader(title)
            st.write(explanation)
            st.caption("Target class 1 means the positive class in heart.csv. This is a model output, not an individual's clinical risk.")

with batch_tab:
    st.header("Batch Testing")
    st.write("Upload a CSV containing the 13 input columns used in heart.csv. A target column is optional and is retained for comparison.")
    st.download_button("Download input template", df[FEATURES].head(0).to_csv(index=False), "heart_input_template.csv", "text/csv")
    upload = st.file_uploader("Choose a CSV file", type=["csv"])
    if upload is not None:
        try:
            batch = pd.read_csv(upload)
            missing = sorted(set(FEATURES) - set(batch.columns))
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            elif batch.empty:
                st.warning("The uploaded CSV has no rows.")
            else:
                scored = batch.copy()
                probs = model.predict_proba(batch[FEATURES])[:, list(model.classes_).index(1)]
                scored["predicted_target"] = (probs >= 0.5).astype(int)
                scored["probability_target_1"] = probs.round(4)
                st.dataframe(scored, use_container_width=True, hide_index=True)
                st.download_button("Download scored results", scored.to_csv(index=False), "heart_predictions.csv", "text/csv")
        except Exception as exc:
            st.error(f"Could not score the uploaded file: {exc}")

with analysis_tab:
    st.header("Dataset Analysis")
    st.caption("Summary of the supplied heart.csv data. The target label is used by the training notebook as a binary class.")
    a, b, c = st.columns(3)
    a.metric("Records", f"{len(df):,}")
    b.metric("Input features", len(FEATURES))
    c.metric("Missing cells", f"{int(df[FEATURES + [TARGET]].isna().sum().sum()):,}")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Target distribution")
        target_counts = df[TARGET].value_counts().sort_index().rename(index={0: "Target 0", 1: "Target 1"})
        st.bar_chart(target_counts, color="#ef3f4b")
    with col_b:
        st.subheader("Age distribution")
        st.bar_chart(df["age"].value_counts().sort_index(), color="#f28b82")
    st.subheader("Feature summary")
    st.dataframe(df[FEATURES].describe().T, use_container_width=True)
    st.caption("Model: Logistic Regression with one-hot encoding for categorical fields and standard scaling for numeric fields. The full supplied dataset is used for training in this demo app.")

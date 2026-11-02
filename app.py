"""
Streamlit App — Disease Prediction System
==========================================
CodeAlpha Machine Learning Internship — Task 4
Author: Aryan (@aryannten)

Run:
    streamlit run app.py
"""

import os
import sys
from typing import Dict, Any

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Ensure src can be imported
sys.path.insert(0, os.path.dirname(__file__))
from src.predict import predict_disease

# ---------------------------------------------------------------------------
# Config & Paths
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "reports", "figures")

DATASET_CONFIG = {
    "Breast Cancer (Wisconsin)": {
        "key": "breast_cancer",
        "title": "Breast Cancer Malignancy Prediction",
        "description": "Predicts whether a breast mass is **Malignant** or **Benign** from digitized cell nuclei features.",
        "pos_label": "Malignant (Tumor Detected)",
        "neg_label": "Benign (No Malignancy)",
        "color": "#E91E63",
        "metrics": {"Model": "Random Forest", "Accuracy": "97.37%", "Recall": "92.86%", "ROC-AUC": "0.9960"},
        "sample_healthy": {
            "mean radius": 13.54, "mean texture": 14.36, "mean perimeter": 87.46,
            "mean area": 566.3, "mean smoothness": 0.09779, "mean compactness": 0.08129,
            "mean concavity": 0.06664, "mean concave points": 0.04781, "mean symmetry": 0.1885,
            "mean fractal dimension": 0.05766, "worst radius": 15.11, "worst texture": 19.26,
            "worst perimeter": 99.7, "worst area": 711.2, "worst concave points": 0.1047,
        },
        "sample_risk": {
            "mean radius": 20.57, "mean texture": 17.77, "mean perimeter": 132.9,
            "mean area": 1326.0, "mean smoothness": 0.08474, "mean compactness": 0.07864,
            "mean concavity": 0.0869, "mean concave points": 0.07017, "mean symmetry": 0.1812,
            "mean fractal dimension": 0.05667, "worst radius": 24.99, "worst texture": 23.41,
            "worst perimeter": 158.8, "worst area": 1956.0, "worst concave points": 0.186,
        },
    },
    "Heart Disease (UCI Cleveland)": {
        "key": "heart_disease",
        "title": "Heart Disease Risk Assessment",
        "description": "Predicts the presence of **Heart Disease** from clinical and demographic indicators.",
        "pos_label": "Heart Disease Present",
        "neg_label": "No Heart Disease Detected",
        "color": "#D32F2F",
        "metrics": {"Model": "Random Forest", "Accuracy": "91.80%", "Recall": "96.43%", "ROC-AUC": "0.9567"},
        "sample_healthy": {
            "age": 45, "sex": 0, "cp": 1, "trestbps": 120, "chol": 190,
            "fbs": 0, "restecg": 0, "thalach": 172, "exang": 0, "oldpeak": 0.0,
            "slope": 1, "ca": 0, "thal": 3.0,
        },
        "sample_risk": {
            "age": 67, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286,
            "fbs": 1, "restecg": 2, "thalach": 108, "exang": 1, "oldpeak": 1.5,
            "slope": 2, "ca": 3, "thal": 7.0,
        },
    },
    "Diabetes (Pima Indians)": {
        "key": "diabetes",
        "title": "Type 2 Diabetes Risk Prediction",
        "description": "Predicts the likelihood of **Type 2 Diabetes** using diagnostic physiological variables.",
        "pos_label": "Diabetic Tendency",
        "neg_label": "Non-Diabetic",
        "color": "#1976D2",
        "metrics": {"Model": "Random Forest", "Accuracy": "73.38%", "Recall": "75.93%", "ROC-AUC": "0.8172"},
        "sample_healthy": {
            "Pregnancies": 1, "Glucose": 85, "BloodPressure": 66,
            "SkinThickness": 29, "Insulin": 94, "BMI": 26.6,
            "DiabetesPedigreeFunction": 0.351, "Age": 31,
        },
        "sample_risk": {
            "Pregnancies": 8, "Glucose": 183, "BloodPressure": 74,
            "SkinThickness": 38, "Insulin": 240, "BMI": 38.2,
            "DiabetesPedigreeFunction": 0.867, "Age": 54,
        },
    },
}

# ---------------------------------------------------------------------------
# UI Setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Disease Prediction System | CodeAlpha",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🩺 Medical Disease Prediction System")
st.caption("CodeAlpha Machine Learning Internship — Task 4 | Developed by Aryan (@aryannten)")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.image("https://img.shields.io/badge/CodeAlpha-ML_Internship-blue?style=for-the-badge", use_column_width=True)
st.sidebar.header("⚙️ Disease Selection")
selected_disease_name = st.sidebar.selectbox(
    "Choose a diagnostic model:",
    list(DATASET_CONFIG.keys()),
)
cfg = DATASET_CONFIG[selected_disease_name]
ds_key = cfg["key"]

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Model Performance")
for metric_k, metric_v in cfg["metrics"].items():
    st.sidebar.metric(label=metric_k, value=metric_v)

st.sidebar.markdown("---")
st.sidebar.warning(
    "⚠️ **Disclaimer:** This application is strictly an educational machine learning demonstration "
    "built for the CodeAlpha Internship. It is **not** a certified medical diagnostic device. "
    "Always consult a licensed medical professional."
)

# ---------------------------------------------------------------------------
# Main Content
# ---------------------------------------------------------------------------
st.subheader(cfg["title"])
st.write(cfg["description"])

# Check that model artifacts exist
model_file = os.path.join(MODELS_DIR, f"best_model_{ds_key}.joblib")
if not os.path.exists(model_file):
    st.error(
        f"Model artifacts for **{selected_disease_name}** were not found. "
        "Please execute the pipeline first (`python run_pipeline.py`)."
    )
    st.stop()

# Preset buttons
st.markdown("#### ⚡ Quick Presets")
preset_col1, preset_col2, preset_col3 = st.columns([1, 1, 2])
with preset_col1:
    load_healthy = st.button("🟢 Load Low-Risk Sample")
with preset_col2:
    load_risk = st.button("🔴 Load High-Risk Sample")

# Session state management for form inputs
if "current_ds" not in st.session_state or st.session_state.current_ds != ds_key:
    st.session_state.current_ds = ds_key
    st.session_state.patient_data = cfg["sample_healthy"].copy()

if load_healthy:
    st.session_state.patient_data = cfg["sample_healthy"].copy()
    st.rerun()

if load_risk:
    st.session_state.patient_data = cfg["sample_risk"].copy()
    st.rerun()

# ---------------------------------------------------------------------------
# Input Form
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📋 Patient Clinical Input")

features_path = os.path.join(MODELS_DIR, f"features_{ds_key}.joblib")
features_list = joblib.load(features_path)

input_dict = {}

if ds_key == "diabetes":
    cols = st.columns(4)
    fields = [
        ("Pregnancies", "Pregnancies (count)", 0.0, 20.0, 1.0),
        ("Glucose", "Glucose (mg/dL)", 0.0, 300.0, 1.0),
        ("BloodPressure", "Blood Pressure (mm Hg)", 0.0, 150.0, 1.0),
        ("SkinThickness", "Skin Thickness (mm)", 0.0, 100.0, 1.0),
        ("Insulin", "Serum Insulin (mu U/ml)", 0.0, 900.0, 1.0),
        ("BMI", "Body Mass Index (BMI)", 0.0, 70.0, 0.1),
        ("DiabetesPedigreeFunction", "Diabetes Pedigree Fn", 0.0, 3.0, 0.01),
        ("Age", "Age (years)", 1.0, 120.0, 1.0),
    ]
    for i, (k, label, min_v, max_v, step) in enumerate(fields):
        col = cols[i % 4]
        def_v = float(st.session_state.patient_data.get(k, cfg["sample_healthy"].get(k, 0.0)))
        input_dict[k] = col.number_input(label, min_value=min_v, max_value=max_v, value=def_v, step=step, key=f"inp_{k}")

elif ds_key == "heart_disease":
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        input_dict["age"] = st.number_input("Age (years)", 1, 120, int(st.session_state.patient_data.get("age", 55)))
        input_dict["sex"] = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
        input_dict["cp"] = st.selectbox("Chest Pain Type", options=[1, 2, 3, 4], format_func=lambda x: {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-anginal", 4: "Asymptomatic"}[x], index=int(st.session_state.patient_data.get("cp", 4)) - 1)
        input_dict["trestbps"] = st.number_input("Resting Blood Pressure (mm Hg)", 50, 250, int(st.session_state.patient_data.get("trestbps", 130)))
    with col2:
        input_dict["chol"] = st.number_input("Serum Cholestoral (mg/dl)", 100, 600, int(st.session_state.patient_data.get("chol", 240)))
        input_dict["fbs"] = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "True (>120)" if x == 1 else "False (<=120)")
        input_dict["restecg"] = st.selectbox("Resting ECG Results", options=[0, 1, 2], format_func=lambda x: {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"}[x], index=int(st.session_state.patient_data.get("restecg", 0)))
        input_dict["thalach"] = st.number_input("Max Heart Rate Achieved", 50, 250, int(st.session_state.patient_data.get("thalach", 150)))
    with col3:
        input_dict["exang"] = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        input_dict["oldpeak"] = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, float(st.session_state.patient_data.get("oldpeak", 1.0)), step=0.1)
        input_dict["slope"] = st.selectbox("Slope of Peak Exercise ST", options=[1, 2, 3], format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x], index=int(st.session_state.patient_data.get("slope", 1)) - 1)
    with col4:
        input_dict["ca"] = st.number_input("Major Vessels Colored by Fluoroscopy (0-3)", 0, 3, int(st.session_state.patient_data.get("ca", 0)))
        input_dict["thal"] = st.selectbox("Thalassemia Status", options=[3.0, 6.0, 7.0], format_func=lambda x: {3.0: "Normal", 6.0: "Fixed Defect", 7.0: "Reversible Defect"}.get(x, str(x)))

else:
    # Breast Cancer: display the top influential features in 3 columns
    st.info("Displaying primary morphological features from fine needle aspirate (FNA). Remaining features default to dataset mean.")
    cols = st.columns(3)
    for i, f in enumerate(features_list):
        col = cols[i % 3]
        default_val = float(st.session_state.patient_data.get(f, 0.0))
        input_dict[f] = col.number_input(f, value=default_val, format="%.4f", key=f"inp_{f}")

# ---------------------------------------------------------------------------
# Prediction Execution
# ---------------------------------------------------------------------------
st.markdown("---")
predict_btn = st.button("🔬 Compute Diagnostic Risk", type="primary", use_container_width=True)

if predict_btn:
    try:
        pred, prob = predict_disease(ds_key, input_dict)
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if pred == 1:
                st.error(f"### ⚠️ Result: {cfg['pos_label']}")
                st.markdown(f"**Calculated Disease Probability: `{prob:.2%}`**")
                st.markdown(
                    "The model has flagged clinical patterns correlating with the condition. "
                    "In a medical triage workflow, high sensitivity/recall prevents missing high-risk patients."
                )
            else:
                st.success(f"### ✅ Result: {cfg['neg_label']}")
                st.markdown(f"**Calculated Disease Probability: `{prob:.2%}`**")
                st.markdown("Clinical indicators fall within baseline ranges.")

        with res_col2:
            st.markdown("#### Diagnostic Probability Gauge")
            st.progress(float(prob))
            if prob >= 0.70:
                st.error("Risk Level: **HIGH**")
            elif prob >= 0.40:
                st.warning("Risk Level: **MODERATE / MONITOR**")
            else:
                st.success("Risk Level: **LOW**")

    except Exception as e:
        st.error(f"Prediction error: {e}")

# ---------------------------------------------------------------------------
# Visualizations tab
# ---------------------------------------------------------------------------
with st.expander("📈 View Model Evaluation Curves & Visualizations"):
    roc_fig_path = os.path.join(FIGURES_DIR, f"roc_{ds_key}.png")
    corr_fig_path = os.path.join(FIGURES_DIR, f"corr_{ds_key}.png")
    fig_col1, fig_col2 = st.columns(2)
    if os.path.exists(roc_fig_path):
        with fig_col1:
            st.image(roc_fig_path, caption=f"ROC Curves — {selected_disease_name}")
    if os.path.exists(corr_fig_path):
        with fig_col2:
            st.image(corr_fig_path, caption=f"Correlation Heatmap — {selected_disease_name}")

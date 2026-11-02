"""
Prediction Module
==================
Load a saved model and predict disease probability for a new patient.
"""

import os
from typing import Dict, Tuple

import numpy as np
import joblib

# Paths
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


# Common alias mappings for diabetes features (friendly name -> OpenML column)
DIABETES_ALIASES = {
    "pregnancies": "preg",
    "glucose": "plas",
    "plasma": "plas",
    "bloodpressure": "pres",
    "pressure": "pres",
    "bp": "pres",
    "skinthickness": "skin",
    "skin": "skin",
    "insulin": "insu",
    "bmi": "mass",
    "mass": "mass",
    "diabetespedigreefunction": "pedi",
    "pedigree": "pedi",
    "pedi": "pedi",
    "age": "age",
    "preg": "preg",
}


def _normalize_patient_dict(
    dataset_name: str,
    patient_dict: Dict[str, float],
    feature_names: list,
) -> Dict[str, float]:
    """Normalize input dictionary to match the model's expected feature names."""
    norm = {}
    lower_input = {k.lower().replace(" ", "").replace("_", ""): v for k, v in patient_dict.items()}

    if dataset_name == "diabetes":
        for k, v in patient_dict.items():
            cleaned_key = k.lower().replace(" ", "").replace("_", "")
            mapped_key = DIABETES_ALIASES.get(cleaned_key, k)
            norm[mapped_key] = float(v)

    elif dataset_name == "heart_disease":
        # Pass through direct matches
        for k, v in patient_dict.items():
            norm[k] = float(v)

        # Handle raw categoricals if provided
        # cp: 1, 2, 3, 4 (cp_2, cp_3, cp_4)
        if "cp" in patient_dict:
            cp_val = patient_dict["cp"]
            for target_col in ["cp_2", "cp_3", "cp_4"]:
                target_val = int(target_col.split("_")[1])
                norm[target_col] = 1.0 if cp_val == target_val else 0.0

        # restecg: 0, 1, 2 (restecg_1, restecg_2)
        if "restecg" in patient_dict:
            re_val = patient_dict["restecg"]
            for target_col in ["restecg_1", "restecg_2"]:
                target_val = int(target_col.split("_")[1])
                norm[target_col] = 1.0 if re_val == target_val else 0.0

        # slope: 1, 2, 3 (slope_2, slope_3)
        if "slope" in patient_dict:
            s_val = patient_dict["slope"]
            for target_col in ["slope_2", "slope_3"]:
                target_val = int(target_col.split("_")[1])
                norm[target_col] = 1.0 if s_val == target_val else 0.0

        # thal: 3, 6, 7 (thal_6.0, thal_7.0)
        if "thal" in patient_dict:
            th_val = float(patient_dict["thal"])
            for target_col in ["thal_6.0", "thal_7.0"]:
                norm[target_col] = 1.0 if target_col == f"thal_{th_val}" else 0.0

    else:
        # Breast cancer or standard names
        for k, v in patient_dict.items():
            norm[k] = float(v)
            norm[k.lower()] = float(v)

    return norm


def predict_disease(
    dataset_name: str,
    patient_dict: Dict[str, float],
) -> Tuple[int, float]:
    """Predict disease for a single patient using the saved best model.

    Args:
        dataset_name: One of 'breast_cancer', 'heart_disease', 'diabetes'.
        patient_dict: Dict mapping feature names → values. Accepts both
            canonical feature names and common clinical aliases.

    Returns:
        (prediction, probability) where prediction is 0 or 1
        and probability is P(disease=1).

    Raises:
        FileNotFoundError: If no saved model/scaler/features exist.
    """
    # Load artifacts
    model_path = os.path.join(MODELS_DIR, f"best_model_{dataset_name}.joblib")
    scaler_path = os.path.join(MODELS_DIR, f"scaler_{dataset_name}.joblib")
    features_path = os.path.join(MODELS_DIR, f"features_{dataset_name}.joblib")

    for p, label in [(model_path, "model"), (scaler_path, "scaler"), (features_path, "features")]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"No saved {label} found at {p}. Run the training pipeline first."
            )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(features_path)

    # Normalize inputs
    norm_dict = _normalize_patient_dict(dataset_name, patient_dict, feature_names)

    # Build feature vector in the exact order expected by the scaler and model
    X = np.array([[norm_dict.get(f, norm_dict.get(f.lower(), 0.0)) for f in feature_names]])

    # Scale
    X_scaled = scaler.transform(X)

    # Predict
    prediction = int(model.predict(X_scaled)[0])
    probability = float(model.predict_proba(X_scaled)[0, 1])

    return prediction, probability


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Demo with test cases for each disease
    demo = {
        "diabetes": {
            "Pregnancies": 6, "Glucose": 148, "BloodPressure": 72,
            "SkinThickness": 35, "Insulin": 160, "BMI": 33.6,
            "DiabetesPedigreeFunction": 0.627, "Age": 50,
        },
        "heart_disease": {
            "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
            "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3,
            "slope": 0, "ca": 0, "thal": 6.0,
        },
        "breast_cancer": {
            "mean radius": 17.99, "mean texture": 10.38, "mean perimeter": 122.8,
            "mean area": 1001.0, "mean smoothness": 0.1184, "worst radius": 25.38,
            "worst concave points": 0.2654,
        },
    }

    print("=" * 60)
    print("  DISEASE PREDICTION VERIFICATION DEMO")
    print("=" * 60)
    for ds, patient in demo.items():
        try:
            pred, prob = predict_disease(ds, patient)
            label = "Positive (disease likely)" if pred == 1 else "Negative (no disease)"
            print(f"[{ds}] -> {label} | Probability: {prob:.2%}")
        except Exception as e:
            print(f"[{ds}] Error: {e}")


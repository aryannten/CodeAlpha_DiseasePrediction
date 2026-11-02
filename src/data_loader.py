"""
Data Loader Module
===================
Loads all three disease datasets with graceful fallback on failure.

Datasets:
    1. Breast Cancer Wisconsin (from sklearn)
    2. Heart Disease - UCI Cleveland (from ucimlrepo or CSV fallback)
    3. Pima Indians Diabetes (from OpenML or CSV fallback)
"""

import os
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

# Directory for cached CSVs
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def load_breast_cancer_data() -> Tuple[pd.DataFrame, str]:
    """Load the Breast Cancer Wisconsin dataset from sklearn.

    Returns:
        Tuple of (DataFrame with features + 'target' column, target column name).
    """
    print("[INFO] Loading Breast Cancer Wisconsin dataset from sklearn...")
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    # 0 = malignant, 1 = benign in sklearn; we invert so 1 = malignant (disease present)
    df["target"] = (data.target == 0).astype(int)
    print(f"[OK]  Loaded: {df.shape[0]} samples, {df.shape[1] - 1} features")
    return df, "target"


def load_heart_disease_data() -> Tuple[pd.DataFrame, str]:
    """Load the Heart Disease (UCI Cleveland) dataset.

    Tries ucimlrepo first, then falls back to a public CSV URL.

    Returns:
        Tuple of (DataFrame with features + 'target' column, target column name).
    """
    df = None

    # --- Attempt 1: ucimlrepo ---
    try:
        print("[INFO] Loading Heart Disease dataset via ucimlrepo...")
        from ucimlrepo import fetch_ucirepo
        heart = fetch_ucirepo(id=45)  # Heart Disease dataset
        X = heart.data.features.copy()
        y = heart.data.targets.copy()
        df = pd.concat([X, y], axis=1)
        # Target: 0 = no disease, 1-4 = disease → binarize to 0/1
        target_col = df.columns[-1]
        df[target_col] = (df[target_col] > 0).astype(int)
        df.rename(columns={target_col: "target"}, inplace=True)
        print(f"[OK]  Loaded via ucimlrepo: {df.shape[0]} samples")
    except Exception as e:
        warnings.warn(f"ucimlrepo failed: {e}. Trying CSV fallback...")

    # --- Attempt 2: Public CSV URL ---
    if df is None:
        try:
            print("[INFO] Falling back to public CSV URL for Heart Disease...")
            url = (
                "https://archive.ics.uci.edu/ml/machine-learning-databases/"
                "heart-disease/processed.cleveland.data"
            )
            columns = [
                "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
            ]
            df = pd.read_csv(url, names=columns, na_values="?")
            df["target"] = (df["target"] > 0).astype(int)
            print(f"[OK]  Loaded via CSV URL: {df.shape[0]} samples")
        except Exception as e:
            raise RuntimeError(
                f"[ERROR] Could not load Heart Disease dataset. "
                f"Check your internet connection.\nDetails: {e}"
            )

    # Replace '?' strings that might have slipped through
    df.replace("?", np.nan, inplace=True)

    # Save a local copy
    csv_path = os.path.join(DATA_DIR, "heart_disease.csv")
    df.to_csv(csv_path, index=False)
    print(f"[OK]  Saved to {csv_path}")

    return df, "target"


def load_diabetes_data() -> Tuple[pd.DataFrame, str]:
    """Load the Pima Indians Diabetes dataset.

    Tries sklearn fetch_openml first, then falls back to a public CSV URL.

    Returns:
        Tuple of (DataFrame with features + 'target' column, target column name).
    """
    df = None

    # --- Attempt 1: OpenML via sklearn ---
    try:
        print("[INFO] Loading Diabetes dataset via OpenML (sklearn)...")
        from sklearn.datasets import fetch_openml
        diabetes = fetch_openml(name="diabetes", version=1, as_frame=True, parser="auto")
        df = diabetes.data.copy()
        df["target"] = (diabetes.target == "tested_positive").astype(int)
        print(f"[OK]  Loaded via OpenML: {df.shape[0]} samples")
    except Exception as e:
        warnings.warn(f"OpenML failed: {e}. Trying CSV fallback...")

    # --- Attempt 2: Public CSV URL ---
    if df is None:
        try:
            print("[INFO] Falling back to public CSV URL for Diabetes...")
            url = (
                "https://raw.githubusercontent.com/jbrownlee/"
                "Datasets/master/pima-indians-diabetes.data.csv"
            )
            columns = [
                "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "target"
            ]
            df = pd.read_csv(url, names=columns)
            print(f"[OK]  Loaded via CSV URL: {df.shape[0]} samples")
        except Exception as e:
            raise RuntimeError(
                f"[ERROR] Could not load Diabetes dataset. "
                f"Check your internet connection.\nDetails: {e}"
            )

    # Save a local copy
    csv_path = os.path.join(DATA_DIR, "pima_diabetes.csv")
    df.to_csv(csv_path, index=False)
    print(f"[OK]  Saved to {csv_path}")

    return df, "target"


def load_dataset(name: str) -> Tuple[pd.DataFrame, str]:
    """Load a dataset by name.

    Args:
        name: One of 'breast_cancer', 'heart_disease', or 'diabetes'.

    Returns:
        Tuple of (DataFrame, target_column_name).

    Raises:
        ValueError: If the dataset name is not recognized.
    """
    loaders = {
        "breast_cancer": load_breast_cancer_data,
        "heart_disease": load_heart_disease_data,
        "diabetes": load_diabetes_data,
    }

    if name not in loaders:
        raise ValueError(
            f"Unknown dataset '{name}'. Choose from: {list(loaders.keys())}"
        )

    return loaders[name]()


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for ds_name in ["breast_cancer", "heart_disease", "diabetes"]:
        df, target = load_dataset(ds_name)
        print(f"\n{'='*50}")
        print(f"Dataset: {ds_name}")
        print(f"Shape: {df.shape}")
        print(f"Target distribution:\n{df[target].value_counts()}\n")


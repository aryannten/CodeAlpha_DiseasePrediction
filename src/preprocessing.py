"""
Preprocessing Module
=====================
Cleans data, handles missing values, encodes categoricals,
and creates train/test splits with proper scaling (no data leakage).
"""

import numpy as np
import pandas as pd
from typing import Tuple, List

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Columns in Pima Diabetes where 0 is physiologically impossible → treat as NaN
# Note: Pregnancies/preg can legitimately be 0, so it is NOT included.
DIABETES_ZERO_COLS = {
    "glucose", "bloodpressure", "skinthickness", "insulin", "bmi",
    "plas", "pres", "skin", "insu", "mass",
}


def handle_missing_values(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Handle missing / invalid values depending on the dataset.

    Args:
        df: Input DataFrame.
        dataset_name: One of 'breast_cancer', 'heart_disease', 'diabetes'.

    Returns:
        Cleaned DataFrame.
    """
    df = df.copy()

    if dataset_name == "diabetes":
        # Replace physiologically impossible zeros with NaN, then impute
        for col in df.columns:
            if col.lower() in DIABETES_ZERO_COLS and col != "target":
                df[col] = df[col].replace(0, np.nan)

    if dataset_name == "heart_disease":
        # Replace '?' strings (some loaders leave them as strings)
        df = df.replace("?", np.nan)

    # Convert all feature columns to numeric (safety net)
    for col in df.columns:
        if col != "target":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Impute remaining NaNs with column median
    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        print(f"[INFO] Imputing {missing_before} missing values with median...")
        for col in df.columns:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

    return df


def encode_categoricals(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """One-hot encode categorical features if needed.

    Heart Disease has categorical columns: cp, restecg, slope, thal.
    Breast Cancer and Diabetes have only numeric features.

    Args:
        df: Input DataFrame.
        dataset_name: Dataset identifier.

    Returns:
        DataFrame with encoded features.
    """
    df = df.copy()

    if dataset_name == "heart_disease":
        cat_cols = []
        # Identify columns that should be treated as categorical
        potential_cats = ["cp", "restecg", "slope", "thal"]
        for col in potential_cats:
            if col in df.columns and df[col].nunique() <= 10:
                cat_cols.append(col)

        if cat_cols:
            print(f"[INFO] One-hot encoding: {cat_cols}")
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    return df


def split_and_scale(
    df: pd.DataFrame,
    target_col: str = "target",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], StandardScaler]:
    """Stratified train/test split + StandardScaler (fit on train only).

    Args:
        df: Preprocessed DataFrame with target column.
        target_col: Name of the target column.
        test_size: Fraction for the test set.
        random_state: Seed for reproducibility.

    Returns:
        (X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler)
    """
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features – fit ONLY on training data to avoid leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"[OK]  Train: {X_train_scaled.shape[0]} | Test: {X_test_scaled.shape[0]}")

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler


def preprocess_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    target_col: str = "target",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], StandardScaler, pd.DataFrame]:
    """Full preprocessing pipeline: clean → encode → clean again → split → scale.

    Args:
        df: Raw DataFrame.
        dataset_name: Dataset identifier.
        target_col: Name of the target column.

    Returns:
        (X_train, X_test, y_train, y_test, feature_names, scaler, cleaned_df)
    """
    print(f"\n{'─'*50}")
    print(f"Preprocessing: {dataset_name}")
    print(f"{'─'*50}")

    # Step 1: Handle missing values (first pass)
    df_clean = handle_missing_values(df, dataset_name)

    # Step 2: Encode categoricals
    df_encoded = encode_categoricals(df_clean, dataset_name)

    # Step 3: Final NaN sweep — encoding can introduce NaN in rare cases
    remaining_nans = df_encoded.isnull().sum().sum()
    if remaining_nans > 0:
        print(f"[INFO] Final sweep: imputing {remaining_nans} residual NaN values...")
        for col in df_encoded.columns:
            if df_encoded[col].isnull().any():
                df_encoded[col] = df_encoded[col].fillna(df_encoded[col].median())

    # Drop any rows that still have NaN (extreme safety net)
    before = len(df_encoded)
    df_encoded = df_encoded.dropna()
    after = len(df_encoded)
    if before != after:
        print(f"[INFO] Dropped {before - after} rows that still contained NaN")

    # Step 4: Split and scale
    X_train, X_test, y_train, y_test, features, scaler = split_and_scale(
        df_encoded, target_col
    )

    return X_train, X_test, y_train, y_test, features, scaler, df_encoded

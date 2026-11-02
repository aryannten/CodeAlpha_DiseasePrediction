"""
Training Module
================
Train and tune four classifiers with cross-validation and hyperparameter search.

Models: Logistic Regression, SVM (RBF), Random Forest, XGBoost
"""

import warnings
from typing import Dict, Any, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    cross_val_score,
)


# ---------------------------------------------------------------------------
# Model definitions + hyperparameter grids
# ---------------------------------------------------------------------------
def get_models_and_params() -> Dict[str, Tuple[Any, Dict]]:
    """Return a dict of model_name → (estimator, param_grid).

    All models use class_weight='balanced' (or scale_pos_weight for XGBoost)
    to handle class imbalance — this is simpler and less leak-prone than SMOTE.

    Returns:
        Dictionary mapping model names to (estimator, param_grid) tuples.
    """
    models = {
        "Logistic Regression": (
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=42,
            ),
            {
                "C": [0.01, 0.1, 1, 10],
                "solver": ["lbfgs"],
            },
        ),
        "SVM (RBF)": (
            SVC(
                kernel="rbf",
                class_weight="balanced",
                probability=True,   # needed for ROC-AUC & predict_proba
                random_state=42,
            ),
            {
                "C": [0.1, 1, 10],
                "gamma": ["scale", "auto"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
            },
        ),
        "XGBoost": (
            XGBClassifier(
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1,
            ),
            {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1],
            },
        ),
    }

    return models


def compute_scale_pos_weight(y: np.ndarray) -> float:
    """Compute scale_pos_weight for XGBoost from target array.

    Args:
        y: Binary target array (0/1).

    Returns:
        Ratio of negative to positive samples.
    """
    neg = np.sum(y == 0)
    pos = np.sum(y == 1)
    return neg / pos if pos > 0 else 1.0


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    dataset_name: str = "",
) -> Dict[str, Any]:
    """Train and tune all four models using GridSearchCV.

    Args:
        X_train: Scaled training features.
        y_train: Training labels.
        dataset_name: Name for logging.

    Returns:
        Dict mapping model_name → best estimator (fitted).
    """
    print(f"\n{'='*60}")
    print(f"  TRAINING MODELS — {dataset_name}")
    print(f"{'='*60}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models_and_params = get_models_and_params()
    best_models: Dict[str, Any] = {}

    for name, (estimator, param_grid) in models_and_params.items():
        print(f"\n▸ {name}")

        # Set XGBoost scale_pos_weight dynamically
        if name == "XGBoost":
            spw = compute_scale_pos_weight(y_train)
            estimator.set_params(scale_pos_weight=spw)
            print(f"  scale_pos_weight = {spw:.2f}")

        # Grid search with ROC-AUC scoring
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
                refit=True,
            )
            grid.fit(X_train, y_train)

        best_models[name] = grid.best_estimator_

        # Also report cross-val recall for reference
        recall_scores = cross_val_score(
            grid.best_estimator_, X_train, y_train, cv=cv, scoring="recall"
        )

        print(f"  Best params : {grid.best_params_}")
        print(f"  CV ROC-AUC  : {grid.best_score_:.4f}")
        print(f"  CV Recall   : {recall_scores.mean():.4f} ± {recall_scores.std():.4f}")

    return best_models


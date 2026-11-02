"""
Evaluation Module
==================
Compute metrics and generate publication-quality plots for every model.

Outputs:
    - Confusion matrix plot per model
    - Combined ROC curve (all models on one plot) per dataset
    - Feature importance bar charts (Random Forest, XGBoost)
    - Comparison table (CSV + printed)
"""

import os
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe in notebooks & scripts)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

# Output directories
FIGURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "reports", "figures"
)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """Compute standard classification metrics.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities for the positive class.

    Returns:
        Dict of metric_name → value.
    """
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    dataset_name: str,
) -> str:
    """Plot and save a confusion matrix heatmap.

    Returns:
        Path to the saved figure.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}\n({dataset_name})")
    fig.tight_layout()

    fname = f"cm_{dataset_name}_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    path = os.path.join(FIGURES_DIR, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_roc_curves(
    y_true: np.ndarray,
    model_probs: Dict[str, np.ndarray],
    dataset_name: str,
) -> str:
    """Plot ROC curves for all models on one graph.

    Args:
        y_true: Ground-truth labels.
        model_probs: Dict of model_name → predicted probabilities.
        dataset_name: Dataset identifier (for title and filename).

    Returns:
        Path to the saved figure.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, y_prob in model_probs.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {dataset_name}")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    fname = f"roc_{dataset_name}.png"
    path = os.path.join(FIGURES_DIR, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    model_name: str,
    dataset_name: str,
    top_n: int = 10,
) -> str:
    """Plot top-N feature importances for tree-based models.

    Args:
        model: Fitted tree-based estimator with feature_importances_.
        feature_names: List of feature names.
        model_name: Model name for the title.
        dataset_name: Dataset name for filename.
        top_n: Number of top features to show.

    Returns:
        Path to the saved figure.
    """
    importances = model.feature_importances_
    n_features = min(top_n, len(feature_names))
    indices = np.argsort(importances)[::-1][:n_features]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        range(n_features),
        importances[indices][::-1],
        color=sns.color_palette("viridis", n_features),
    )
    ax.set_yticks(range(n_features))
    ax.set_yticklabels([feature_names[i] for i in indices][::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}\n({dataset_name})")
    fig.tight_layout()

    fname = f"fi_{dataset_name}_{model_name.replace(' ', '_')}.png"
    path = os.path.join(FIGURES_DIR, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------
def evaluate_all_models(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    dataset_name: str,
) -> pd.DataFrame:
    """Evaluate all trained models on the test set.

    Prints classification reports, saves plots, and returns a comparison table.

    Args:
        models: Dict of model_name → fitted estimator.
        X_test: Scaled test features.
        y_test: Test labels.
        feature_names: Feature names (for importance plots).
        dataset_name: Dataset identifier.

    Returns:
        DataFrame with one row per model and columns for each metric.
    """
    print(f"\n{'='*60}")
    print(f"  EVALUATION — {dataset_name}")
    print(f"{'='*60}")

    results = []
    model_probs: Dict[str, np.ndarray] = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics["Model"] = name
        results.append(metrics)
        model_probs[name] = y_prob

        # Print classification report
        print(f"\n▸ {name}")
        print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

        # Confusion matrix plot
        plot_confusion_matrix(y_test, y_pred, name, dataset_name)

        # Feature importance for tree-based models
        if hasattr(model, "feature_importances_"):
            plot_feature_importance(model, feature_names, name, dataset_name)

    # ROC curves (all models in one plot)
    plot_roc_curves(y_test, model_probs, dataset_name)

    # Comparison table
    df_results = pd.DataFrame(results)
    df_results = df_results[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]]
    df_results = df_results.sort_values("ROC-AUC", ascending=False).reset_index(drop=True)

    # Save CSV
    csv_path = os.path.join(FIGURES_DIR, f"comparison_{dataset_name}.csv")
    df_results.to_csv(csv_path, index=False)

    print(f"\n📊 Comparison Table — {dataset_name}")
    print(df_results.to_string(index=False))
    print(f"\n[OK]  Results saved to {csv_path}")

    return df_results

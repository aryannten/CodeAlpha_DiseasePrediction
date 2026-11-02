"""Run the disease prediction notebook end-to-end and save results."""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(__file__))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
np.random.seed(42)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from src.data_loader import load_dataset
from src.preprocessing import preprocess_dataset
from src.train import train_all_models
from src.evaluate import evaluate_all_models, FIGURES_DIR

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

DATASETS = ['breast_cancer', 'heart_disease', 'diabetes']

# ── 1. LOAD ──────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 1: LOADING DATASETS")
print("="*70)
raw_data = {}
for ds_name in DATASETS:
    try:
        df, target_col = load_dataset(ds_name)
        raw_data[ds_name] = (df, target_col)
        print(f'  [OK] {ds_name}: {df.shape}')
    except Exception as e:
        print(f'  [ERROR] {ds_name}: {e}')

# ── 2. EDA ───────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 2: EDA & PLOTS")
print("="*70)

for ds_name, (df, target_col) in raw_data.items():
    print(f'\n--- EDA: {ds_name} ---')
    print(f'Shape: {df.shape}')
    print(f'Missing: {df.isnull().sum().sum()}')
    print(f'Target distribution:\n{df[target_col].value_counts().to_dict()}')

    # Class balance plot
    fig, ax = plt.subplots(figsize=(4, 3))
    df[target_col].value_counts().plot.bar(ax=ax, color=['#4CAF50', '#F44336'])
    ax.set_title(f'Class Balance — {ds_name}')
    ax.set_xlabel('Class'); ax.set_ylabel('Count')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f'class_balance_{ds_name}.png'), dpi=150)
    plt.close(fig)

    # Correlation heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    fig_size = (12, 10) if numeric_df.shape[1] <= 20 else (16, 14)
    fig, ax = plt.subplots(figsize=fig_size)
    sns.heatmap(numeric_df.corr(), cmap='coolwarm', center=0,
                linewidths=0.3, ax=ax, annot=(numeric_df.shape[1] <= 15))
    ax.set_title(f'Correlation Heatmap — {ds_name}')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f'corr_{ds_name}.png'), dpi=150)
    plt.close(fig)

    # Feature distributions by target
    feat_cols = [c for c in numeric_df.columns if c != target_col][:8]
    n_feats = len(feat_cols)
    ncols = min(4, n_feats)
    nrows = (n_feats + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    for i, col in enumerate(feat_cols):
        for cls in [0, 1]:
            subset = df[df[target_col] == cls][col].dropna()
            axes[i].hist(subset, bins=25, alpha=0.5, label=f'Class {cls}', density=True)
        axes[i].set_title(col, fontsize=10)
        axes[i].legend(fontsize=8)
    for j in range(n_feats, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f'Feature Distributions — {ds_name}', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f'dist_{ds_name}.png'), dpi=150)
    plt.close(fig)

    print(f'  [OK] EDA plots saved for {ds_name}')

# ── 3. PREPROCESSING ────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 3: PREPROCESSING")
print("="*70)
processed_data = {}
for ds_name, (df, target_col) in raw_data.items():
    X_train, X_test, y_train, y_test, features, scaler, df_clean = preprocess_dataset(
        df, ds_name, target_col
    )
    processed_data[ds_name] = {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'features': features, 'scaler': scaler,
    }

# ── 4. TRAINING ─────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 4: TRAINING & TUNING")
print("="*70)
all_models = {}
for ds_name, data in processed_data.items():
    best_models = train_all_models(
        data['X_train'], data['y_train'], dataset_name=ds_name
    )
    all_models[ds_name] = best_models

# ── 5. EVALUATION ────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 5: EVALUATION")
print("="*70)
all_results = {}
for ds_name, data in processed_data.items():
    df_results = evaluate_all_models(
        models=all_models[ds_name],
        X_test=data['X_test'],
        y_test=data['y_test'],
        feature_names=data['features'],
        dataset_name=ds_name,
    )
    all_results[ds_name] = df_results

# ── 6. BEST MODEL SELECTION & SAVING ────────────────────────────────────
print("\n" + "="*70)
print("  STEP 6: BEST MODEL SELECTION")
print("="*70)
best_summary = []
for ds_name, df_res in all_results.items():
    df_sorted = df_res.sort_values(['ROC-AUC', 'Recall'], ascending=False)
    best_row = df_sorted.iloc[0]
    best_model_name = best_row['Model']

    print(f'\n[BEST] {ds_name}: Best model = {best_model_name}')
    print(f'   ROC-AUC={best_row["ROC-AUC"]:.4f}  Recall={best_row["Recall"]:.4f}  '
          f'F1={best_row["F1-Score"]:.4f}  Accuracy={best_row["Accuracy"]:.4f}')

    # Save model, scaler, and feature names
    best_estimator = all_models[ds_name][best_model_name]
    joblib.dump(best_estimator, os.path.join(MODELS_DIR, f'best_model_{ds_name}.joblib'))
    joblib.dump(processed_data[ds_name]['scaler'], os.path.join(MODELS_DIR, f'scaler_{ds_name}.joblib'))
    joblib.dump(processed_data[ds_name]['features'], os.path.join(MODELS_DIR, f'features_{ds_name}.joblib'))

    best_summary.append({
        'Dataset': ds_name,
        'Best Model': best_model_name,
        'Accuracy': round(best_row['Accuracy'], 4),
        'Precision': round(best_row['Precision'], 4),
        'Recall': round(best_row['Recall'], 4),
        'F1-Score': round(best_row['F1-Score'], 4),
        'ROC-AUC': round(best_row['ROC-AUC'], 4),
    })

df_summary = pd.DataFrame(best_summary)
df_summary.to_csv(os.path.join(FIGURES_DIR, 'best_models_summary.csv'), index=False)

print(f'\n\n{"="*70}')
print("  FINAL RESULTS SUMMARY")
print(f'{"="*70}')
print(df_summary.to_string(index=False))
print(f'\n[OK] All models saved to {MODELS_DIR}')
print(f'[OK] All figures saved to {FIGURES_DIR}')

# ── 7. PREDICTION DEMO ──────────────────────────────────────────────────
print("\n" + "="*70)
print("  PREDICTION DEMO")
print("="*70)
from src.predict import predict_disease

# Test with a sample from each dataset
for ds_name in DATASETS:
    try:
        # Use first test sample as demo
        features = processed_data[ds_name]['features']
        X_test_sample = processed_data[ds_name]['X_test'][0]
        scaler = processed_data[ds_name]['scaler']
        # Inverse transform to get original values
        X_orig = scaler.inverse_transform(X_test_sample.reshape(1, -1))[0]
        patient = dict(zip(features, X_orig))
        pred, prob = predict_disease(ds_name, patient)
        label = 'Positive (disease)' if pred == 1 else 'Negative (no disease)'
        print(f'  {ds_name}: {label} | P(disease)={prob:.2%}')
    except Exception as e:
        print(f'  {ds_name}: Error - {e}')

print("\n[DONE] Pipeline completed successfully!")

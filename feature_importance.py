# feature_importance_extratrees.py
"""
Compute feature importances for ExtraTreesRegressor (multi-output).
- trains ExtraTreesRegressor on the same temporal-split training data you use
- repeats across multiple seeds and returns mean ± std
- computes permutation importance on test set (mean ± std across seeds)
- optionally computes SHAP (if shap is installed)
Outputs:
 - feature_importances_builtin_avg.csv
 - permutation_importance_avg.csv
 - (optional) shap_summary_avg.csv
 - plots: builtin_feature_importance.png, permutation_feature_importance.png, shap_summary.png (if shap)
"""

import os
import glob
import re
import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
import random

# -------------------------
# Config (adjust paths / params as needed)
# -------------------------
NUM_NODES = 10
PATH_PROCESSED = "dataset/processed/10/"     # folder where runX(10)_processed.csv are
GLOB_PATTERN = os.path.join(PATH_PROCESSED, "*_processed.csv")
SEEDS = [42, 123, 2025]   # seeds to average across (change/add/remove as desired)
N_ESTIMATORS = 200
N_JOBS = -1
PERM_N_REPEATS = 10
PERM_RANDOM_STATE = 42
OUT_DIR = "results_feature_importance"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# Helper: process CSV -> train/val/test splits (same scheme you use)
# -------------------------
def build_splits_from_processed_csvs(path_glob=GLOB_PATTERN, num_nodes=NUM_NODES):
    files = sorted(glob.glob(path_glob))
    train_dfs, val_dfs, test_dfs = [], [], []
    for fp in files:
        df = pd.read_csv(fp)
        df = df.sort_values('sample_id').reset_index(drop=True)
        train_dfs.append(df.iloc[0:80].copy())
        val_dfs.append(df.iloc[80:90].copy())
        test_dfs.append(df.iloc[90:100].copy())
    train_combined = pd.concat(train_dfs, ignore_index=True)
    val_combined = pd.concat(val_dfs, ignore_index=True)
    test_combined = pd.concat(test_dfs, ignore_index=True)
    # feature / output column lists (same as your pipeline)
    feature_cols = [c for c in train_combined.columns if c.startswith(('x', 'y', 'z', 't'))]
    output_cols = [c for c in train_combined.columns if c.startswith('output')]
    return train_combined, val_combined, test_combined, feature_cols, output_cols

# -------------------------
# Load and prepare data
# -------------------------
train_df, val_df, test_df, feature_cols, output_cols = build_splits_from_processed_csvs()
print(f"Found {len(feature_cols)} features and {len(output_cols)} outputs per sample.")
print(f"Train/Val/Test sizes: {len(train_df)}/{len(val_df)}/{len(test_df)}")

X_train = train_df[feature_cols].values
y_train = train_df[output_cols].values
X_val = val_df[feature_cols].values
y_val = val_df[output_cols].values
X_test = test_df[feature_cols].values
y_test = test_df[output_cols].values

# Scale features (fit on train only)
feature_scaler = StandardScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_val_scaled = feature_scaler.transform(X_val)
X_test_scaled = feature_scaler.transform(X_test)

# We'll keep original (unscaled) X_test for permutation importance if desired, but permutation on scaled is okay.
# Use scaled features consistently
X_test_for_perm = X_test_scaled
y_test_true = y_test

# -------------------------
# Storage for seed-wise importances
# -------------------------
builtin_imps = []       # list of arrays (n_features,) per seed
perm_imps = []          # list of arrays (n_features,) per seed
perm_imps_std_errors = []  # store std from permutation result if available
seed_metrics = []       # optional: store test MSE per seed

# -------------------------
# Train per-seed ExtraTrees and compute importances
# -------------------------
for seed in SEEDS:
    print(f"\n=== Seed {seed} ===")
    random.seed(seed)
    np.random.seed(seed)

    model = ExtraTreesRegressor(n_estimators=N_ESTIMATORS, random_state=seed, n_jobs=N_JOBS)
    t0 = time.time()
    model.fit(X_train_scaled, y_train)   # ExtraTreesRegressor supports multi-output targets
    t1 = time.time()
    print(f"  Trained ExtraTrees in {t1-t0:.1f}s")

    # Built-in feature importances
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_.astype(float)    # shape (n_features,)
    else:
        # fallback (should not be needed)
        fi = np.zeros(len(feature_cols), dtype=float)
    builtin_imps.append(fi)

    # Evaluate on test (quick sanity)
    y_pred_test = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test_true, y_pred_test)
    seed_metrics.append({'seed': seed, 'test_mse': mse})
    print(f"  Test MSE (seed {seed}): {mse:.6f}")

    # Permutation importance on test set (scoring negative MSE -> higher is better; we'll take -result.importances_mean to get MSE increase)
    # But sklearn permutation_importance returns importances wrt scoring function provided (higher better by default),
    # so for 'neg_mean_squared_error' larger (i.e. less negative) is better; we want to know increase in MSE so we will compute:
    # perm = permutation_importance(model, X_test_for_perm, y_test_true, n_repeats=..., scoring='neg_mean_squared_error')
    perm = permutation_importance(model, X_test_for_perm, y_test_true,
                                  n_repeats=PERM_N_REPEATS, random_state=PERM_RANDOM_STATE, n_jobs=N_JOBS,
                                  scoring='neg_mean_squared_error')
    # permutation_importance returns importances for multioutput by flattening? For multioutput regressor, sklearn computes
    # a single aggregated score per feature (averaging across outputs) when estimator supports multioutput. Good.
    perm_mean = perm.importances_mean  # shape (n_features,)
    perm_std = perm.importances_std
    # Because scoring='neg_mean_squared_error', a more negative value indicates worse; permutation_importance returns array where
    # higher is better relative to baseline; The common interpretation: drop in score. To produce "importance" as increase in MSE:
    # we can take -perm_mean (so higher positive means greater increase in MSE when feature is permuted).
    perm_increase_mse = -perm_mean
    perm_imps.append(perm_increase_mse)
    perm_imps_std_errors.append(perm_std)

# -------------------------
# Aggregate across seeds
# -------------------------
builtin_imps = np.stack(builtin_imps, axis=0)   # (n_seeds, n_features)
perm_imps = np.stack(perm_imps, axis=0)         # (n_seeds, n_features)

builtin_mean = builtin_imps.mean(axis=0)
builtin_std = builtin_imps.std(axis=0, ddof=0)

perm_mean = perm_imps.mean(axis=0)
perm_std = perm_imps.std(axis=0, ddof=0)

# build DataFrame with feature names
feat_names = feature_cols
df_builtin = pd.DataFrame({
    'feature': feat_names,
    'importance_mean': builtin_mean,
    'importance_std': builtin_std
}).sort_values('importance_mean', ascending=False).reset_index(drop=True)

df_perm = pd.DataFrame({
    'feature': feat_names,
    'perm_increase_mse_mean': perm_mean,
    'perm_increase_mse_std': perm_std
}).sort_values('perm_increase_mse_mean', ascending=False).reset_index(drop=True)

# Save CSVs
df_builtin.to_csv(os.path.join(OUT_DIR, "feature_importances_builtin_avg.csv"), index=False)
df_perm.to_csv(os.path.join(OUT_DIR, "permutation_importance_avg.csv"), index=False)
pd.DataFrame(seed_metrics).to_csv(os.path.join(OUT_DIR, "seed_test_mse.csv"), index=False)

print(f"\nSaved builtin and permutation importance CSVs to {OUT_DIR}/")

# -------------------------
# Quick plots
# -------------------------
def plot_bar(df, val_col, err_col, title, out_png, topk=30):
    dfp = df.copy().head(topk)
    plt.figure(figsize=(10, max(4, 0.25 * len(dfp))))
    y = np.arange(len(dfp))
    plt.barh(y, dfp[val_col], xerr=dfp[err_col].values, align='center', ecolor='black', capsize=3)
    plt.yticks(y, dfp['feature'])
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel(val_col)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

plot_bar(df_builtin, 'importance_mean', 'importance_std',
         'ExtraTrees built-in feature importances (mean ± std over seeds)',
         os.path.join(OUT_DIR, 'builtin_feature_importance.png'), topk=50)

plot_bar(df_perm, 'perm_increase_mse_mean', 'perm_increase_mse_std',
         'Permutation importance: avg increase in MSE (mean ± std over seeds)',
         os.path.join(OUT_DIR, 'permutation_importance.png'), topk=50)

print("Saved feature importance plots.")

# -------------------------
# (Optional) SHAP for tree model if available
# -------------------------
try:
    import shap
    print("\nSHAP is available — computing TreeExplainer summary (may take a bit)...")
    # We'll compute SHAP on one representative model trained with the first seed (already trained above).
    # Re-train the model with the first seed to get a fresh instance (or reuse last model if it's from that seed).
    seed0 = SEEDS[0]
    model_shap = ExtraTreesRegressor(n_estimators=N_ESTIMATORS, random_state=seed0, n_jobs=N_JOBS)
    model_shap.fit(X_train_scaled, y_train)
    # shap.TreeExplainer supports multioutput trees; shap_values will be a list per output
    explainer = shap.TreeExplainer(model_shap)
    # Use a sample subset for speed
    sample_idx = np.random.choice(X_test_for_perm.shape[0], size=min(200, X_test_for_perm.shape[0]), replace=False)
    X_sample = X_test_for_perm[sample_idx]
    shap_values = explainer.shap_values(X_sample)  # list of arrays (n_outputs, n_samples, n_features) OR array
    # shap_values may be a list when multioutput; convert to (n_outputs, n_samples, n_features) if needed
    if isinstance(shap_values, list):
        # compute mean(|shap|) per feature across samples and outputs
        shap_abs_avg_per_output = [np.abs(sv).mean(axis=0) for sv in shap_values]  # list of (n_features,)
        shap_abs_avg = np.stack(shap_abs_avg_per_output, axis=0).mean(axis=0)       # average across outputs
    else:
        # shap_values shape (n_samples, n_features) if explainer aggregated; take mean abs
        shap_abs_avg = np.abs(shap_values).mean(axis=0)

    df_shap = pd.DataFrame({'feature': feat_names, 'shap_mean_abs': shap_abs_avg}).sort_values('shap_mean_abs', ascending=False).reset_index(drop=True)
    df_shap.to_csv(os.path.join(OUT_DIR, "shap_tree_summary.csv"), index=False)
    # plot
    plot_bar(df_shap, 'shap_mean_abs', None, 'SHAP mean(|value|) (sampled test set)', os.path.join(OUT_DIR, 'shap_summary.png'), topk=50)
    print("Saved SHAP summary to CSV and plot.")
except Exception as e:
    print("\nSHAP not available or failed. To enable SHAP install `shap` package (pip install shap).")
    # print(e)  # optional debug

# -------------------------
# Print top features summary to stdout
# -------------------------
print("\nTop features by built-in ExtraTrees importance (mean ± std):")
print(df_builtin.head(20).to_string(index=False))

print("\nTop features by permutation importance (avg increase in MSE):")
print(df_perm.head(20).to_string(index=False))

print("\nDone.")

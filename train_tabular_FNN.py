
import re
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import copy
import time
import pandas as pd
import random


# First, let's recreate the processed datasets
def process_log_file(filename, num_nodes=10):
    """Process a single log file and return structured DataFrame"""

    with open(filename, 'r') as file:
        content = file.read()

    # Normalize whitespace
    text = content.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)

    # Extract coordinate and feature data
    coord_pattern = r'\(array\(\[([^\]]+)\]\),\s*array\(\[([^\]]+)\]\),\s*array\(\[([^\]]+)\]\)\)'
    feature_pattern = r'\(\[([^\]]+)\],\s*\[([^\]]+)\],\s*\[([^\]]+)\]\)'

    coord_matches = re.findall(coord_pattern, text)
    feature_matches = re.findall(feature_pattern, text)

    def parse_numeric_array(array_str):
        array_str = re.sub(r'np\.float64\(([^)]+)\)', r'\1', array_str)
        values = [float(x.strip()) for x in array_str.split(',')]
        return values

    processed_samples = []

    for i in range(min(len(coord_matches), len(feature_matches))):
        x_coords = parse_numeric_array(coord_matches[i][0])
        y_coords = parse_numeric_array(coord_matches[i][1])
        z_coords = parse_numeric_array(coord_matches[i][2])

        output_vals = parse_numeric_array(feature_matches[i][0])
        result_vals = parse_numeric_array(feature_matches[i][1])
        additional_features = parse_numeric_array(feature_matches[i][2])

        if len(x_coords) == len(y_coords) == len(z_coords) == len(output_vals) == len(result_vals) == len(
                additional_features) == num_nodes:
            sample_data = {'sample_id': i}

            # Add features
            for j in range(num_nodes):
                sample_data[f'x{j + 1}'] = x_coords[j]
            for j in range(num_nodes):
                sample_data[f'y{j + 1}'] = y_coords[j]
            for j in range(num_nodes):
                sample_data[f'z{j + 1}'] = z_coords[j]
            for j in range(num_nodes):
                sample_data[f't{j + 1}'] = additional_features[j]

            # Add outputs
            for j in range(num_nodes):
                sample_data[f'output{j + 1}'] = output_vals[j]
            for j in range(num_nodes):
                sample_data[f'result{j + 1}'] = result_vals[j]

            processed_samples.append(sample_data)

    return pd.DataFrame(processed_samples)


num_nodes = 10
# Process all log files
log_files = [
    f'run1({num_nodes}).log', f'run2({num_nodes}).log', f'run3({num_nodes}).log', f'run4({num_nodes}).log',
    f'run5({num_nodes}).log',
    f'run6({num_nodes}).log', f'run7({num_nodes}).log', f'run8({num_nodes}).log', f'run9({num_nodes}).log',
    f'run10({num_nodes}).log'
]

print("=== PROCESSING LOG FILES ===")
datasets = {}

PATH_RAW_DATA = "dataset/raw_nodes/10/"
for filename in log_files:
    path = PATH_RAW_DATA + filename
    print(f"Processing {filename}...")
    df = process_log_file(path, num_nodes=num_nodes)
    datasets[filename] = df
    print(f"  ✅ {len(df)} samples processed")

path_processed = 'dataset/processed/10/'
os.makedirs(path_processed, exist_ok=True)
for key in datasets.keys():
    datasets[key].to_csv(path_processed + key.replace('.log', '') + '_processed.csv', index=False)

print(f"\nProcessed {len(datasets)} files successfully")


# Now do temporal splitting
print(f"\n=== TEMPORAL SPLITTING ===")
print("Strategy: First 80 samples → Train, Next 10 → Val, Last 10 → Test")

train_dfs = []
val_dfs = []
test_dfs = []

for filename, df in datasets.items():
    # Ensure temporal order
    df = df.sort_values('sample_id').reset_index(drop=True)

    # Split temporally
    train_df = df.iloc[0:80].copy()
    val_df = df.iloc[80:90].copy()
    test_df = df.iloc[90:100].copy()

    # Add run identifier
    run_num = int(filename.replace('run', '').replace(f'({num_nodes}).log', ''))
    train_df['run_id'] = run_num
    val_df['run_id'] = run_num
    test_df['run_id'] = run_num

    train_dfs.append(train_df)
    val_dfs.append(val_df)
    test_dfs.append(test_df)

    print(f"Run {run_num}: Train {len(train_df)}, Val {len(val_df)}, Test {len(test_df)}")

# Combine all splits
train_combined = pd.concat(train_dfs, ignore_index=True)
val_combined = pd.concat(val_dfs, ignore_index=True)
test_combined = pd.concat(test_dfs, ignore_index=True)

print(f"\n=== FINAL DATASETS ===")
print(f"Training set: {len(train_combined)} samples")
print(f"Validation set: {len(val_combined)} samples")
print(f"Test set: {len(test_combined)} samples")

# Define features and targets (using output as labels, ignoring result)
feature_cols = [col for col in train_combined.columns if col.startswith(('x', 'y', 'z', 't'))]
output_cols = [col for col in train_combined.columns if col.startswith('output')]

print(f"Features: {len(feature_cols)} columns")
print(f"Output targets: {len(output_cols)} columns (ignoring result columns)")

# Prepare data
X_train = train_combined[feature_cols].values
y_train = train_combined[output_cols].values
X_val = val_combined[feature_cols].values
y_val = val_combined[output_cols].values
X_test = test_combined[feature_cols].values
y_test = test_combined[output_cols].values

print(f"\nData shapes:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("\n✅ Data preparation complete!")
print("✅ Features scaled using StandardScaler")
print("✅ Ready for model training")

# Define models to test
print("\n=== DEFINING MODELS ===")
random_state = 123




# ---------- Config ----------
SEEDS = [42, 123, 2025, 31]   # change or extend if you want different seeds
NUM_SEEDS = len(SEEDS)
BATCH_SIZE = 16
N_EPOCHS = 200
PATIENCE = 20
LR = 1e-3
WEIGHT_DECAY = 0
SAVE_CSV = True
OUT_CSV = f"results_tabular_deep/deep_mlp_seed_results_avg_seeds_{num_nodes}_nodes.csv"
DROPOUT = 0.2


# ---------- Simple utility functions ----------
def set_all_seeds(seed: int):
    """Set seeds for python, numpy, torch (cpu + cuda)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch deterministic flags
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # older torch versions:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def compute_metrics_np(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

# ---------- Dataset & Model ----------
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype('float32')
        self.y = y.astype('float32')
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DeepMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256,128,64), output_dim=8, dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# ---------- Training + Eval loops ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

def train_model_torch(model, train_loader, val_loader, criterion, optimizer,
                      n_epochs=100, patience=10, model_name="DeepMLP"):
    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, n_epochs+1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        # validation
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                val_running += criterion(preds, yb).item() * xb.size(0)
        val_loss = val_running / len(val_loader.dataset)

        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            # early stopping
            break

    model.load_state_dict(best_model_wts)
    return model

def evaluate_torch_model(model, loader):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb).cpu().numpy()
            preds.append(out)
            trues.append(yb.numpy())
    preds = np.vstack(preds)
    trues = np.vstack(trues)
    return preds, trues

# ---------- Prepare target scaler once (so comparisons across seeds are apples-to-apples) ----------
# NOTE: expects y_train, y_val, y_test to be defined from your data-prep earlier
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_val_scaled = y_scaler.transform(y_val)
y_test_scaled = y_scaler.transform(y_test)

# Data loaders use num_workers=0 for full reproducibility
train_dataset = TabularDataset(X_train_scaled, y_train_scaled)
val_dataset = TabularDataset(X_val_scaled, y_val_scaled)
test_dataset = TabularDataset(X_test_scaled, y_test_scaled)

# We'll create DataLoaders inside the seed loop using a seeded Generator
# ---------- Run experiments across seeds ----------
all_seed_rows = []

for seed in SEEDS:
    print(f"\n--- RUNNING SEED {seed} ---")
    set_all_seeds(seed)

    # DataLoader generators for deterministic shuffle
    g_train = torch.Generator()
    g_train.manual_seed(seed)
    g_val = torch.Generator()
    g_val.manual_seed(seed + 7)
    g_test = torch.Generator()
    g_test.manual_seed(seed + 13)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, generator=g_train)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, generator=g_val)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, generator=g_test)

    # Re-init model for each seed (weights initialization depends on seed)
    input_dim = X_train_scaled.shape[1]
    output_dim = y_train.shape[1]
    model = DeepMLP(input_dim=input_dim, hidden_dims=(256,128,64), output_dim=output_dim, dropout=DROPOUT)

    # Re-seed again for weight init order (some backends read RNG during init)
    set_all_seeds(seed+1)

    # optimizer & loss
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    t0 = time.time()
    model_trained = train_model_torch(model, train_loader, val_loader,
                                      criterion, optimizer, n_epochs=N_EPOCHS,
                                      patience=PATIENCE, model_name=f"DeepMLP_seed{seed}")
    t1 = time.time()
    print(f"Seed {seed} training time: {t1-t0:.1f}s")

    # Evaluate (predictions are on scaled target space)
    y_train_pred_scaled, y_train_true_scaled = evaluate_torch_model(model_trained, train_loader)
    y_val_pred_scaled, y_val_true_scaled = evaluate_torch_model(model_trained, val_loader)
    y_test_pred_scaled, y_test_true_scaled = evaluate_torch_model(model_trained, test_loader)

    # inverse transform to original target scale
    y_train_pred = y_scaler.inverse_transform(y_train_pred_scaled)
    y_val_pred = y_scaler.inverse_transform(y_val_pred_scaled)
    y_test_pred = y_scaler.inverse_transform(y_test_pred_scaled)

    y_train_true = y_scaler.inverse_transform(y_train_true_scaled)
    y_val_true = y_scaler.inverse_transform(y_val_true_scaled)
    y_test_true = y_scaler.inverse_transform(y_test_true_scaled)

    # compute metrics
    train_metrics = compute_metrics_np(y_train_true, y_train_pred)
    val_metrics = compute_metrics_np(y_val_true, y_val_pred)
    test_metrics = compute_metrics_np(y_test_true, y_test_pred)

    print(f"Seed {seed} metrics -> Train R2: {train_metrics['r2']:.4f}, Val R2: {val_metrics['r2']:.4f}, Test R2: {test_metrics['r2']:.4f}")

    # collect rows
    for ds_name, metrics in [('train', train_metrics), ('val', val_metrics), ('test', test_metrics)]:
        row = {
            'seed': seed,
            'model': 'DeepMLP',
            'dataset': ds_name,
            'mse': metrics['mse'],
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'r2': metrics['r2']
        }
        all_seed_rows.append(row)

# ---------- Summarize across seeds ----------
df_seeds = pd.DataFrame(all_seed_rows)

# compute mean and std for each dataset
summary_rows = []
for ds in ['train', 'val', 'test']:
    sub = df_seeds[df_seeds['dataset'] == ds]
    summary = {
        'model': 'DeepMLP',
        'dataset': ds,
        'mse_mean': sub['mse'].mean(),
        'mse_std': sub['mse'].std(),
        'rmse_mean': sub['rmse'].mean(),
        'rmse_std': sub['rmse'].std(),
        'mae_mean': sub['mae'].mean(),
        'mae_std': sub['mae'].std(),
        'r2_mean': sub['r2'].mean(),
        'r2_std': sub['r2'].std(),
        'n_seeds': len(sub)
    }
    summary_rows.append(summary)

summary_df = pd.DataFrame(summary_rows)

print("\n=== PER-SEED RESULTS ===")
print(df_seeds.sort_values(['seed','dataset']).reset_index(drop=True))

print("\n=== SUMMARY ACROSS SEEDS (mean ± std) ===")
for _, r in summary_df.iterrows():
    print(f"{r['dataset'].upper()}: R2 = {r['r2_mean']:.4f} ± {r['r2_std']:.4f} | RMSE = {r['rmse_mean']:.4f} ± {r['rmse_std']:.4f}")

# save CSVs
if SAVE_CSV:
    df_seeds.to_csv(OUT_CSV.replace('.csv', '_per_seed.csv'), index=False)
    summary_df.to_csv(OUT_CSV.replace('.csv', '_summary.csv'), index=False)
    print(f"Saved per-seed results_sequential to {OUT_CSV.replace('.csv','_per_seed.csv')}")
    print(f"Saved summary results_sequential to {OUT_CSV.replace('.csv','_summary.csv')}")


# done
print("All seeds finished.")

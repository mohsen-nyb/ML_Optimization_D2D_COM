# sequential_pipeline_updated.py
# Extended: teacher-forced + autoregressive (rollout) per-horizon evaluation
import os
import glob
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import copy
import time
import random

# -------------------------
# CONFIG
# -------------------------
NUM_NODES = 10
PATH_PROCESSED = f"dataset/processed/{NUM_NODES}/"
GLOB_PATTERN = os.path.join(PATH_PROCESSED, "*_processed.csv")

NODE_FEATURES = ['x', 'y', 'z', 't']   # features per node present in CSV
WINDOW_SIZE = 10                       # configurable sequence length (you can change)
BATCH_SIZE = 64
N_EPOCHS = 200
PATIENCE = 20
LR = 1e-3
WEIGHT_DECAY = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 52   # set to None to disable deterministic seeds
NUM_WORKERS = 0  # reproducible
WARMUP_T = 89     # last index of warmup (we will rollout from 90..99)
TEST_RANGE = list(range(90, 100))  # t=90..99 inclusive

# -------------------------
# UTIL: metrics
# -------------------------
def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

# -------------------------
# Preprocessing: build sequence samples
# -------------------------
def build_seq_samples_from_df(df, window_size=WINDOW_SIZE):
    df = df.sort_values('sample_id').reset_index(drop=True)
    T = len(df)
    feat_per_node = len(NODE_FEATURES) + 1  # +1 for prev_output
    seq_samples = []

    xs = df[[f'x{i+1}' for i in range(NUM_NODES)]].values
    ys = df[[f'y{i+1}' for i in range(NUM_NODES)]].values
    zs = df[[f'z{i+1}' for i in range(NUM_NODES)]].values
    ts = df[[f't{i+1}' for i in range(NUM_NODES)]].values
    outs = df[[f'output{i+1}' for i in range(NUM_NODES)]].values

    for t in range(T):
        seq_feats = np.zeros((window_size, NUM_NODES, feat_per_node), dtype=np.float32)
        for idx_in_window in range(window_size):
            k = t - (window_size - 1) + idx_in_window
            if k < 0:
                continue
            node_feats = np.stack([xs[k], ys[k], zs[k], ts[k]], axis=1)
            if k - 1 >= 0:
                prev_out = outs[k-1]
            else:
                prev_out = np.zeros((NUM_NODES,), dtype=np.float32)
            prev_out = prev_out.reshape(NUM_NODES, 1)
            seq_feats[idx_in_window] = np.concatenate([node_feats, prev_out], axis=1)
        y_target = outs[t].astype(np.float32)
        seq_feats_flat = seq_feats.reshape(window_size, NUM_NODES * feat_per_node).astype(np.float32)
        seq_samples.append((t, seq_feats_flat, y_target))
    return seq_samples

def build_dataset_from_all_runs(path_glob=GLOB_PATTERN, window_size=WINDOW_SIZE):
    files = sorted(glob.glob(path_glob))
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []

    for fp in files:
        df = pd.read_csv(fp)
        samples = build_seq_samples_from_df(df, window_size=window_size)
        for (t_idx, X_seq, y_t) in samples:
            if 0 <= t_idx <= 79:
                X_train_list.append(X_seq); y_train_list.append(y_t)
            elif 80 <= t_idx <= 89:
                X_val_list.append(X_seq); y_val_list.append(y_t)
            elif 90 <= t_idx <= 99:
                X_test_list.append(X_seq); y_test_list.append(y_t)

    X_train = np.stack(X_train_list, axis=0) if len(X_train_list) > 0 else np.zeros((0, window_size, NUM_NODES*(len(NODE_FEATURES)+1)))
    y_train = np.stack(y_train_list, axis=0) if len(y_train_list) > 0 else np.zeros((0, NUM_NODES))
    X_val = np.stack(X_val_list, axis=0) if len(X_val_list) > 0 else np.zeros((0, window_size, NUM_NODES*(len(NODE_FEATURES)+1)))
    y_val = np.stack(y_val_list, axis=0) if len(y_val_list) > 0 else np.zeros((0, NUM_NODES))
    X_test = np.stack(X_test_list, axis=0) if len(X_test_list) > 0 else np.zeros((0, window_size, NUM_NODES*(len(NODE_FEATURES)+1)))
    y_test = np.stack(y_test_list, axis=0) if len(y_test_list) > 0 else np.zeros((0, NUM_NODES))

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }

# -------------------------
# Build datasets (reads CSVs)
# -------------------------
print("Building sequence datasets from processed CSVs...")
data = build_dataset_from_all_runs(window_size=WINDOW_SIZE)
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']

print(f"Shapes: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"        X_val  ={X_val.shape}, y_val  ={y_val.shape}")
print(f"        X_test ={X_test.shape}, y_test ={y_test.shape}")

# -------------------------
# Scale features and targets (fit on train only)
# -------------------------
N_train = X_train.shape[0]
seq_len = X_train.shape[1]
feat_dim = X_train.shape[2]  # NUM_NODES * feat_per_node

feature_scaler = StandardScaler()
feature_scaler.fit(X_train.reshape(-1, feat_dim))
X_train_scaled = feature_scaler.transform(X_train.reshape(-1, feat_dim)).reshape(N_train, seq_len, feat_dim)
X_val_scaled = feature_scaler.transform(X_val.reshape(-1, feat_dim)).reshape(X_val.shape[0], seq_len, feat_dim)
X_test_scaled = feature_scaler.transform(X_test.reshape(-1, feat_dim)).reshape(X_test.shape[0], seq_len, feat_dim)

target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train)
y_val_scaled = target_scaler.transform(y_val)
y_test_scaled = target_scaler.transform(y_test)

# -------------------------
# PyTorch Dataset wrappers
# -------------------------
class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = SeqDataset(X_train_scaled, y_train_scaled)
val_ds = SeqDataset(X_val_scaled, y_val_scaled)
test_ds = SeqDataset(X_test_scaled, y_test_scaled)

def get_loaders(batch_size=BATCH_SIZE, seed=SEED):
    if seed is not None:
        g = torch.Generator(); g.manual_seed(seed)
    else:
        g = None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, val_loader, test_loader

# -------------------------
# Models: LSTM & Transformer
# -------------------------
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, bidirectional=False, dropout=0.1, output_dim=NUM_NODES):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional, dropout=dropout)
        last_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(last_dim, last_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(last_dim//2, output_dim)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)

class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1, output_dim=NUM_NODES):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, output_dim)
        )
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
    def forward(self, x):
        x = self.input_proj(x) + self.pos_emb.to(x.device)
        out = self.encoder(x)
        last = out[:, -1, :]
        return self.head(last)

# -------------------------
# Training & Evaluation utilities
# -------------------------
def train_one_model(model, train_loader, val_loader, n_epochs=N_EPOCHS, patience=PATIENCE, lr=LR, wd=WEIGHT_DECAY, device=DEVICE):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_model = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    no_imp = 0

    for epoch in range(1, n_epochs+1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
        train_loss = running / len(train_loader.dataset)

        model.eval()
        vrun = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                preds = model(xb)
                vrun += criterion(preds, yb).item() * xb.size(0)
        val_loss = vrun / len(val_loader.dataset)

        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            no_imp = 0
        else:
            no_imp += 1

        if no_imp >= patience:
            break
        if epoch == 1 or epoch % 10 == 0:
            print(f" Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

    model.load_state_dict(best_model)
    return model

def evaluate_model_on_loader(model, loader, device=DEVICE):
    model.eval()
    preds_list = []; trues_list = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb).cpu().numpy()
            preds_list.append(out); trues_list.append(yb.numpy())
    preds = np.vstack(preds_list); trues = np.vstack(trues_list)
    return preds, trues

def predict_rollout_for_run(
    model, df_run, feature_scaler, target_scaler,
    window_size=WINDOW_SIZE,
    warmup_t=WARMUP_T, mode='rollout', device=DEVICE):
    """
    Predict autoregressively for t in TEST_RANGE for a single run.
    mode: 'rollout' -> after warmup use preds as prev_output; 'teacher' -> always use true prev outputs.
    Returns:
      preds_all: (T, NUM_NODES) predictions for all t in run (we fill for t=0..T-1)
      true_all: (T, NUM_NODES) ground-truth outputs
    """
    model.eval()
    model.to(device)
    df = df_run.sort_values('sample_id').reset_index(drop=True)
    xs = df[[f'x{i+1}' for i in range(NUM_NODES)]].values
    ys = df[[f'y{i+1}' for i in range(NUM_NODES)]].values
    zs = df[[f'z{i+1}' for i in range(NUM_NODES)]].values
    ts = df[[f't{i+1}' for i in range(NUM_NODES)]].values
    outs = df[[f'output{i+1}' for i in range(NUM_NODES)]].values
    T = len(df)
    feat_per_node = len(NODE_FEATURES) + 1
    feat_dim = NUM_NODES * feat_per_node

    preds_all = np.zeros((T, NUM_NODES), dtype=np.float32)
    # buffer gets initialized with GT only ONCE per run
    rollout_prev_outputs = outs.copy()

    for t in range(T):
        window_frames = np.zeros((window_size, NUM_NODES, feat_per_node), dtype=np.float32)
        for w_idx in range(window_size):
            k = t - (window_size - 1) + w_idx
            if k < 0:
                continue
            node_feats = np.stack([xs[k], ys[k], zs[k], ts[k]], axis=1)
            if k - 1 >= 0:
                if mode == 'teacher':
                    prev_out = outs[k - 1]
                elif k - 1 <= warmup_t:
                    prev_out = outs[k - 1]
                else:
                    prev_out = rollout_prev_outputs[k - 1]
            else:
                prev_out = np.zeros((NUM_NODES,), dtype=np.float32)
            window_frames[w_idx] = np.concatenate([node_feats, prev_out.reshape(NUM_NODES, 1)], axis=1)
        window_flat = window_frames.reshape(window_size, feat_dim)
        window_scaled = feature_scaler.transform(window_flat).reshape(1, window_size, feat_dim).astype(np.float32)
        x_tensor = torch.tensor(window_scaled, dtype=torch.float32).to(device)
        with torch.no_grad():
            y_pred_scaled = model(x_tensor).cpu().numpy()[0]
        y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(1, -1))[0]
        preds_all[t] = y_pred.astype(np.float32)
        # Update buffer for t > warmup only ONCE per run - strict autoregressive!
        if mode == 'rollout' and t > warmup_t:
            rollout_prev_outputs[t] = y_pred.astype(np.float32)

    return preds_all, outs




# -------------------------
# Run training for both models
# -------------------------
if SEED is not None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

train_loader, val_loader, test_loader = get_loaders(batch_size=BATCH_SIZE, seed=SEED)
input_dim = X_train_scaled.shape[2]
print("Input dim per timestep:", input_dim)

results = []

# LSTM
print("\n=== TRAINING LSTM REGRESSOR ===")
lstm = LSTMRegressor(input_dim=input_dim, hidden_dim=128, num_layers=2, bidirectional=False, dropout=0.1)
t0 = time.time()
lstm_trained = train_one_model(lstm, train_loader, val_loader)
t1 = time.time()
print(f"LSTM training time: {t1-t0:.1f}s")

# teacher-forced evaluation using loader (this matches model inputs that include true prev outputs)
y_train_pred_s, y_train_true_s = evaluate_model_on_loader(lstm_trained, train_loader)
y_val_pred_s,   y_val_true_s   = evaluate_model_on_loader(lstm_trained, val_loader)
y_test_pred_s,  y_test_true_s  = evaluate_model_on_loader(lstm_trained, test_loader)

y_train_pred = target_scaler.inverse_transform(y_train_pred_s)
y_val_pred = target_scaler.inverse_transform(y_val_pred_s)
y_test_pred = target_scaler.inverse_transform(y_test_pred_s)

y_train_true = target_scaler.inverse_transform(y_train_true_s)
y_val_true = target_scaler.inverse_transform(y_val_true_s)
y_test_true = target_scaler.inverse_transform(y_test_true_s)

train_metrics = compute_metrics(y_train_true, y_train_pred)
val_metrics = compute_metrics(y_val_true, y_val_pred)
test_metrics = compute_metrics(y_test_true, y_test_pred)

print("LSTM Teacher-forced Metrics: train R2 {:.4f} | val R2 {:.4f} | test R2 {:.4f}".format(train_metrics['r2'], val_metrics['r2'], test_metrics['r2']))
results.append({'model':'LSTM', 'train':train_metrics, 'val':val_metrics, 'test':test_metrics})

# compute rollouts across runs for per-horizon metrics
all_files = sorted(glob.glob(GLOB_PATTERN))
# DataFrames to collect across runs per horizon
horizon_rows = []
final_rows = []

for fp in all_files:
    df_run = pd.read_csv(fp)
    # teacher-forced predictions (consistent with dataset inputs) per-run for t=90..99
    preds_teacher, outs_true = predict_rollout_for_run(lstm_trained, df_run, feature_scaler, target_scaler,
                                                       window_size=WINDOW_SIZE, warmup_t=WARMUP_T, mode='teacher', device=DEVICE)
    preds_rollout, _ = predict_rollout_for_run(lstm_trained, df_run, feature_scaler, target_scaler,
                                               window_size=WINDOW_SIZE, warmup_t=WARMUP_T, mode='rollout', device=DEVICE)
    # collect per-horizon (h=1..len(TEST_RANGE))
    for h_idx, t in enumerate(TEST_RANGE, start=1):
        # teacher-forced
        y_true_t = outs_true[t]          # (num_nodes,)
        y_teacher_t = preds_teacher[t]   # (num_nodes,)
        y_roll_t = preds_rollout[t]      # (num_nodes,)
        # metrics flattened across nodes (you can also store per-node)
        m_teacher = compute_metrics(y_true_t.flatten(), y_teacher_t.flatten())
        m_roll = compute_metrics(y_true_t.flatten(), y_roll_t.flatten())
        horizon_rows.append({
            'runfile': os.path.basename(fp),
            'model': 'LSTM',
            'horizon': h_idx,
            't_index': t,
            'teacher_mse': m_teacher['mse'], 'teacher_rmse': m_teacher['rmse'], 'teacher_mae': m_teacher['mae'], 'teacher_r2': m_teacher['r2'],
            'roll_mse': m_roll['mse'], 'roll_rmse': m_roll['rmse'], 'roll_mae': m_roll['mae'], 'roll_r2': m_roll['r2']
        })
    # final aggregated test-horizon (t=90..99) flattened
    y_true_block = outs_true[TEST_RANGE, :].reshape(-1, NUM_NODES)
    y_roll_block = preds_rollout[TEST_RANGE, :].reshape(-1, NUM_NODES)
    final_m = compute_metrics(y_true_block.flatten(), y_roll_block.flatten())
    final_rows.append({'runfile': os.path.basename(fp), 'model':'LSTM', **final_m})

horizon_df = pd.DataFrame(horizon_rows)
horizon_summary = horizon_df.groupby('horizon').agg({
    'teacher_mse':'mean','teacher_rmse':'mean','teacher_mae':'mean','teacher_r2':'mean',
    'roll_mse':'mean','roll_rmse':'mean','roll_mae':'mean','roll_r2':'mean'
}).reset_index()
horizon_summary.to_csv("results_sequential/sequential_rollout_per_horizon_LSTM.csv", index=False)
pd.DataFrame(final_rows).to_csv("results_sequential/sequential_rollout_summary_LSTM.csv", index=False)
print("\nSaved LSTM per-horizon and summary CSVs")


# Save ONE average metrics row (not per run) for sequential_rollout_summary_LSTM.csv
final_df = pd.DataFrame(final_rows)
agg = final_df[['mse', 'rmse', 'mae', 'r2']].mean()
summary_dict = {'model': 'LSTM'}
summary_dict.update(agg.to_dict())
summary_df = pd.DataFrame([summary_dict])
summary_df.to_csv("results_sequential/sequential_rollout_summary_LSTM.csv", index=False)
print("Saved aggregated LSTM summary CSV")

# -------------------------
# Transformer
# -------------------------
print("\n=== TRAINING TRANSFORMER REGRESSOR ===")
transformer = TransformerRegressor(input_dim=input_dim, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1)
t0 = time.time()
trans_trained = train_one_model(transformer, train_loader, val_loader)
t1 = time.time()
print(f"Transformer training time: {t1-t0:.1f}s")

y_train_pred_s, y_train_true_s = evaluate_model_on_loader(trans_trained, train_loader)
y_val_pred_s,   y_val_true_s   = evaluate_model_on_loader(trans_trained, val_loader)
y_test_pred_s,  y_test_true_s  = evaluate_model_on_loader(trans_trained, test_loader)

y_train_pred = target_scaler.inverse_transform(y_train_pred_s)
y_val_pred = target_scaler.inverse_transform(y_val_pred_s)
y_test_pred = target_scaler.inverse_transform(y_test_pred_s)

y_train_true = target_scaler.inverse_transform(y_train_true_s)
y_val_true = target_scaler.inverse_transform(y_val_true_s)
y_test_true = target_scaler.inverse_transform(y_test_true_s)

train_metrics = compute_metrics(y_train_true, y_train_pred)
val_metrics = compute_metrics(y_val_true, y_val_pred)
test_metrics = compute_metrics(y_test_true, y_test_pred)

print("Transformer Teacher-forced Metrics: train R2 {:.4f} | val R2 {:.4f} | test R2 {:.4f}".format(train_metrics['r2'], val_metrics['r2'], test_metrics['r2']))
results.append({'model':'Transformer', 'train':train_metrics, 'val':val_metrics, 'test':test_metrics})

# rollout per-horizon for transformer
horizon_rows = []
final_rows = []
for fp in all_files:
    df_run = pd.read_csv(fp)
    preds_teacher, outs_true = predict_rollout_for_run(trans_trained, df_run, feature_scaler, target_scaler,
                                                       window_size=WINDOW_SIZE, warmup_t=WARMUP_T, mode='teacher', device=DEVICE)
    preds_rollout, _ = predict_rollout_for_run(trans_trained, df_run, feature_scaler, target_scaler,
                                               window_size=WINDOW_SIZE, warmup_t=WARMUP_T, mode='rollout', device=DEVICE)
    for h_idx, t in enumerate(TEST_RANGE, start=1):
        y_true_t = outs_true[t]; y_teacher_t = preds_teacher[t]; y_roll_t = preds_rollout[t]
        m_teacher = compute_metrics(y_true_t.flatten(), y_teacher_t.flatten())
        m_roll = compute_metrics(y_true_t.flatten(), y_roll_t.flatten())
        horizon_rows.append({
            'runfile': os.path.basename(fp),
            'model': 'Transformer',
            'horizon': h_idx,
            't_index': t,
            'teacher_mse': m_teacher['mse'], 'teacher_rmse': m_teacher['rmse'], 'teacher_mae': m_teacher['mae'], 'teacher_r2': m_teacher['r2'],
            'roll_mse': m_roll['mse'], 'roll_rmse': m_roll['rmse'], 'roll_mae': m_roll['mae'], 'roll_r2': m_roll['r2']
        })
    y_true_block = outs_true[TEST_RANGE, :].reshape(-1, NUM_NODES)
    y_roll_block = preds_rollout[TEST_RANGE, :].reshape(-1, NUM_NODES)
    final_m = compute_metrics(y_true_block.flatten(), y_roll_block.flatten())
    final_rows.append({'runfile': os.path.basename(fp), 'model':'Transformer', **final_m})

horizon_df = pd.DataFrame(horizon_rows)
horizon_summary = horizon_df.groupby('horizon').agg({
    'teacher_mse':'mean','teacher_rmse':'mean','teacher_mae':'mean','teacher_r2':'mean',
    'roll_mse':'mean','roll_rmse':'mean','roll_mae':'mean','roll_r2':'mean'
}).reset_index()
horizon_summary.to_csv("results_sequential/sequential_rollout_per_horizon_Transformer.csv", index=False)
pd.DataFrame(final_rows).to_csv("results_sequential/sequential_rollout_summary_Transformer.csv", index=False)
print("\nSaved Transformer per-horizon and summary CSVs")
final_df = pd.DataFrame(final_rows)
agg = final_df[['mse', 'rmse', 'mae', 'r2']].mean()
summary_dict = {'model': 'Transformer'}
summary_dict.update(agg.to_dict())
summary_df = pd.DataFrame([summary_dict])
summary_df.to_csv("results_sequential/sequential_rollout_summary_Transformer.csv", index=False)
print("Saved aggregated Transformer summary CSV")



# -------------------------------
# Save teacher-forced aggregated results_sequential (train/val/test)
# -------------------------------
out_df_rows = []
for r in results:
    for ds in ['train', 'val', 'test']:
        row = {'model': r['model'], 'dataset': ds}
        row.update(r[ds])
        out_df_rows.append(row)
out_df = pd.DataFrame(out_df_rows)
out_df.to_csv("results_sequential/sequential_model_results.csv", index=False)
print("\nSaved sequential_model_results.csv")

# ======================================================
# CHARTS: Plot per-horizon metrics (LSTM vs Transformer)
# ======================================================
import matplotlib.pyplot as plt

def plot_per_horizon_chart(csv_path, model_name):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 6))
    plt.plot(df['horizon'], df['teacher_mse'], label=f'{model_name} Teacher MSE', marker='o')
    plt.plot(df['horizon'], df['roll_mse'], label=f'{model_name} Rollout MSE', marker='x')
    plt.xlabel('Forecast Horizon (t steps ahead)')
    plt.ylabel('MSE')
    plt.title(f'{model_name} MSE per Forecast Horizon')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results_sequential/{model_name}_mse_per_horizon.png')
    plt.close()

    # Repeat for RMSE, MAE, R2
    for metric in ['rmse', 'mae', 'r2']:
        plt.figure(figsize=(10,6))
        plt.plot(df['horizon'], df[f'teacher_{metric}'], label=f'{model_name} Teacher {metric.upper()}', marker='o')
        plt.plot(df['horizon'], df[f'roll_{metric}'], label=f'{model_name} Rollout {metric.upper()}', marker='x')
        plt.xlabel('Forecast Horizon (t steps ahead)')
        plt.ylabel(metric.upper())
        plt.title(f'{model_name} {metric.upper()} per Forecast Horizon')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'results_sequential/{model_name}_{metric}_per_horizon.png')
        plt.close()

plot_per_horizon_chart("results_sequential/sequential_rollout_per_horizon_LSTM.csv", "LSTM")
plot_per_horizon_chart("results_sequential/sequential_rollout_per_horizon_Transformer.csv", "Transformer")

print("Finished and saved all charts.")
print("Done.")

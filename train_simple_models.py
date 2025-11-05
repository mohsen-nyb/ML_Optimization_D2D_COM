import pandas as pd
import numpy as np
import re
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error



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
        
        if len(x_coords) == len(y_coords) == len(z_coords) == len(output_vals) == len(result_vals) == len(additional_features) == num_nodes:
            sample_data = {'sample_id': i}
            
            # Add features
            for j in range(num_nodes):
                sample_data[f'x{j+1}'] = x_coords[j]
            for j in range(num_nodes):
                sample_data[f'y{j+1}'] = y_coords[j]
            for j in range(num_nodes):
                sample_data[f'z{j+1}'] = z_coords[j]
            for j in range(num_nodes):
                sample_data[f't{j+1}'] = additional_features[j]
                
            # Add outputs
            for j in range(num_nodes):
                sample_data[f'output{j+1}'] = output_vals[j]
            for j in range(num_nodes):
                sample_data[f'result{j+1}'] = result_vals[j]
                
            processed_samples.append(sample_data)
    
    return pd.DataFrame(processed_samples)


num_nodes=10
# Process all log files
log_files = [
    f'run1({num_nodes}).log', f'run2({num_nodes}).log', f'run3({num_nodes}).log', f'run4({num_nodes}).log', f'run5({num_nodes}).log',
    f'run6({num_nodes}).log', f'run7({num_nodes}).log', f'run8({num_nodes}).log', f'run9({num_nodes}).log', f'run10({num_nodes}).log'
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
    datasets[key].to_csv(path_processed+ key.replace('.log', '') + '_processed.csv', index=False)



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
random_state=42

models = {
    'Linear Regression': MultiOutputRegressor(LinearRegression()),
    'Ridge Regression': MultiOutputRegressor(Ridge(alpha=1.0, random_state=random_state)),
    'ElasticNet': MultiOutputRegressor(ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state)),
    'Random Forest': MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)),
    'Extra Trees': MultiOutputRegressor(ExtraTreesRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)),
    'Gradient Boosting': MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, random_state=random_state)),
    'K-Nearest Neighbors': MultiOutputRegressor(KNeighborsRegressor(n_neighbors=5)),
    'Support Vector Regression': MultiOutputRegressor(SVR(kernel='rbf', gamma='scale')),
    # 'Neural Network (Small)': MultiOutputRegressor(
    #     MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=random_state)),
    # 'Neural Network (Large)': MultiOutputRegressor(
    #     MLPRegressor(hidden_layer_sizes=(200, 100, 50), max_iter=1000, random_state=random_state))
}

print(f"Defined {len(models)} models to test")
print("Models:", list(models.keys()))


# Function to evaluate models
def evaluate_model(y_true, y_pred, dataset_name):
    """Calculate comprehensive metrics for multi-output regression"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        'dataset': dataset_name,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


# -------------------------
# Train and evaluate all models across multiple seeds
# -------------------------
print("\n=== TRAINING AND EVALUATION WITH MULTIPLE SEEDS ===")
seeds = [42, 123, 2025, 31]  # Example: 3 seeds
all_results = []

for seed in seeds:
    print(f"\n=== Seed: {seed} ===")
    np.random.seed(seed)

    for model_name, model_base in models.items():
        try:
            # Re-initialize model with current seed if possible
            if hasattr(model_base.estimator, 'random_state'):
                model = MultiOutputRegressor(type(model_base.estimator)(**model_base.estimator.get_params()))
                if hasattr(model.estimator, 'random_state'):
                    model.estimator.random_state = seed
            else:
                model = MultiOutputRegressor(type(model_base.estimator)(**model_base.estimator.get_params()))

            # Train
            model.fit(X_train_scaled, y_train)

            # Predict
            y_train_pred = model.predict(X_train_scaled)
            y_val_pred = model.predict(X_val_scaled)
            y_test_pred = model.predict(X_test_scaled)

            # Evaluate
            train_metrics = evaluate_model(y_train, y_train_pred, 'train')
            val_metrics = evaluate_model(y_val, y_val_pred, 'val')
            test_metrics = evaluate_model(y_test, y_test_pred, 'test')

            # Store metrics with seed info
            all_results.append({
                'model': model_name,
                'seed': seed,
                'train_mse': train_metrics['mse'],
                'train_rmse': train_metrics['rmse'],
                'train_mae': train_metrics['mae'],
                'train_r2': train_metrics['r2'],
                'val_mse': val_metrics['mse'],
                'val_rmse': val_metrics['rmse'],
                'val_mae': val_metrics['mae'],
                'val_r2': val_metrics['r2'],
                'test_mse': test_metrics['mse'],
                'test_rmse': test_metrics['rmse'],
                'test_mae': test_metrics['mae'],
                'test_r2': test_metrics['r2'],
            })

        except Exception as e:
            print(f"  ❌ Error training {model_name} with seed {seed}: {str(e)}")
            continue

# Convert to DataFrame
results_df = pd.DataFrame(all_results)

# Compute mean and std across seeds
avg_std_results = results_df.groupby('model').agg(['mean', 'std']).reset_index()

# Flatten multi-level columns
avg_std_results.columns = ['_'.join(col).strip('_') for col in avg_std_results.columns.values]

# Save results
path = 'results_simple_models/'
os.makedirs(path, exist_ok=True)
avg_std_results.to_csv(path + f'results_simple_models_{num_nodes}_5seeds_avg_std.csv', index=False)

print("\n=== AVERAGED RESULTS ACROSS 3 SEEDS ===")
print(avg_std_results)
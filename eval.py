# Create comprehensive results_sequential analysis
import pandas as pd

# Pivot the results_sequential for better analysis
results_pivot = results_df.pivot(index='model', columns='dataset', values=['r2', 'rmse', 'mae'])

print("=== COMPREHENSIVE RESULTS ANALYSIS ===")
print("\n1. R² SCORES (Higher is better)")
r2_results = results_pivot['r2'].round(4)
r2_results = r2_results.sort_values('test', ascending=False)
print(r2_results)

print("\n2. RMSE SCORES (Lower is better)")
rmse_results = results_pivot['rmse'].round(4)
rmse_results = rmse_results.sort_values('test', ascending=True)
print(rmse_results)

print("\n3. MAE SCORES (Lower is better)")
mae_results = results_pivot['mae'].round(4)
mae_results = mae_results.sort_values('test', ascending=True)
print(mae_results)

# Identify best models
print("\n=== TOP PERFORMING MODELS ===")

# Best models by test R²
test_r2_sorted = r2_results.sort_values('test', ascending=False)
print("\nTOP 5 MODELS BY TEST R² SCORE:")
for i, (model, row) in enumerate(test_r2_sorted.head().iterrows()):
    print(f"{i+1:2d}. {model:25s} - Test R²: {row['test']:.4f}, Val R²: {row['val']:.4f}")

# Check for overfitting (large gap between train and test)
print("\nOVERFITTING ANALYSIS (Train R² - Test R²):")
overfitting = r2_results['train'] - r2_results['test']
overfitting_sorted = overfitting.sort_values(ascending=True)
for model, gap in overfitting_sorted.items():
    status = "✅ Good" if gap < 0.3 else "⚠️  Moderate" if gap < 0.5 else "❌ High"
    print(f"{model:25s}: {gap:6.3f} {status}")

# Model recommendations
print("\n=== MODEL RECOMMENDATIONS ===")

# Best overall performance
best_test_r2 = test_r2_sorted.iloc[0]
best_model_name = test_r2_sorted.index[0]
print(f"🏆 BEST OVERALL: {best_model_name}")
print(f"   Test R²: {best_test_r2['test']:.4f}")
print(f"   Val R²:  {best_test_r2['val']:.4f}")
print(f"   Train R²: {best_test_r2['train']:.4f}")

# Best with low overfitting
low_overfitting = overfitting_sorted[overfitting_sorted < 0.3]
if len(low_overfitting) > 0:
    best_balanced = low_overfitting.index[0]
    best_balanced_scores = r2_results.loc[best_balanced]
    print(f"\n🎯 BEST BALANCED (Low Overfitting): {best_balanced}")
    print(f"   Test R²: {best_balanced_scores['test']:.4f}")
    print(f"   Val R²:  {best_balanced_scores['val']:.4f}")
    print(f"   Train R²: {best_balanced_scores['train']:.4f}")
    print(f"   Overfitting gap: {overfitting[best_balanced]:.3f}")

# Save results_sequential
results_df.to_csv('ml_model_results.csv', index=False)
r2_results.to_csv('model_r2_scores.csv')
print(f"\n✅ Results saved to 'ml_model_results.csv' and 'model_r2_scores.csv'")

# Feature importance analysis for best tree-based model
tree_models = ['Random Forest', 'Extra Trees', 'Gradient Boosting']
best_tree_model = None
best_tree_r2 = -np.inf

for model_name in tree_models:
    if model_name in r2_results.index:
        test_r2 = r2_results.loc[model_name, 'test']
        if test_r2 > best_tree_r2:
            best_tree_r2 = test_r2
            best_tree_model = model_name

if best_tree_model:
    print(f"\n=== FEATURE IMPORTANCE ({best_tree_model}) ===")
    
    # Get the trained model
    trained_model = models[best_tree_model]
    
    # Get feature importance (average across all output estimators)
    importances = []
    for estimator in trained_model.estimators_:
        importances.append(estimator.feature_importances_)
    
    avg_importance = np.mean(importances, axis=0)
    
    # Create feature importance DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': avg_importance
    }).sort_values('importance', ascending=False)
    
    print("\nTOP 10 MOST IMPORTANT FEATURES:")
    for i, (_, row) in enumerate(feature_importance_df.head(10).iterrows()):
        print(f"{i+1:2d}. {row['feature']:6s}: {row['importance']:.4f}")
    
    # Group by feature type
    feature_types = {
        'x': [f'x{i}' for i in range(1, 9)],
        'y': [f'y{i}' for i in range(1, 9)],
        'z': [f'z{i}' for i in range(1, 9)],
        't': [f't{i}' for i in range(1, 9)]
    }
    
    print(f"\nFEATURE TYPE IMPORTANCE:")
    for ftype, features in feature_types.items():
        type_importance = feature_importance_df[feature_importance_df['feature'].isin(features)]['importance'].sum()
        print(f"{ftype}-coordinates: {type_importance:.4f}")

print("\n🎉 MACHINE LEARNING ANALYSIS COMPLETE!")
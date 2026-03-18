#!/usr/bin/env python3
import joblib
import pandas as pd
import numpy as np

print("Checking Feature List...")
print("=" * 70)

try:
    features = joblib.load('artifacts/feature_list.pkl')
    print(f'\n✓ Feature list loaded: {len(features)} features')
    for i, f in enumerate(features, 1):
        print(f'  {i:2d}. {f}')
except Exception as e:
    print(f'✗ Error loading feature list: {e}')

print("\n" + "=" * 70)
print("Checking Scaler...")

try:
    scaler = joblib.load('artifacts/scaler.pkl')
    print(f'✓ Scaler loaded: {scaler.n_features_in_} features expected')
except Exception as e:
    print(f'✗ Error: {e}')

print("\n" + "=" * 70)
print("Testing Prediction...")

# Load test data
df = pd.read_csv('data/flood.csv')
X = df.drop(columns=['FloodProbability'])
y = df['FloodProbability']

# Load models and preprocessing
from src.preprocess import engineer_features, handle_outliers_zscore

# Get params
outlier_params = joblib.load('artifacts/outlier_params.pkl')
feature_list = joblib.load('artifacts/feature_list.pkl')
scaler = joblib.load('artifacts/scaler.pkl')
xgb = joblib.load('artifacts/models/xgboost_model.pkl')

# Test with first row
test = X.iloc[0:1].copy()
print(f'\nTest input (first row):')
print(f'  Shape: {test.shape}')

# Apply transformations (like API does)
# 1. Outlier capping
for col, p in outlier_params.items():
    if col in test.columns:
        lower = p['mean'] - 3 * p['std']
        upper = p['mean'] + 3 * p['std']
        test[col] = test[col].clip(lower=lower, upper=upper)

# 2. Feature engineering
test = engineer_features(test)
print(f'  After feature eng: {test.shape}')

# 3. Reindex to match feature list
test = test.reindex(columns=feature_list, fill_value=0)
print(f'  After reindex: {test.shape}')
print(f'  Columns match scaler: {test.shape[1]} == {scaler.n_features_in_}')

# 4. Scale
test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns)

# 5. Predict
pred = xgb.predict(test_scaled)[0]
print(f'\nPrediction: {pred:.6f}')
print(f'Actual target: {y.iloc[0]:.6f}')
print(f'Match: {abs(pred - y.iloc[0]) < 0.1}')

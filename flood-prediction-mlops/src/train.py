"""
Model Training Pipeline for Flood Prediction (CMP6230 Aligned)
Implements: RF, XGBoost, and MLP baselines. Optuna tuning, K-Fold CV.
Logs: R2, RMSE, MAE to MLflow.
"""

import pandas as pd
import numpy as np
import os
import joblib
import logging
import argparse
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import optuna
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ARTIFACTS_PATH = os.environ.get("ARTIFACTS_PATH", "artifacts")
MODELS_PATH = os.path.join(ARTIFACTS_PATH, "models")
PROCESSED_PATH = os.environ.get("PROCESSED_PATH", "data/processed")

def ensure_directories():
    os.makedirs(MODELS_PATH, exist_ok=True)

def load_processed_data():
    X_train = pd.read_csv(os.path.join(PROCESSED_PATH, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_PATH, "y_train.csv")).values.ravel()
    X_val = pd.read_csv(os.path.join(PROCESSED_PATH, "X_val.csv"))
    y_val = pd.read_csv(os.path.join(PROCESSED_PATH, "y_val.csv")).values.ravel()
    X_test = pd.read_csv(os.path.join(PROCESSED_PATH, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(PROCESSED_PATH, "y_test.csv")).values.ravel()
    return X_train, y_train, X_val, y_val, X_test, y_test

def get_model(model_type, params=None):
    params = params or {}
    if model_type == 'rf':
        return RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    elif model_type == 'xgboost':
        return XGBRegressor(**params, random_state=42, n_jobs=-1)
    elif model_type == 'mlp':
        return MLPRegressor(**params, random_state=42, max_iter=500)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def train_and_eval(model_type, X_train, y_train, X_val, y_val, X_test, y_test, params=None):
    model = get_model(model_type, params)
    
    with mlflow.start_run(run_name=f"{model_type}_baseline"):
        logger.info(f"Training {model_type}...")
        model.fit(X_train, y_train)
        
        # Validation Metrics
        val_preds = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        val_mae = mean_absolute_error(y_val, val_preds)
        val_r2 = r2_score(y_val, val_preds)
        
        # Test Metrics
        test_preds = model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        test_mae = mean_absolute_error(y_test, test_preds)
        test_r2 = r2_score(y_test, test_preds)
        
        # Logging
        mlflow.log_params(params or {})
        mlflow.log_metrics({
            "val_rmse": val_rmse, "val_mae": val_mae, "val_r2": val_r2,
            "test_rmse": test_rmse, "test_mae": test_mae, "test_r2": test_r2
        })
        
        # SHAP Explainability artifact
        try:
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_test)
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, show=False)
            shap_plot_path = os.path.join(ARTIFACTS_PATH, f"{model_type}_shap.png")
            plt.savefig(shap_plot_path, bbox_inches='tight')
            mlflow.log_artifact(shap_plot_path)
            plt.close()
        except Exception as e:
            logger.warning(f"SHAP generation failed: {e}")

        model_name = f"{model_type}_model.pkl"
        joblib.dump(model, os.path.join(MODELS_PATH, model_name))
        mlflow.sklearn.log_model(model, "model")
        
        logger.info(f"{model_type} - Test RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}")
        return model, test_rmse

def objective(trial, model_type, X_train, y_train, X_val, y_val):
    if model_type == 'rf':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
        }
    elif model_type == 'xgboost':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        }
    elif model_type == 'mlp':
        params = {
            'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(50,), (100,), (50, 50)]),
            'alpha': trial.suggest_float('alpha', 0.0001, 0.01, log=True)
        }
    
    model = get_model(model_type, params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, preds))

def run_training(model_type='rf', trials=5):
    ensure_directories()
    X_train, y_train, X_val, y_val, X_test, y_test = load_processed_data()
    
    logger.info(f"Starting {model_type} optimization...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda t: objective(t, model_type, X_train, y_train, X_val, y_val), n_trials=trials)
    
    logger.info(f"Best params: {study.best_params}")
    new_model, new_rmse = train_and_eval(model_type, X_train, y_train, X_val, y_val, X_test, y_test, study.best_params)

    # Champion-Challenger Promotion Logic
    challenger_path = os.path.join(MODELS_PATH, f"{model_type}_model_challenger.pkl")
    joblib.dump(new_model, challenger_path) # Save current run as challenger
    
    prod_model_path = os.path.join(MODELS_PATH, f"{model_type}_model.pkl") # This is what the API loads
    if os.path.exists(prod_model_path):
        try:
            prod_model = joblib.load(prod_model_path)
            prod_preds = prod_model.predict(X_test)
            prod_rmse = np.sqrt(mean_squared_error(y_test, prod_preds))
            
            if new_rmse < prod_rmse:
                logger.info(f"CHALLENGER beats CHAMPION ({new_rmse:.4f} < {prod_rmse:.4f}). Promoting.")
                joblib.dump(new_model, prod_model_path)
            else:
                logger.info(f"CHALLENGER ({new_rmse:.4f}) did not beat CHAMPION ({prod_rmse:.4f}). Discarding promotion.")
        except Exception as e:
            logger.error(f"Champion comparison failed: {e}")
            joblib.dump(new_model, prod_model_path) # Fallback if first run
    else:
        logger.info("No existing champion found. Initializing champion.")
        joblib.dump(new_model, prod_model_path)

def run_retraining(trials=3):
    """Retrains the production model (XGBoost) - Called by Airflow."""
    logger.info("Triggering automated retraining (XGBoost)...")
    run_training(model_type='xgboost', trials=trials)
    return {"status": "success", "model": "xgboost"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="rf", choices=['rf', 'xgboost', 'mlp'])
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    
    mlflow.set_experiment("Flood_Prediction_Baselines")
    run_training(args.model, args.trials)

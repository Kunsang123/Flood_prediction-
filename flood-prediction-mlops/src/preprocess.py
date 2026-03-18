"""
Data Preprocessing Pipeline for Flood Prediction (CMP6230 Aligned)
Implements: StandardScaler, Z-score capping, Specific Feature Engineering (Monsoon x Urbanization).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import logging
from src.store import get_db_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ARTIFACTS_PATH = os.environ.get("ARTIFACTS_PATH", "artifacts")
PROCESSED_PATH = os.environ.get("PROCESSED_PATH", "data/processed")


def ensure_directories():
    os.makedirs(ARTIFACTS_PATH, exist_ok=True)
    os.makedirs(PROCESSED_PATH, exist_ok=True)


def load_staging_data() -> pd.DataFrame:
    logger.info("Loading staging data from MariaDB...")
    engine = get_db_connection()
    df = pd.read_sql("SELECT * FROM flood_staging", engine)
    logger.info(f"Loaded {len(df)} rows from DB")
    return df


def handle_outliers_zscore(df: pd.DataFrame, threshold: float = 3.0, params: dict = None) -> tuple:
    """Outlier management using Z-score capping (Winsorization) - Required by Report."""
    df_clean = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if params is None:
        logger.info(f"Calculating and capping outliers (threshold={threshold})...")
        params = {}
        for col in numeric_cols:
            if col in ['FloodProbability', 'id']: continue
            mu = df[col].mean()
            std = df[col].std()
            params[col] = {'mean': mu, 'std': std}
            
            upper_limit = mu + threshold * std
            lower_limit = mu - threshold * std
            df_clean[col] = df_clean[col].clip(lower=lower_limit, upper=upper_limit)
    else:
        logger.info("Capping outliers using provided parameters...")
        for col, p in params.items():
            if col in df_clean.columns:
                upper_limit = p['mean'] + threshold * p['std']
                lower_limit = p['mean'] - threshold * p['std']
                df_clean[col] = df_clean[col].clip(lower=lower_limit, upper=upper_limit)
                
    return df_clean, params


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering as specified in CMP6230 Report."""
    logger.info("Performing feature engineering (Interaction layers)...")
    df = df.copy()
    
    # 1. Primary Interaction (Explicitly requested in Report Section II & III)
    df['MonsoonIntensity_x_Urbanization'] = df['MonsoonIntensity'] * df['Urbanization']
    
    # 2. Secondary Interactions (Aligned with previous training schema for stability)
    df['Deforestation_x_ClimateChange'] = df['Deforestation'] * df['ClimateChange']
    df['DrainageSystems_x_Urbanization'] = df['DrainageSystems'] * df['Urbanization']
    
    return df


def scale_features(df: pd.DataFrame, fit: bool = False) -> tuple:
    """Scaling numeric features using StandardScaler - Refined per report Figure 11/12."""
    scaler_path = os.path.join(ARTIFACTS_PATH, "scaler.pkl")
    
    # Identify numeric features
    df_numeric = df.select_dtypes(include=[np.number])
    
    if fit:
        logger.info("Fitting and applying StandardScaler...")
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns)
        joblib.dump(scaler, scaler_path)
    else:
        logger.info("Applying pre-fitted StandardScaler...")
        scaler = joblib.load(scaler_path)
        df_scaled = pd.DataFrame(scaler.transform(df_numeric), columns=df_numeric.columns)
        
    return df_scaled, scaler


def run_preprocessing(target_col: str = 'FloodProbability') -> dict:
    ensure_directories()
    df = load_staging_data()
    
    # Drop non-feature administrative columns
    admin_cols = ['id', 'ingestion_timestamp', 'prediction_timestamp']
    df = df.drop(columns=[c for c in admin_cols if c in df.columns])
    
    # Outlier handling via Z-score capping
    df_clean, outlier_params = handle_outliers_zscore(df)
    joblib.dump(outlier_params, os.path.join(ARTIFACTS_PATH, "outlier_params.pkl"))
    
    # Feature engineering (Interaction layers)
    df_eng = engineer_features(df_clean)
    
    if target_col not in df_eng.columns:
        raise ValueError(f"Target column '{target_col}' not found")
        
    X = df_eng.drop(columns=[target_col])
    y = df_eng[target_col]
    
    # Ensure numeric types only
    X = X.select_dtypes(include=[np.number])
    
    # Apply StandardScaler
    X_scaled, _ = scale_features(X, fit=True)
    
    # Data split (70:15:15) as specified in Report Section II (Model Development)
    # 70% train, 30% temp (which splits to 15% val, 15% test)
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    X_train.to_csv(os.path.join(PROCESSED_PATH, "X_train.csv"), index=False)
    pd.DataFrame(y_train).to_csv(os.path.join(PROCESSED_PATH, "y_train.csv"), index=False)
    X_val.to_csv(os.path.join(PROCESSED_PATH, "X_val.csv"), index=False)
    pd.DataFrame(y_val).to_csv(os.path.join(PROCESSED_PATH, "y_val.csv"), index=False)
    X_test.to_csv(os.path.join(PROCESSED_PATH, "X_test.csv"), index=False)
    pd.DataFrame(y_test).to_csv(os.path.join(PROCESSED_PATH, "y_test.csv"), index=False)
    
    joblib.dump(list(X_train.columns), os.path.join(ARTIFACTS_PATH, "feature_list.pkl"))
    
    logger.info(f"Preprocessing complete. Saved {len(X_train.columns)} features. Split: 70/15/15.")
    return {'n_features': len(X_train.columns), 'split': '70/15/15'}


if __name__ == "__main__":
    run_preprocessing()

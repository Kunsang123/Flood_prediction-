"""
EDA Script for Flood Prediction

Generates: descriptive statistics, correlation matrices, histograms,
boxplots, feature importance analysis. Outputs saved as artifacts.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import mlflow
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EDA_OUTPUT_DIR = os.environ.get("EDA_OUTPUT_DIR", os.path.join(os.environ.get("ARTIFACTS_PATH", "artifacts"), "eda"))


def ensure_output_dir():
    os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)


def load_data(file_path: str) -> pd.DataFrame:
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def generate_descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Generating descriptive statistics...")
    stats = df.describe().T
    stats['missing'] = df.isnull().sum()
    stats['missing_pct'] = (df.isnull().sum() / len(df) * 100).round(2)
    stats.to_csv(os.path.join(EDA_OUTPUT_DIR, "descriptive_statistics.csv"))
    return stats


def generate_correlation_matrix(df: pd.DataFrame):
    logger.info("Generating correlation matrix...")
    corr = df.corr()
    corr.to_csv(os.path.join(EDA_OUTPUT_DIR, "correlation_matrix.csv"))
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap='RdBu_r', center=0, square=True, linewidths=0.5)
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_DIR, "correlation_heatmap.png"), dpi=150, bbox_inches='tight')
    plt.close()


def generate_histograms(df: pd.DataFrame):
    logger.info("Generating histograms...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_cols = 4
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()
    for idx, col in enumerate(numeric_cols):
        axes[idx].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
        axes[idx].set_title(col, fontsize=10)
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].set_visible(False)
    plt.suptitle('Feature Distributions', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_DIR, "histograms.png"), dpi=150, bbox_inches='tight')
    plt.close()


def generate_boxplots(df: pd.DataFrame):
    logger.info("Generating boxplots...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_cols = 4
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()
    for idx, col in enumerate(numeric_cols):
        sns.boxplot(data=df, y=col, ax=axes[idx], color='steelblue')
        axes[idx].set_title(col, fontsize=10)
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].set_visible(False)
    plt.suptitle('Feature Boxplots (Outlier Detection)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_DIR, "boxplots.png"), dpi=150, bbox_inches='tight')
    plt.close()


def generate_feature_importance(df: pd.DataFrame, target_col: str = 'FloodProbability') -> pd.DataFrame:
    logger.info("Calculating feature importance...")
    if target_col not in df.columns:
        return None
    X = df.drop(columns=[target_col])
    y = df[target_col]
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importance_df = pd.DataFrame({
        'feature': X.columns, 'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    importance_df.to_csv(os.path.join(EDA_OUTPUT_DIR, "feature_importance.csv"), index=False)
    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
    plt.title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_DIR, "feature_importance.png"), dpi=150, bbox_inches='tight')
    plt.close()
    return importance_df


def log_to_mlflow(artifacts_dir: str):
    try:
        mlflow.set_experiment("Flood_Prediction_EDA")
        with mlflow.start_run(run_name="eda_analysis"):
            for f in os.listdir(artifacts_dir):
                fp = os.path.join(artifacts_dir, f)
                if os.path.isfile(fp):
                    mlflow.log_artifact(fp)
            mlflow.log_param("timestamp", datetime.now().isoformat())
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")


def run_eda(data_path: str, log_mlflow: bool = False) -> dict:
    logger.info("=" * 50)
    logger.info("Starting Exploratory Data Analysis")
    logger.info("=" * 50)
    ensure_output_dir()
    df = load_data(data_path)
    stats = generate_descriptive_statistics(df)
    generate_correlation_matrix(df)
    generate_histograms(df)
    generate_boxplots(df)
    importance = generate_feature_importance(df)
    if log_mlflow:
        log_to_mlflow(EDA_OUTPUT_DIR)
    logger.info("EDA Completed!")
    return {'statistics': stats, 'feature_importance': importance, 'artifacts_dir': EDA_OUTPUT_DIR}


if __name__ == "__main__":
    try:
        results = run_eda("data/flood.csv", log_mlflow=False)
        if results['feature_importance'] is not None:
            print("\nTop 5 Features:")
            print(results['feature_importance'].head())
    except Exception as e:
        logger.error(f"EDA failed: {e}")
        exit(1)

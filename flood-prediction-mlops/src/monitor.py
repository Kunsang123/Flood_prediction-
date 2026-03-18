"""
Data Monitoring Pipeline for Flood Prediction
Implements: KS Test for Data Drift, PSI (Population Stability Index) for Concept Drift.
"""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import logging
import os
import json
from datetime import datetime
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REPORTS_PATH = os.environ.get("REPORTS_PATH", os.path.join(os.environ.get("ARTIFACTS_PATH", "artifacts"), "reports"))


def ensure_directories():
    os.makedirs(REPORTS_PATH, exist_ok=True)


def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI).
    PSI = sum((actual% - expected%) * ln(actual% / expected%))
    """
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Use quantiles for buckets on expected data
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints) # Handle discrete values
    
    # Ensure we have at least one bucket boundary
    if len(breakpoints) < 2:
        # If all values are identical, create a small range
        breakpoints = np.array([expected[0] - 0.01, expected[0] + 0.01])

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]
    
    expected_percents = expected_counts / len(expected)
    actual_percents = actual_counts / len(actual)
    
    # Avoid division by zero and log of zero
    expected_percents = np.clip(expected_percents, 1e-5, None)
    actual_percents = np.clip(actual_percents, 1e-5, None)
    
    psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    return float(psi_value)


def detect_drift(reference_path: str, current_path: str, threshold_ks: float = 0.05, threshold_psi: float = 0.2) -> Dict[str, Any]:
    logger.info("Starting Multi-dimensional Drift Detection (KS + PSI)")
    ensure_directories()

    ref_df = pd.read_csv(reference_path)
    curr_df = pd.read_csv(current_path)

    features = [c for c in ref_df.columns if c in curr_df.columns and pd.api.types.is_numeric_dtype(ref_df[c])]

    drifted_features: List[str] = []
    high_psi_features: List[str] = []
    feature_reports: Dict[str, Dict[str, Any]] = {}

    total_psi = 0.0
    count = 0

    for feat in features:
        if feat == 'FloodProbability': continue
        
        stat, pval = ks_2samp(ref_df[feat].dropna().values, curr_df[feat].dropna().values)
        is_ks_drift = bool(pval < threshold_ks)
        
        psi_val = calculate_psi(ref_df[feat].dropna().values, curr_df[feat].dropna().values)
        is_high_psi = bool(psi_val >= threshold_psi)
        
        feature_reports[feat] = {
            'ks_statistic': float(stat),
            'ks_p_value': float(pval),
            'ks_drift': is_ks_drift,
            'psi_value': psi_val,
            'high_psi': is_high_psi
        }
        
        if is_ks_drift: drifted_features.append(feat)
        if is_high_psi: high_psi_features.append(feat)
            
        total_psi += psi_val
        count += 1

    avg_psi = total_psi / count if count > 0 else 0.0
    
    drift_report: Dict[str, Any] = {
        'timestamp': datetime.now().isoformat(),
        'reference_samples': len(ref_df),
        'current_samples': len(curr_df),
        'ks_threshold': threshold_ks,
        'psi_threshold': threshold_psi,
        'features': feature_reports,
        'average_psi': avg_psi,
        'drift_detected': len(drifted_features) > 0,
        'retrain_recommended': bool(avg_psi >= threshold_psi or len(high_psi_features) > 2),
        'drifted_features': drifted_features,
        'high_psi_features': high_psi_features
    }

    report_path = os.path.join(REPORTS_PATH, f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, 'w') as f:
        json.dump(drift_report, f, indent=2)

    return drift_report


if __name__ == "__main__":
    try:
        # Self-check
        if os.path.exists("data/processed/X_train.csv"):
            detect_drift("data/processed/X_train.csv", "data/processed/X_train.csv")
    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        exit(1)

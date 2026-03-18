"""Flood Prediction Monitoring DAG (Enhanced Verbosity & Protocol Safe)"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
import sys
import os
import json
import logging

sys.path.insert(0, '/opt/airflow')

# Ensure protocols are always used for service URLs
API_BASE_URL = os.environ.get("API_URL", "http://api:8000")

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def setup_env():
    os.environ['DATA_PATH'] = '/opt/airflow/data'
    os.environ['STAGING_PATH'] = '/opt/airflow/data/staging'
    os.environ['PROCESSED_PATH'] = '/opt/airflow/data/processed'
    os.environ['ARTIFACTS_PATH'] = '/opt/airflow/artifacts'
    os.environ['MODELS_PATH'] = '/opt/airflow/artifacts/models'
    os.environ['EDA_OUTPUT_DIR'] = '/opt/airflow/artifacts/eda'
    os.environ['REPORTS_PATH'] = '/opt/airflow/artifacts/reports'

def check_data_drift(**kwargs):
    setup_env()
    logging.info("Starting Data Drift Check...")
    ref_path = '/opt/airflow/data/processed/X_train.csv'
    cur_path = '/opt/airflow/data/processed/X_test.csv'
    
    if not os.path.exists(ref_path) or not os.path.exists(cur_path):
        logging.warning(f"Data files missing. Ref: {os.path.exists(ref_path)}, Cur: {os.path.exists(cur_path)}")
        return 'check_model_performance'
        
    try:
        from src.monitor import detect_drift
        report = detect_drift(reference_path=ref_path, current_path=cur_path)
        
        avg_psi = report.get('average_psi', 0)
        retrain_rec = report.get('retrain_recommended', False)
        
        logging.info(f"Drift Analysis - Avg PSI: {avg_psi:.4f}, Retrain Recommended: {retrain_rec}")
        kwargs['ti'].xcom_push(key='drift_report', value=json.dumps(report))
        
        if retrain_rec:
            logging.info(">>> Retraining recommended based on drift detection.")
            return 'trigger_retraining'
        
        logging.info("No significant drift detected. Skipping retraining.")
        return 'check_model_performance'
        
    except Exception as e:
        logging.error(f"Drift check failed: {e}")
        return 'check_model_performance'

def trigger_retraining(**kwargs):
    setup_env()
    logging.info("Starting automated retraining task...")
    try:
        from src.train import run_retraining
        results = run_retraining(trials=3)
        logging.info(f"Retraining completed successfully: {results}")
        return f"Retraining complete: {results}"
    except Exception as e:
        logging.error(f"Retraining task failed: {e}")
        raise

def check_model_performance(**kwargs):
    import requests
    logging.info(f"Checking API performance at {API_BASE_URL}...")
    try:
        test_data = {
            "MonsoonIntensity": 10, "TopographyDrainage": 10,
            "RiverManagement": 10, "Deforestation": 10,
            "Urbanization": 10, "ClimateChange": 10,
            "DamsQuality": 10, "Siltation": 10,
            "AgriculturalPractices": 10, "Encroachments": 10,
            "IneffectiveDisasterPreparedness": 10, "DrainageSystems": 10,
            "CoastalVulnerability": 10, "Landslides": 10,
            "Watersheds": 10, "DeterioratingInfrastructure": 10,
            "PopulationScore": 10, "WetlandLoss": 10,
            "InadequatePlanning": 10, "PoliticalFactors": 10
        }
        # Use full URL with protocol
        url = f"{API_BASE_URL}/predict"
        logging.info(f"Sending request to {url}")
        
        response = requests.post(url, json=test_data, 
                               headers={"X-API-Key": "secret-token"}, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            logging.info("API performance check successful.")
            kwargs['ti'].xcom_push(key='perf_check', value=json.dumps(result))
            return "Performance check passed"
        
        err_msg = f"API error: {response.status_code} - {response.text}"
        logging.error(err_msg)
        return err_msg
    except Exception as e:
        logging.error(f"Performance check crashed: {e}")
        return str(e)

def log_monitoring_results(**kwargs):
    ti = kwargs['ti']
    drift_report = ti.xcom_pull(task_ids='drift_check', key='drift_report')
    perf_check = ti.xcom_pull(task_ids='check_model_performance', key='perf_check')
    logging.info(f"Monitoring Cycle Finished.\nDrift: {drift_report}\nPerformance: {perf_check}")
    return "Monitoring results summarized."

with DAG(
    'flood_prediction_monitoring',
    default_args=default_args,
    description='Monitor model performance and data drift',
    schedule_interval='@daily',
    catchup=False,
    tags=['flood', 'mlops', 'monitoring'],
) as dag:

    drift_check = BranchPythonOperator(task_id='drift_check', python_callable=check_data_drift)
    retrain = PythonOperator(task_id='trigger_retraining', python_callable=trigger_retraining)
    perf_check = PythonOperator(task_id='check_model_performance', python_callable=check_model_performance)
    log_results = PythonOperator(task_id='log_results', python_callable=log_monitoring_results,
                                  trigger_rule='none_failed_min_one_success')

    drift_check >> [retrain, perf_check]
    retrain >> log_results
    perf_check >> log_results

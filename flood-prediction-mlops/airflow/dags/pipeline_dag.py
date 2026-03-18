"""Flood Prediction Pipeline DAG (CMP6230 Aligned)"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
import os

sys.path.insert(0, '/opt/airflow')

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def setup_env():
    """Set env vars so source modules use correct paths inside Airflow container."""
    os.environ['DATA_PATH'] = '/opt/airflow/data'
    os.environ['STAGING_PATH'] = '/opt/airflow/data/staging'
    os.environ['PROCESSED_PATH'] = '/opt/airflow/data/processed'
    os.environ['ARTIFACTS_PATH'] = '/opt/airflow/artifacts'
    os.environ['MODELS_PATH'] = '/opt/airflow/artifacts/models'
    os.environ['EDA_OUTPUT_DIR'] = '/opt/airflow/artifacts/eda'
    os.environ['REPORTS_PATH'] = '/opt/airflow/artifacts/reports'


def run_ingestion(**kwargs):
    setup_env()
    from src.ingest import ingest_data
    # Corrected signature: only source_path is required
    result = ingest_data(
        source_path='/opt/airflow/data/flood.csv'
    )
    kwargs['ti'].xcom_push(key='ingest_result', value=str(result))
    return "Ingestion complete"


def run_eda_task(**kwargs):
    setup_env()
    from src.eda import run_eda
    # Point to the actual source data for EDA
    result = run_eda(
        data_path='/opt/airflow/data/flood.csv',
        log_mlflow=True
    )
    return "EDA complete"


def run_preprocessing(**kwargs):
    setup_env()
    from src.preprocess import run_preprocessing as preprocess
    result = preprocess()
    return "Preprocessing complete"


def run_training_task(**kwargs):
    setup_env()
    from src.train import run_retraining
    # run_retraining uses XGBoost as the production baseline
    results = run_retraining(trials=3)
    kwargs['ti'].xcom_push(key='training_results', value=str(results))
    return "Training complete"


def run_health_check(**kwargs):
    import requests
    try:
        # Use secret-token for authentication
        response = requests.get("http://api:8000/health", timeout=10)
        if response.status_code == 200:
            return "API health check passed"
        return f"API health check failed: {response.status_code}"
    except Exception as e:
        return f"API health check error: {e}"


with DAG(
    'flood_prediction_pipeline',
    default_args=default_args,
    description='End-to-end flood prediction MLOps pipeline',
    schedule_interval='@weekly',
    catchup=False,
    tags=['flood', 'mlops', 'pipeline'],
) as dag:

    ingest = PythonOperator(task_id='ingest_data', python_callable=run_ingestion)
    eda = PythonOperator(task_id='run_eda', python_callable=run_eda_task)
    preprocess = PythonOperator(task_id='preprocess_data', python_callable=run_preprocessing)
    train = PythonOperator(task_id='train_models', python_callable=run_training_task)
    health_check = PythonOperator(task_id='health_check', python_callable=run_health_check)

    ingest >> eda >> preprocess >> train >> health_check

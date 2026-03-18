# Flood Prediction MLOps Pipeline

End-to-end MLOps pipeline for flood prediction using Python, Apache Airflow, MLflow, Docker, FastAPI, and Streamlit.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        FLOOD PREDICTION MLOPS                           │
├─────────────────────────────────────────────────────────────────────────┤
│  Data Layer           │  ML Pipeline          │  Serving Layer          │
│  ───────────          │  ────────────         │  ─────────────          │
│  • Raw Data           │  • Preprocessing      │  • FastAPI              │
│  • Staging Data       │  • Training           │  • Streamlit            │
│  • Processed Data     │  • MLflow Tracking    │  • Prometheus           │
│  • MariaDB            │  • Model Registry     │  • Grafana              │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.10+ (for local development)
- WSL2 (recommended for Windows)

### 1. Start All Services
```bash
docker-compose up -d --build
```

### 2. Access Services
| Service | URL | Credentials |
|---------|-----|-------------|
| Airflow | http://localhost:8080 | admin / admin |
| MLflow     | http://localhost:5001 | -             |
| API | http://localhost:8000 | - |
| Frontend | http://localhost:8501 | - |
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | - |
| MinIO | http://localhost:9001 | minioadmin / minioadmin |

### 3. Run the Pipeline
1. Open Airflow UI
2. Enable `flood_prediction_pipeline` DAG
3. Trigger manually or wait for scheduled run

## Project Structure
```
flood-prediction-mlops/
├── airflow/dags/           # Airflow DAGs
├── api/                    # FastAPI application
├── artifacts/              # Model artifacts & reports
├── data/
│   ├── raw/               # Raw ingested data
│   ├── staging/           # Validated data
│   └── processed/         # Train/Val/Test splits
├── docker/                 # Service Dockerfiles
├── frontend/               # Streamlit dashboard
├── monitoring/
│   ├── prometheus/        # Prometheus config
│   └── grafana/           # Grafana dashboards
├── notebooks/              # EDA notebooks
├── src/                    # Core ML code
│   ├── eda.py             # Exploratory analysis
│   ├── fix_shap.py        # SHAP explanation fixing
│   ├── ingest.py          # Data ingestion
│   ├── monitor.py         # Drift detection
│   ├── preprocess.py      # Preprocessing
│   ├── setup_db.py        # Database setup scripts
│   ├── store.py           # Database storage
│   ├── train.py           # Model training
│   └── validation.py      # Data validation utilities
├── docker-compose.yml
├── flood.csv
├── monitor.log
└── requirements.txt
```

## Features

### Data Pipeline
- **Great Expectations** validation
- Three-layer architecture (raw → staging → processed)
- Z-score outlier handling
- Feature engineering (interaction features)

### Model Training
- **Models**: Random Forest, XGBoost, MLP
- **Hyperparameter Tuning**: Optuna
- **Cross-Validation**: K-Fold
- **Metrics**: R², RMSE, MAE

### Experiment Tracking
- MLflow for metrics, params, artifacts
- Model versioning & registry

### Monitoring
- Prometheus metrics from API
- Grafana dashboards
- KS test for data drift detection
- Automated retraining triggers

### Deployment
- FastAPI REST API (`/predict` endpoint)
- Streamlit interactive dashboard
- Docker containerization

## API Usage

```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MonsoonIntensity": 10,
    "TopographyDrainage": 8,
    "RiverManagement": 7,
    "Deforestation": 6,
    "Urbanization": 9,
    "ClimateChange": 7,
    "DamsQuality": 5,
    "Siltation": 4,
    "AgriculturalPractices": 6,
    "Encroachments": 5,
    "IneffectiveDisasterPreparedness": 8,
    "DrainageSystems": 6,
    "CoastalVulnerability": 7,
    "Landslides": 4,
    "Watersheds": 5,
    "DeterioratingInfrastructure": 6,
    "PopulationScore": 8,
    "WetlandLoss": 5,
    "InadequatePlanning": 7,
    "PoliticalFactors": 4
  }'
```

## Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run preprocessing
python src/preprocess.py

# Train models
python src/train.py --model all --trials 10

# Start API locally
uvicorn api.app:app --reload

# Start frontend locally
streamlit run frontend/dashboard.py
```

## License
MIT

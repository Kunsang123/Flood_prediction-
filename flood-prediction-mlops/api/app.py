"""FastAPI Application for Flood Prediction Model Serving (CMP6230 Aligned)"""

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import os
import logging
import time
import json
import redis
import hashlib
from starlette.concurrency import run_in_threadpool
from datetime import datetime
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Flood Prediction API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

PREDICTION_COUNTER = Counter('flood_predictions_total', 'Total predictions')
PREDICTION_LATENCY = Histogram('flood_prediction_latency_seconds', 'Prediction latency')
PREDICTION_VALUE = Histogram('flood_prediction_value', 'Predicted values', buckets=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])

model = None
scaler = None
outlier_params = None
feature_list = None
model_name_loaded = None

ARTIFACTS_PATH = os.getenv("ARTIFACTS_PATH", "artifacts")
MODELS_PATH = os.getenv("MODELS_PATH", os.path.join(ARTIFACTS_PATH, "models"))
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
API_TOKEN = os.getenv("API_TOKEN", "secret-token")

api_key_header = APIKeyHeader(name="X-API-Key")

def get_redis_client():
    try:
        return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        return None

def verify_token(api_key: str = Security(api_key_header)):
    if api_key != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

class PredictionRequest(BaseModel):
    MonsoonIntensity: float = Field(..., ge=0, le=20)
    TopographyDrainage: float = Field(..., ge=0, le=20)
    RiverManagement: float = Field(..., ge=0, le=20)
    Deforestation: float = Field(..., ge=0, le=20)
    Urbanization: float = Field(..., ge=0, le=20)
    ClimateChange: float = Field(..., ge=0, le=20)
    DamsQuality: float = Field(..., ge=0, le=20)
    Siltation: float = Field(..., ge=0, le=20)
    AgriculturalPractices: float = Field(..., ge=0, le=20)
    Encroachments: float = Field(..., ge=0, le=20)
    IneffectiveDisasterPreparedness: float = Field(..., ge=0, le=20)
    DrainageSystems: float = Field(..., ge=0, le=20)
    CoastalVulnerability: float = Field(..., ge=0, le=20)
    Landslides: float = Field(..., ge=0, le=20)
    Watersheds: float = Field(..., ge=0, le=20)
    DeterioratingInfrastructure: float = Field(..., ge=0, le=20)
    PopulationScore: float = Field(..., ge=0, le=20)
    WetlandLoss: float = Field(..., ge=0, le=20)
    InadequatePlanning: float = Field(..., ge=0, le=20)
    PoliticalFactors: float = Field(..., ge=0, le=20)

class PredictionResponse(BaseModel):
    flood_probability: float  # Percentage 0-100
    flood_probability_score: float = None  # Normalized 0-1
    risk_level: str
    timestamp: str
    model_version: str = "1.0.0"
    cached: bool = False

def load_artifacts(requested_model_override: str = None):
    global model, scaler, outlier_params, feature_list, model_name_loaded
    
    logger.info(f"Model reload triggered. Requested: {requested_model_override}")
    
    new_model = None
    new_model_name = None
    
    # Load Model
    requested_model = requested_model_override or os.getenv("MODEL_TYPE", "xgboost")
    search_order = [requested_model, 'xgboost', 'rf', 'mlp', 'catboost']
    seen = set()
    ordered_search = []
    for m in search_order:
        if m not in seen:
            ordered_search.append(m)
            seen.add(m)

    for m_type in ordered_search:
        mp = os.path.join(MODELS_PATH, f"{m_type}_model.pkl")
        if os.path.exists(mp):
            try:
                logger.info(f"Attempting to load {m_type} from {mp}...")
                loaded = joblib.load(mp)
                new_model = loaded
                new_model_name = m_type
                logger.info(f"Successfully loaded model: {m_type}")
                break
            except Exception as e:
                logger.warning(f"Failed to load {mp}: {e}")

    # Load Scaler
    new_scaler = None
    sp = os.path.join(ARTIFACTS_PATH, "scaler.pkl")
    if os.path.exists(sp):
        try:
            new_scaler = joblib.load(sp)
            logger.info("Scaler loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load scaler: {e}")
    
    # Use atomic swaps to avoid health check failures during load
    if new_model:
        model = new_model
        model_name_loaded = new_model_name
    
    if new_scaler:
        scaler = new_scaler

    # Load Outlier Params (non-critical)
    op = os.path.join(ARTIFACTS_PATH, "outlier_params.pkl")
    if os.path.exists(op):
        try:
            outlier_params = joblib.load(op)
        except: pass
        
    # Load Feature List (non-critical)
    fp = os.path.join(ARTIFACTS_PATH, "feature_list.pkl")
    if os.path.exists(fp):
        try:
            feature_list = joblib.load(fp)
        except: pass
    
    logger.info("Artifact reload complete.")

def get_risk_level(p: float) -> str:
    """Risk classification based on normalized probability [0, 1]"""
    if p <= 0.3: return "LOW"
    elif p <= 0.6: return "MODERATE"
    elif p <= 0.8: return "HIGH"
    else: return "CRITICAL"

@app.on_event("startup")
async def startup_event():
    load_artifacts()

@app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "model_name": model_name_loaded or "None"
    }

@app.get("/model/info")
async def model_info():
    if not model:
        raise HTTPException(status_code=404, detail="No model loaded")
    
    n_features = len(feature_list) if feature_list else 0
    return {
        "model_type": model_name_loaded,
        "n_features": n_features,
        "artifacts_path": MODELS_PATH,
        "features": feature_list
    }

@app.post("/model/load")
async def manual_load(model_type: str = None, token: str = Depends(verify_token)):
    # Run synchronous load_artifacts in a threadpool to keep the event loop free
    await run_in_threadpool(load_artifacts, requested_model_override=model_type)
    return {"status": "success", "loaded_model": model_name_loaded}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, token: str = Depends(verify_token)):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded on server")
        
    start = time.time()
    req_dict = request.dict()
    
    # Redis Cache with stable hashing (includes model name to avoid cross-model leakage)
    r = get_redis_client()
    cache_payload = {"params": req_dict, "model": model_name_loaded}
    cache_key = f"pred:{hashlib.sha256(json.dumps(cache_payload, sort_keys=True).encode()).hexdigest()}"
    if r:
        try:
            cached_val = r.get(cache_key)
            if cached_val:
                latency = time.time() - start
                PREDICTION_LATENCY.observe(latency)
                return PredictionResponse(**json.loads(cached_val), cached=True)
        except: pass

    df = pd.DataFrame([req_dict])
    
    # 1. Outlier Capping - DISABLED for user inputs since UI constrains to [0, 20]
    # Original 3-sigma clipping prevented extrapolation beyond training data range (~0-17)
    # Users legitimately input 20, so we allow the full [0, 20] range to be used
    # if outlier_params:
    #     for col, p in outlier_params.items():
    #         if col in df.columns:
    #             lower = p['mean'] - 3 * p['std']
    #             upper = p['mean'] + 3 * p['std']
    #             df[col] = df[col].clip(lower=lower, upper=upper)

    # 2. Feature Engineering
    try:
        from src.preprocess import engineer_features
        df = engineer_features(df)
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
    
    # Ensure correct columns
    if feature_list:
        df = df.reindex(columns=feature_list, fill_value=0)
    
    # 3. Scaling
    if scaler:
        try:
            df = pd.DataFrame(scaler.transform(df), columns=df.columns)
        except Exception as e:
            logger.error(f"Scaling failed: {e}")
        
    # 4. Prediction
    try:
        prediction = float(model.predict(df)[0])
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    
    # Clip to valid range [0, 1] - models trained on [0, 1] target range
    prediction = np.clip(prediction, 0, 1)
    
    # Keep original normalized value [0, 1]
    prediction_normalized = prediction
    
    # Convert to percentage [0, 100] for user display
    prediction_percentage = prediction_normalized * 100
    
    response_data = {
        "flood_probability": round(prediction_percentage, 2),  # Show as percentage 0-100%
        "flood_probability_score": round(prediction_normalized, 4),  # Also show 0-1 scale
        "risk_level": get_risk_level(prediction_normalized),
        "timestamp": datetime.now().isoformat()
    }

    if r:
        try:
            r.setex(cache_key, 3600, json.dumps(response_data))
        except: pass

    # Record metrics
    latency = time.time() - start
    PREDICTION_COUNTER.inc()
    PREDICTION_LATENCY.observe(latency)
    PREDICTION_VALUE.observe(prediction_normalized)
    return PredictionResponse(**response_data, cached=False)

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

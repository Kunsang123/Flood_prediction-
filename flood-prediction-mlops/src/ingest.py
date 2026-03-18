"""
Data Ingestion Pipeline for Flood Prediction
Implements: AWS S3 (MinIO) audit storage, LOAD DATA INFILE, Star Schema ELT.
"""

import pandas as pd
import logging
import os
from datetime import datetime
from sqlalchemy import text
import boto3
from botocore.exceptions import ClientError

from src.store import get_db_connection, execute_sql, store_data
from src.validation import validate_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# AWS S3 / MinIO Configuration
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://minio:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
S3_BUCKET = "flood-raw-audit"


def get_s3_client():
    """Initialize S3 client (MinIO compatible)"""
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        use_ssl=False
    )


def ensure_s3_bucket():
    """Create S3 bucket for raw data audit"""
    try:
        s3 = get_s3_client()
        s3.head_bucket(Bucket=S3_BUCKET)
        logger.info(f"S3 bucket '{S3_BUCKET}' exists")
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            s3 = get_s3_client()
            s3.create_bucket(Bucket=S3_BUCKET)
            logger.info(f"Created S3 bucket '{S3_BUCKET}' for raw data audit")
        else:
            raise


def upload_csv_to_s3(file_path: str, timestamp: str) -> str:
    """Upload unaltered CSV to S3 for audit and rollback"""
    ensure_s3_bucket()
    s3 = get_s3_client()
    s3_key = f"raw-csv/flood_{timestamp}.csv"
    
    with open(file_path, 'rb') as f:
        s3.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=f)
    
    logger.info(f"Raw CSV uploaded to S3: s3://{S3_BUCKET}/{s3_key}")
    return s3_key


def load_data(file_path: str) -> pd.DataFrame:
    logger.info(f"Loading data from {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source file not found: {file_path}")
    df = pd.read_csv(file_path)
    return df


def ingest_data(source_path: str) -> dict:
    """
    Complete ingestion pipeline using LOAD DATA INFILE and ELT
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Load data for validation
        df = load_data(source_path)
        
        # Validate using Great Expectations
        if not validate_data(df):
            raise ValueError("Great Expectations validation failed")
        
        # 2. UPLOAD UNALTERED CSV TO AWS S3 FOR AUDIT AND ROLLBACK
        logger.info("Uploading unaltered CSV to S3 for audit trail...")
        s3_key = upload_csv_to_s3(source_path, timestamp)
        
        # 3. LOAD DATA INFILE into flood_raw
        abs_path = os.path.abspath(source_path).replace('\\', '/')
        
        logger.info(f"Loading data into flood_raw using LOAD DATA INFILE from {abs_path}")
        
        # ELT: Fresh load. Use a single session to disable FK checks effectively.
        engine = get_db_connection()
        with engine.begin() as conn:
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
            conn.execute(text("TRUNCATE TABLE fact_flood_prediction"))
            conn.execute(text("TRUNCATE TABLE flood_raw"))
            conn.execute(text("TRUNCATE TABLE flood_staging"))
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
        
        load_query = f"""
        LOAD DATA INFILE '{abs_path}'
        INTO TABLE flood_raw
        FIELDS TERMINATED BY ','
        OPTIONALLY ENCLOSED BY '"'
        LINES TERMINATED BY '\\n'
        IGNORE 1 LINES
        (MonsoonIntensity, TopographyDrainage, RiverManagement, Deforestation, Urbanization, 
         ClimateChange, DamsQuality, Siltation, AgriculturalPractices, Encroachments, 
         IneffectiveDisasterPreparedness, DrainageSystems, CoastalVulnerability, Landslides, 
         Watersheds, DeterioratingInfrastructure, PopulationScore, WetlandLoss, 
         InadequatePlanning, PoliticalFactors, FloodProbability);
        """
        try:
            execute_sql(load_query)
        except Exception as e:
            logger.warning(f"LOAD DATA INFILE failed: {e}. Falling back to standard store_data.")
            store_data(df, "flood_raw", if_exists='append')

        # 4. ELT: Populate Star Schema
        logger.info("Performing ELT: Populating Dimension and Fact tables...")
        
        # Populate dimensions first
        execute_sql("INSERT IGNORE INTO dim_climate (MonsoonIntensity, ClimateChange) SELECT DISTINCT MonsoonIntensity, ClimateChange FROM flood_raw")
        execute_sql("INSERT IGNORE INTO dim_infrastructure (DamsQuality, DrainageSystems, TopographyDrainage, RiverManagement, DeterioratingInfrastructure) SELECT DISTINCT DamsQuality, DrainageSystems, TopographyDrainage, RiverManagement, DeterioratingInfrastructure FROM flood_raw")
        execute_sql("INSERT IGNORE INTO dim_governance (PoliticalFactors, InadequatePlanning, IneffectiveDisasterPreparedness) SELECT DISTINCT PoliticalFactors, InadequatePlanning, IneffectiveDisasterPreparedness FROM flood_raw")
        execute_sql("INSERT IGNORE INTO dim_socio_env (Urbanization, Deforestation, Siltation, AgriculturalPractices, Encroachments, CoastalVulnerability, Landslides, Watersheds, PopulationScore, WetlandLoss) SELECT DISTINCT Urbanization, Deforestation, Siltation, AgriculturalPractices, Encroachments, CoastalVulnerability, Landslides, Watersheds, PopulationScore, WetlandLoss FROM flood_raw")
        
        # Populate Fact table
        fact_query = """
            INSERT INTO fact_flood_prediction (climate_id, infra_id, gov_id, socio_id, FloodProbability, raw_id)
            SELECT 
                c.climate_id, i.infra_id, g.gov_id, s.socio_id, r.FloodProbability, r.id
            FROM flood_raw r
            JOIN dim_climate c ON r.MonsoonIntensity = c.MonsoonIntensity AND r.ClimateChange = c.ClimateChange
            JOIN dim_infrastructure i ON r.DamsQuality = i.DamsQuality AND r.DrainageSystems = i.DrainageSystems 
                AND r.TopographyDrainage = i.TopographyDrainage AND r.RiverManagement = i.RiverManagement 
                AND r.DeterioratingInfrastructure = i.DeterioratingInfrastructure
            JOIN dim_governance g ON r.PoliticalFactors = g.PoliticalFactors AND r.InadequatePlanning = g.InadequatePlanning 
                AND r.IneffectiveDisasterPreparedness = g.IneffectiveDisasterPreparedness
            JOIN dim_socio_env s ON r.Urbanization = s.Urbanization AND r.Deforestation = s.Deforestation 
                AND r.Siltation = s.Siltation AND r.AgriculturalPractices = s.AgriculturalPractices 
                AND r.Encroachments = s.Encroachments AND r.CoastalVulnerability = s.CoastalVulnerability 
                AND r.Landslides = s.Landslides AND r.Watersheds = s.Watersheds 
                AND r.PopulationScore = s.PopulationScore AND r.WetlandLoss = s.WetlandLoss
            """
        execute_sql(fact_query)
            
        # 5. Populate flood_staging
        execute_sql("INSERT INTO flood_staging SELECT * FROM flood_raw")

        result = {
            'status': 'success',
            'records': len(df),
            'timestamp': datetime.now().isoformat(),
            's3_audit_key': s3_key
        }
        return result
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/flood.csv")
    args = parser.parse_args()
    try:
        res = ingest_data(args.source)
        print(f"Ingestion successful: {res}")
    except Exception as e:
        exit(1)

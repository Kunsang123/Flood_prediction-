"""
Database Setup Script - MariaDB Star Schema
"""

import logging
from sqlalchemy import text
from src.store import get_db_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_star_schema():
    engine = get_db_connection()
    
    # We use DROP TABLE to ensure we have the correct columns for this run
    drop_queries = [
        "DROP TABLE IF EXISTS fact_flood_prediction",
        "DROP TABLE IF EXISTS dim_climate",
        "DROP TABLE IF EXISTS dim_infrastructure",
        "DROP TABLE IF EXISTS dim_governance",
        "DROP TABLE IF EXISTS dim_socio_env",
        "DROP TABLE IF EXISTS flood_raw",
        "DROP TABLE IF EXISTS flood_staging"
    ]

    queries = [
        # 1. Raw table (preserving original data)
        """
        CREATE TABLE flood_raw (
            id INT AUTO_INCREMENT PRIMARY KEY,
            MonsoonIntensity INT,
            TopographyDrainage INT,
            RiverManagement INT,
            Deforestation INT,
            Urbanization INT,
            ClimateChange INT,
            DamsQuality INT,
            Siltation INT,
            AgriculturalPractices INT,
            Encroachments INT,
            IneffectiveDisasterPreparedness INT,
            DrainageSystems INT,
            CoastalVulnerability INT,
            Landslides INT,
            Watersheds INT,
            DeterioratingInfrastructure INT,
            PopulationScore INT,
            WetlandLoss INT,
            InadequatePlanning INT,
            PoliticalFactors INT,
            FloodProbability FLOAT,
            ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB;
        """,
        
        # 2. Dimension Tables
        """
        CREATE TABLE dim_climate (
            climate_id INT AUTO_INCREMENT PRIMARY KEY,
            MonsoonIntensity INT,
            ClimateChange INT,
            UNIQUE KEY uq_climate (MonsoonIntensity, ClimateChange)
        ) ENGINE=InnoDB;
        """,
        """
        CREATE TABLE dim_infrastructure (
            infra_id INT AUTO_INCREMENT PRIMARY KEY,
            DamsQuality INT,
            DrainageSystems INT,
            TopographyDrainage INT,
            RiverManagement INT,
            DeterioratingInfrastructure INT,
            UNIQUE KEY uq_infra (DamsQuality, DrainageSystems, TopographyDrainage, RiverManagement, DeterioratingInfrastructure)
        ) ENGINE=InnoDB;
        """,
        """
        CREATE TABLE dim_governance (
            gov_id INT AUTO_INCREMENT PRIMARY KEY,
            PoliticalFactors INT,
            InadequatePlanning INT,
            IneffectiveDisasterPreparedness INT,
            UNIQUE KEY uq_gov (PoliticalFactors, InadequatePlanning, IneffectiveDisasterPreparedness)
        ) ENGINE=InnoDB;
        """,
        """
        CREATE TABLE dim_socio_env (
            socio_id INT AUTO_INCREMENT PRIMARY KEY,
            Urbanization INT,
            Deforestation INT,
            Siltation INT,
            AgriculturalPractices INT,
            Encroachments INT,
            CoastalVulnerability INT,
            Landslides INT,
            Watersheds INT,
            PopulationScore INT,
            WetlandLoss INT,
            UNIQUE KEY uq_socio (Urbanization, Deforestation, Siltation, AgriculturalPractices, Encroachments, CoastalVulnerability, Landslides, Watersheds, PopulationScore, WetlandLoss)
        ) ENGINE=InnoDB;
        """,
        
        # 3. Fact Table
        """
        CREATE TABLE fact_flood_prediction (
            fact_id INT AUTO_INCREMENT PRIMARY KEY,
            climate_id INT,
            infra_id INT,
            gov_id INT,
            socio_id INT,
            FloodProbability FLOAT,
            raw_id INT,
            prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (climate_id) REFERENCES dim_climate(climate_id),
            FOREIGN KEY (infra_id) REFERENCES dim_infrastructure(infra_id),
            FOREIGN KEY (gov_id) REFERENCES dim_governance(gov_id),
            FOREIGN KEY (socio_id) REFERENCES dim_socio_env(socio_id),
            FOREIGN KEY (raw_id) REFERENCES flood_raw(id)
        ) ENGINE=InnoDB;
        """,
        
        # 4. Staging table for processing (Mirror of flood_raw)
        """
        CREATE TABLE flood_staging (
            id INT AUTO_INCREMENT PRIMARY KEY,
            MonsoonIntensity INT,
            TopographyDrainage INT,
            RiverManagement INT,
            Deforestation INT,
            Urbanization INT,
            ClimateChange INT,
            DamsQuality INT,
            Siltation INT,
            AgriculturalPractices INT,
            Encroachments INT,
            IneffectiveDisasterPreparedness INT,
            DrainageSystems INT,
            CoastalVulnerability INT,
            Landslides INT,
            Watersheds INT,
            DeterioratingInfrastructure INT,
            PopulationScore INT,
            WetlandLoss INT,
            InadequatePlanning INT,
            PoliticalFactors INT,
            FloodProbability FLOAT,
            ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB;
        """
    ]
    
    with engine.begin() as conn:
        for q in drop_queries:
             conn.execute(text(q))
             logger.info(f"Dropped: {q}")
        for query in queries:
            try:
                conn.execute(text(query))
                logger.info(f"Created: {query.strip().splitlines()[0]}...")
            except Exception as e:
                logger.error(f"Error executing query: {e}")
    
    logger.info("Star Schema setup complete.")


if __name__ == "__main__":
    setup_star_schema()

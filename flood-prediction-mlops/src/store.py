import pandas as pd
from sqlalchemy import create_engine, text
import logging
import os
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_db_connection():
    user = os.getenv("MYSQL_USER", "flood_user")
    password = os.getenv("MYSQL_PASSWORD", "flood_password")
    host = os.getenv("MYSQL_HOST", "mariadb")
    port = os.getenv("MYSQL_PORT", "3306")
    db_name = os.getenv("MYSQL_DATABASE", "flood_prediction")
    engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}")
    return engine


def store_data(df: pd.DataFrame, table_name: str, if_exists: str = 'replace'):
    try:
        logger.info(f"Storing data to '{table_name}'...")
        engine = get_db_connection()
        df.to_sql(table_name, con=engine, if_exists=if_exists, index=False)
        logger.info("Stored successfully.")
    except Exception as e:
        logger.error(f"Failed to store: {e}")
        raise


def execute_sql(query: str, params: Optional[Dict[str, Any]] = None):
    """Execute raw SQL query."""
    engine = get_db_connection()
    # Use begin() for automatic commit in SQLAlchemy 1.4+
    with engine.begin() as conn:
        try:
            conn.execute(text(query), params or {})
            logger.info("SQL executed successfully.")
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            raise

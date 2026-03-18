import great_expectations as ge
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def validate_data(df: pd.DataFrame) -> bool:
    """Validate data using Great Expectations."""
    logger.info("Validating data with Great Expectations...")
    gedf = ge.from_pandas(df)
    
    # Expect columns to exist
    expected_columns = [
        'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement',
        'Deforestation', 'Urbanization', 'ClimateChange', 'DamsQuality',
        'Siltation', 'AgriculturalPractices', 'Encroachments',
        'IneffectiveDisasterPreparedness', 'DrainageSystems',
        'CoastalVulnerability', 'Landslides', 'Watersheds',
        'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
        'InadequatePlanning', 'PoliticalFactors', 'FloodProbability'
    ]
    
    for col in expected_columns:
        gedf.expect_column_to_exist(col)
        
    # Expect values to be within range [0, 30] for features (be more inclusive of outliers)
    for col in expected_columns[:-1]:
        gedf.expect_column_values_to_be_between(col, min_value=0, max_value=30)
        
    # Expect target to be within [0, 1]
    gedf.expect_column_values_to_be_between('FloodProbability', min_value=0, max_value=1)
    
    # Expect no nulls - Loop over columns as GE expect_column_values_to_not_be_null takes one col
    for col in expected_columns:
        gedf.expect_column_values_to_not_be_null(col)
    
    results = gedf.validate()
    
    if not results["success"]:
        logger.error("Great Expectations Validation Failed!")
        # Log failure details
        for res in results["results"]:
            if not res["success"]:
                logger.error(f"Failed expectation: {res['expectation_config']['expectation_type']} on {res['expectation_config']['kwargs'].get('column')}")
        return False
        
    logger.info("Great Expectations Validation Passed!")
    return True

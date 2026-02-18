import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataPreprocessing
from src.model_training import ModelTraining
from config.path_config import *
from utils.common_functions import read_yaml
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger("training_pipeline")

def run_training_pipeline():
    try:
        print("DEBUG: Starting training pipeline...")
        logger.info("Starting the training pipeline")
        config = read_yaml(CONFIG_PATH)
        print("DEBUG: Config read successfully.")
        
        # 1. Data Ingestion
        print("DEBUG: Starting Data Ingestion...")
        data_ingestion = DataIngestion(config)
        data_ingestion.run()
        print("DEBUG: Data Ingestion completed.")
        
        # 2. Data Preprocessing
        print("DEBUG: Starting Data Preprocessing...")
        data_preprocessing = DataPreprocessing(config)
        data_preprocessing.run()
        print("DEBUG: Data Preprocessing completed.")
        
        # 3. Model Training
        print("DEBUG: Starting Model Training...")
        model_training = ModelTraining(config)
        model_training.run()
        print("DEBUG: Model Training completed.")
        
        logger.info("Training pipeline completed successfully")
        print("DEBUG: Training pipeline finished successfully!")
    except Exception as e:
        print(f"DEBUG ERROR: {e}")
        logger.error(f"Error while running the training pipeline: {e}")
        raise CustomException(e, sys)
        logger.error(f"Error while running the training pipeline: {e}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    run_training_pipeline()
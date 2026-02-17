from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataPreprocessing
from src.model_training import ModelTraining
from config.path_config import *
from config.config import get_config
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger("training_pipeline")

def run_training_pipeline():
    try:
        logger.info("Starting the training pipeline")
        config = get_config()
        # ================================================================================================================================
        # Data Ingestion
        # ================================================================================================================================
        data_ingestion = DataIngestion(config)
        data_ingestion.run()
        # ================================================================================================================================
        # Data Preprocessing
        # ================================================================================================================================
        data_preprocessing = DataPreprocessing(config)
        data_preprocessing.run()
        # ================================================================================================================================
        # Model Training
        # ================================================================================================================================
        model_training = ModelTraining(config)
        model_training.run()
        logger.info("Training pipeline completed successfully")
    except Exception as e:
        logger.error("Error while running the training pipeline")
        raise CustomException(e, sys)
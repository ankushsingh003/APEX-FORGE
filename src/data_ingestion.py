import os
import sys
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from utils.common_functions import read_yaml
from config.path_config import *

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        self.config = config["data_ingestion"]
        self.train_ratio = self.config["train_ratio"]
        # In this local setup, we look for data in archive (1)/booking.csv
        self.local_data_path = "archive (1)/booking.csv"
        
        os.makedirs(os.path.dirname(RAW_FILE_PATH), exist_ok=True)
        logger.info("Data Ingestion Initialized")
         
    def ingest_local_data(self):
        try:
            if os.path.exists(self.local_data_path):
                logger.info(f"Found local data at {self.local_data_path}. Copying to {RAW_FILE_PATH}")
                shutil.copy(self.local_data_path, RAW_FILE_PATH)
            else:
                logger.error(f"Local data not found at {self.local_data_path}")
                raise FileNotFoundError(f"Missing {self.local_data_path}")
        except Exception as e:
            logger.error(f"Error while ingesting local data: {e}")
            raise CustomException(e, sys)

    def split_data(self):
        try:
            logger.info(f"Splitting data from {RAW_FILE_PATH}")
            df = pd.read_csv(RAW_FILE_PATH)
            
            train_df, test_df = train_test_split(df, test_size=1-self.train_ratio, random_state=42)
            
            train_df.to_csv(TRAIN_FILE_PATH, index=False)
            test_df.to_csv(TEST_FILE_PATH, index=False)
            logger.info(f"Data split saved to {TRAIN_FILE_PATH} and {TEST_FILE_PATH}")
        except Exception as e:
            logger.error(f"Error while splitting data: {e}")
            raise CustomException(e, sys)
    
    def run(self):
        try:
            self.ingest_local_data()
            self.split_data()
            logger.info("Data Ingestion run completed")
        except Exception as e:
            logger.error("Error in data ingestion run")
            raise CustomException(e, sys)

if __name__ == "__main__":
    from utils.common_functions import read_yaml
    from config.path_config import CONFIG_PATH
    config = read_yaml(CONFIG_PATH)
    data_ingestion = DataIngestion(config)
    data_ingestion.run()
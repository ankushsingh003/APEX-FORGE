import os
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from src.utils.common_functions import read_yaml
import sys
from config import *

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self,config):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.dataset_name = self.config["dataset_name"]
        self.train_test_ratio = self.config["train_ratio"]
        

        os.makedirs(os.path.dirname(RAW_FILE_PATH), exist_ok=True)
        logger.info(f"Data Ingestion Initiated with bucket name: {self.bucket_name} and dataset name: {self.dataset_name}")
         
    def download_csv_from_gcp(self):
        try:
            storage_client = storage.Client()
            bucket = storage_client.get_bucket(self.bucket_name)
            blob = bucket.blob(self.dataset_name)
            blob.download_to_filename(RAW_FILE_PATH)
            logger.info(f"Dataset downloaded from GCP bucket: {self.bucket_name} and dataset name: {self.dataset_name}")
        except Exception as e:
            logger.error(f"Error while downloading dataset from GCP bucket: {self.bucket_name} and dataset name: {self.dataset_name}")
            raise CustomException(e, sys)


    def split_data(self):
        try:
            df = pd.read_csv(RAW_FILE_PATH)
            train_df, test_df = train_test_split(df, test_size=self.train_test_ratio, random_state=42)
            train_df.to_csv(TRAIN_FILE_PATH, index=False)
            test_df.to_csv(TEST_FILE_PATH, index=False)
            logger.info(f"Dataset split into train and test with ratio: {self.train_test_ratio}")
        except Exception as e:
            logger.error(f"Error while splitting dataset into train and test with ratio: {self.train_test_ratio}")
            raise CustomException(e, sys)
    

    def run(self):
        try:
            logger.info("Data Ingestion Initiated")
            self.download_csv_from_gcp()
            self.split_data()
            logger.info("Data Ingestion Completed")
        except Exception as e:
            logger.error(f"Error while running data ingestion")
            raise CustomException(e, sys)

    
if __name__ == "__main__":
    config = read_yaml(CONFIG_PATH)
    data_ingestion = DataIngestion(config)
    data_ingestion.run()
    
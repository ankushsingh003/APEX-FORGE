import os
import sys
import pandas as pd
import yaml
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            raise Exception(f"File not found: {file_path}")
        with open(file_path, "r") as file:
            config = yaml.safe_load(file)
            logger.info("Config file read successfully")
            return config
    except Exception as e:
        logger.error(f"Error while reading config file: {e}")
        raise CustomException(e, sys)

def load_data(file_path):
    try:
        if not os.path.exists(file_path):
            raise Exception(f"File not found: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error while loading data from {file_path}: {e}")
        raise CustomException(e, sys)

import os
import pandas as pd
from src.logger import get_logger
from src.custom_exception import CustomException
import yaml


logger = get_logger(__name__)

def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            raise CustomException("File not found")
        with open(file_path, "r") as file:
            config = yaml.safe_load(file)
            logger.info("Config file read successfully")
            return config
    except Exception as e:
        logger.error("Error while reading config file")
        raise CustomException("Error while reading config file", e)
    

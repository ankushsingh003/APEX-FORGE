import os
import pandas as pd
import numpy as np 
from src.logger import get_logger
from config.paths_config import *
from src.custom_exception import CustomException
from utils.common_functions import load_data, read_yaml
from sklearn.ensemble import LabelEncoder

from imblearrn.over_sampling import SMOTE
logger = get_logger(__name__)


class DataPreprocessing:
    def __init__(self ,  train_path , test_path , processed_dir , config_path  ):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config_path = config_path
        self.config = read_yaml(config_path)
        
        
    def preprocess(self):
        try:
            logger.info("Starting data preprocessing")
            
            logger.info(f" Dropping the columns")
            
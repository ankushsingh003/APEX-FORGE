import os
import sys
import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from src.logger import get_logger
from config.path_config import *
from src.custom_exception import CustomException
from utils.common_functions import load_data, read_yaml

logger = get_logger(__name__)

class DataPreprocessing:
    def __init__(self, config):
        self.config = config
        self.cat_cols = self.config["data_processing"]["categorical_cols"]
        self.num_cols = self.config["data_processing"]["numerical_cols"]
        self.skew_threshold = self.config["data_processing"]["skew_threshold"]
        self.num_features_to_select = self.config["data_processing"]["num_features_to_select"]
        
    def preprocess_df(self, df):
        try:
            logger.info("Preprocessing dataframe")
            df = df.copy()
            
            # Drop unnecessary columns
            for col in ["Booking_ID", "date of reservation"]:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)
            
            df.drop_duplicates(inplace=True)
            
            # Fill missing values if any
            df.fillna(method='ffill', inplace=True)

            # Label Encoding
            le = LabelEncoder()
            for col in self.cat_cols:
                if col in df.columns:
                    df[col] = le.fit_transform(df[col].astype(str))
            
            # Skewness Handling
            for col in self.num_cols:
                if col in df.columns:
                    if df[col].skew() > self.skew_threshold:
                        df[col] = np.log1p(df[col])
            
            return df
        except Exception as e:
            logger.error(f"Error while preprocessing dataframe: {e}")
            raise CustomException(e, sys)
    
    def balance_data(self, df):
        try:
            logger.info("Balancing the data using SMOTE")
            X = df.drop(columns=["booking status"])
            y = df["booking status"]
            
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
            balanced_df["booking status"] = y_resampled
            return balanced_df
        except Exception as e:
            logger.error(f"Error while balancing data: {e}")
            raise CustomException(e, sys) 
            
    def feature_selection(self, df):
        try:
            logger.info("Performing feature selection using RandomForest")
            X = df.drop(columns=["booking status"])
            y = df["booking status"]
            
            model = RandomForestClassifier(random_state=42)
            model.fit(X, y)
            
            feature_importance = pd.Series(model.feature_importances_, index=X.columns)
            top_features = feature_importance.nlargest(self.num_features_to_select).index.tolist()
            
            logger.info(f"Top {self.num_features_to_select} features selected: {top_features}")
            return df[top_features + ["booking status"]]
        except Exception as e:
            logger.error(f"Error while performing feature selection: {e}")
            raise CustomException(e, sys)

    def run(self):
        try:
            logger.info("Data preprocessing started")
            # Load raw train/test split from ingestion
            train = load_data(TRAIN_FILE_PATH)
            test = load_data(TEST_FILE_PATH)
            
            # Preprocess
            train = self.preprocess_df(train)
            test = self.preprocess_df(test)
            
            # Balance (only train usually, but following original logic for now)
            train = self.balance_data(train)
            # test = self.balance_data(test) # Usually don't balance test
            
            # Feature Selection
            train_selected = self.feature_selection(train)
            # Match test columns to train selected columns
            columns_to_keep = train_selected.columns.tolist()
            test_selected = test[columns_to_keep]
            
            # Save
            os.makedirs(PROCESSED_DIR, exist_ok=True)
            train_selected.to_csv(PROCESSED_TRAIN_PATH, index=False)
            test_selected.to_csv(PROCESSED_TEST_PATH, index=False)
            
            logger.info("Data preprocessing completed and saved")
        except Exception as e:
            logger.error(f"Error in preprocessing run: {e}")
            raise CustomException(e, sys)

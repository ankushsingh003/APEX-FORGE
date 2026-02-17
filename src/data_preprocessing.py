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
            train.drop(columns=["Booking_ID"] , inplace=True)
            train.ddrop_duplicates(inplace=True)
            cat_cols = self.config["data_processing"]["categorical_cols"]
            num_cols = self.config["data_processing"]["numerical_cols"]
            
            logger.info(f"Applying LabelEncoder")
            mappings = {}
            for col in categorical_cols:
                train[col] = label_encoder.fit_transform(train[col])

                mappings[col] = { label:code for label , code in zip( label_encoder.classes_ , label_encoder.transform(label_encoder.classes_))}
            
            logger.info(f"LabelEncoder applied successfully")
            logger.info(f"Mappings:  ")
            for col , mapping in mappings.items():
                logger.info(f"{col}: {mapping}")

            logger.info(f"Doing Skewness Handling")
            skew_threshold = self.config["data_processing"]["skew_threshold"]
            skewness = X_numeric.apply(lambda x:x.skew())

            for column in skewness[skewness > skew_threshold].index:
                X_numeric[column] = np.log1p(X_numeric[column])

        except Exception as e:
            logger.error("Error while doing skewness handling")
            raise CustomException(e, sys)
    
    def balance_data(self , train):
        try:
            logger.info("Balancing the data")
            X = train.drop( columns=["booking status"])
            Y = train["booking status"]
            smote = SMOTE()
            X_resampled , Y_resampled = smote.fit_resample(X , Y)
            balanced_df  = pd.DataFrame(X_resampled , columns = X.columns)
            balanced_df["booking status"] = Y_resampled
            logger.info("Data balanced successfully")
            return balanced_df
        except Exception as e:
            logger.error("Error while balancing the data")
            raise CustomException(e, sys) 
            
    def feature_selection(self , df):
        try:
            logger.info("Feature selection")
            X = df.drop( columns = "booking_status")
            y = df["booking_status"]
            model = RandomForestClassifier(random_state=42)
            feature_importance = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                "feature": X.columns,
                "important": feature_importance
            })
            feature_importance_df.sort_values( by = "important" , ascending = False )
            num_features_to_select = self.config["data_processing"]["num_features_to_select"]
            top_feature_importance_df  = feature_importance_df.sort_values( by = "important" , ascending = False )
            top_10_features = top_feature_importance_df["feature"].head(num_features_to_select).values
            top_10_df = df[top_10_features.tolist() + ["booking_status"]]
            logger.info(f"Top {num_features_to_select} features selected")
            return top_10_df
        except Exception as e:
            logger.error("Error while doing feature selection")
            raise CustomException(e, sys)

    def save_data( self , file_path):
        try:
            logger.info("Saving the data")
            df.to_csv(file_path , index = False)
            logger.info("Data saved successfully")
        except Exception as e:
            logger.error("Error while saving the data")
            raise CustomException(e, sys)


    def process(self):
        try:
            logger.info("Data preprocessing Initiated")
            train = load_data(self.train_path)
            test = load_data(self.test_path)
            train = self.preprocess(train)
            test = self.preprocess(test)
            train = self.balance_data(train)
            test = self.balance_data(test)
            train = self.feature_selection(train)
            test = self.feature_selection(test)
            self.save_data(train , PROCESSED_TRAIN_PATH)
            self.save_data(test , PROCESSED_TEST_PATH)
            logger.info("Data preprocessing Completed")
        except Exception as e:
            logger.error("Error while doing data preprocessing")
            raise CustomException(e, sys)

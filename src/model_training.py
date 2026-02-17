import os 
import pandas as pd 
import joblib
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix , precision_score , recall_score , f1_score , roc_auc_score , log_loss , mean_squared_error , mean_absolute_error , r2_score , mean_squared_log_error , median_absolute_error , mean_absolute_percentage_error , mean_poisson_deviance , mean_gamma_deviance , mean_tweedie_deviance
from src.logger import get_logger
from src.custom_exception import CustomException
from config.path_config import *
from config.model_params import LightGBM_params
from utils.common_functions import load_data , read_yaml
from scipy.stats import randint

logger = get_logger("model_training")

class ModelTraining:
    def __init__(self , train_path , test_path , model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path

        self.params_dict = LIGHTGBM_PARAMS
        self.random_search = RANDOM_SEARCH_PARAMS
    
    def load_and_split_data(self):
        try:
            logger.info(f"Loading and splitting the data")
            train = load_data(self.train_path)
            test = load_data(self.test_path)
            X_train = train.drop(columns=["booking status"])
            y_train = train["booking status"]
            X_test = test.drop(columns=["booking status"])
            y_test = test["booking status"]
            logger.info("Data loaded and split successfully")
            return X_train , y_train , X_test , y_test
        except Exception as e:
            logger.error("Error while loading and splitting the data")
            raise CustomException(e, sys)
            
        
    def train_lgbm( self , X_train  , y_train):
        try:
            logger.info(f"Training LightGBM model")
            lgbm = lgb.LGBMClassifier(random_state=self.random_search["random_state"])
            
            random_search = RandomizedSearchCV(
                estimator=rf,
                param_distributions=param_dist,
                n_iter=5,
                cv=5,
                n_jobs=-1,
                random_state=42,
                scoring='accuracy'
            )
            logger.info("Starting the model training")
            random_search.fit(X_train , y_train)
            logger.info("Hyperparamtere training completed")
            
            best_params = random_search.best_params_
            logger.info(f"Got the best parameters: {best_params}")
            best_lgbm_model = random_search.best_estimator_
            logger.info("Best LightGBM model trained successfully")
            return best_lgbm_model
        except Exception as e:
            logger.error("Error while training LightGBM model")
            raise CustomException(e, sys)


    def evaluate_model(self , model , X_test , y_test):
        try:
            logger.info("Evaluating the model")
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test , y_pred)
            logger.info(f"Accuracy: {accuracy}")
            precision = precision_score(y_test , y_pred)
            logger.info(f"Precision: {precision}")
            recall = recall_score(y_test , y_pred)
            logger.info(f"Recall: {recall}")
            f1 = f1_score(y_test , y_pred)
            logger.info(f"F1 Score: {f1}")
            
            return accuracy , precision , recall , f1 
        except Exception as e:
            logger.error("Error while evaluating the model")
            raise CustomException(e, sys)


            
    def save_model(self,model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
            logger.info(f"Saving the model to {self.model_output_path}")
            joblib.dump(model, self.model_output_path)
            logger.info(f"Model saved to {self.model_output_path}")
        except Exception as e:
            logger.error("Error while saving the model")
            raise CustomException(e, sys)

    def run(self):
        try:
            logger.info("Starting the model training pipeline")
            X_train , y_train , X_test , y_test = self.load_and_split_data()
            model = self.train_lgbm(X_train , y_train)
            accuracy , precision , recall , f1 = self.evaluate_model(model , X_test , y_test)
            self.save_model(model)
            logger.info("Model training pipeline completed")
        except Exception as e:
            logger.error("Error while running the model training pipeline")
            raise CustomException(e, sys)


if __name__ == "__main__":
    model_training = ModelTraining(TRAIN_FILE_PATH, TEST_FILE_PATH, MODEL_PATH)
    model_training.run()
    
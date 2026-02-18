import os
import sys
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.path_config import *
from utils.common_functions import load_data
import mlflow
import mlflow.sklearn

logger = get_logger("model_training")

class ModelTraining:
    def __init__(self, config):
        self.config = config
        self.model_path = MODEL_PATH
        
    def load_data(self):
        try:
            logger.info("Loading processed data for training")
            train = load_data(PROCESSED_TRAIN_PATH)
            test = load_data(PROCESSED_TEST_PATH)
            
            X_train = train.drop(columns=["booking status"])
            y_train = train["booking status"]
            X_test = test.drop(columns=["booking status"])
            y_test = test["booking status"]
            
            return X_train, y_train, X_test, y_test
        except Exception as e:
            logger.error("Error while loading processed data")
            raise CustomException(e, sys)

    def train_model(self, X_train, y_train):
        try:
            logger.info("Training LightGBM model")
            # Using basic params or defaults since config structure might vary
            model = lgb.LGBMClassifier(random_state=42)
            
            # Simple hyperparameter optimization
            param_grid = {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'num_leaves': [31, 50]
            }
            
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=3,
                cv=2,
                random_state=42,
                n_jobs=-1
            )
            
            random_search.fit(X_train, y_train)
            logger.info(f"Best parameters: {random_search.best_params_}")
            return random_search.best_estimator_
        except Exception as e:
            logger.error(f"Error while training model: {e}")
            raise CustomException(e, sys)

    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info("Evaluating model")
            y_pred = model.predict(X_test)
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred)
            }
            logger.info(f"Metrics: {metrics}")
            return metrics
        except Exception as e:
            logger.error("Error evaluating model")
            raise CustomException(e, sys)

    def save_model(self, model):
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            logger.info(f"Saving model to {self.model_path}")
            joblib.dump(model, self.model_path)
        except Exception as e:
            logger.error("Error saving model")
            raise CustomException(e, sys)

    def run(self):
        try:
            # Set local tracking URI for MLflow to avoid connection issues
            mlflow.set_tracking_uri("file:./mlruns")
            
            with mlflow.start_run():
                X_train, y_train, X_test, y_test = self.load_data()
                model = self.train_model(X_train, y_train)
                metrics = self.evaluate_model(model, X_test, y_test)
                self.save_model(model)
                
                # Log to MLflow
                for name, value in metrics.items():
                    mlflow.log_metric(name, value)
                
                mlflow.sklearn.log_model(model, "model")
                logger.info("Model training pipeline complete")
        except Exception as e:
            logger.error("Error in model training run")
            raise CustomException(e, sys)

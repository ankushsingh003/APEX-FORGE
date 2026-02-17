# Classification Models and Hyperparameter Tuning Code

# 1. Import Statements
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd

# 2. Model Evaluation Function
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "SVC": SVC(),
    "Gaussian NB": GaussianNB(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": LGBMClassifier()
}

def evaluate_models(models, X_train, X_test, y_train, y_test):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='weighted'),
            "Recall": recall_score(y_test, y_pred, average='weighted'),
            "F1-Score": f1_score(y_test, y_pred, average='weighted')
        }
        print(f"--- {name} ---\n{classification_report(y_test, y_pred)}\n")
    return pd.DataFrame(results).T

# 3. Hyperparameter Tuning for Random Forest
rf_param_dist = {
    'n_estimators': [100, 300, 500, 800, 1000],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [10, 30, 50, 70, 90, 110, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Example tuning:
# rf_random = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=rf_param_dist, n_iter=20, cv=3, verbose=2, random_state=42, n_jobs=-1)
# rf_random.fit(X_train, y_train)

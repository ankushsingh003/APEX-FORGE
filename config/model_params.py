from scipy.stats import randint

LIGHTGBM_PARAMS = {
    "n_estimators": randint(100, 1000),
    "learning_rate": randint(0.01, 0.3),
    "max_depth": randint(3, 15),
    "num_leaves": randint(31, 150),
    "min_child_samples": randint(10, 100),
    "subsample": randint(0.5, 1.0),
    "colsample_bytree": randint(0.5, 1.0),
    "reg_alpha": randint(0, 10),
    "reg_lambda": randint(0, 10)
}



RANDOM_SEARCH_PARAMS = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=5,
    cv=5,
    n_jobs=-1,
    random_state=42,
    scoring='accuracy'
)


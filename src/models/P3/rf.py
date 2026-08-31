from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import pandas as pd
import numpy as np

from mlflow_tracking.tracking_utils import (
    configure_tracking,
    mlflow_run,
    log_dataset_context,
    log_split_config,
    log_model_params,
    log_metrics,
    log_model_and_artifacts,
)

from mlflow_tracking.experiments import get_experiment_name


# Configs
DATASET_PATH = "data/datasets/clan_war_performance_regression.parquet"
RANDOM_STATE = 42


# Loading the dataset
data = pd.read_parquet(DATASET_PATH)


# Defining features and target
X = data.drop(columns=[
    "clan_tag",
    "war_frequency",
    "war_league",
    "capital_league",
    "type",
    "location_name",
    "war_success_rate",
])

y = data["war_success_rate"]


# Train and test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
)


# MLflow experiment
experiment_name = get_experiment_name("p3")
configure_tracking()


with mlflow_run(
    experiment_name,
    run_name="RF RandomizedSearchCV",
):

    # Logging dataset context
    log_dataset_context(
        DATASET_PATH,
        row_count=len(data),
        feature_count=X.shape[1],
        target="war_success_rate",
    )

    # Logging train/test split config
    log_split_config(
        split_strategy="train_test_split",
        test_size=0.2,
        random_seed=RANDOM_STATE,
    )

    # Base model
    model = RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # Hyperparameter search space
    param_distributions = {
        "n_estimators": [100, 150, 200, 250, 300],
        "max_depth": [None, 10, 15, 20, 25, 30],
        "min_samples_split": [2, 5, 10, 15],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": [1.0, "sqrt", "log2"],
        "bootstrap": [True],
    }

    # Randomized Search
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="neg_mean_absolute_error",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        return_train_score=False,
    )

    # Training
    search.fit(
        X_train,
        y_train,
    )

    # Best model
    best_model = search.best_estimator_

    # Best parameters
    best_params = search.best_params_

    # Logging best model parameters
    log_model_params({
        "n_estimators": best_params["n_estimators"],
        "max_depth": best_params["max_depth"],
        "min_samples_split": best_params["min_samples_split"],
        "min_samples_leaf": best_params["min_samples_leaf"],
        "max_features": best_params["max_features"],
        "bootstrap": best_params["bootstrap"],
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "tuning_method": "RandomizedSearchCV",
        "n_iter": 20,
        "cv": 3,
        "scoring": "neg_mean_absolute_error",
    })

    # Prediction
    y_pred = best_model.predict(X_test)

    # Metrics
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
    }

    # Logging metrics
    log_metrics(metrics)

    # Logging best model and artifacts
    log_model_and_artifacts(best_model)
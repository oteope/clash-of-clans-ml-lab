from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
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
DATASET_PATH = "data/datasets/clan_performance_classification.parquet"
RANDOM_STATE = 42


# Loading the dataset
data = pd.read_parquet(DATASET_PATH)


# Defining features and target
X = data.drop(columns="performance_class").select_dtypes(include=["number", "bool"])

y = data["performance_class"]


# Train and test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
)


# MLflow experiment
experiment_name = get_experiment_name("p4")
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
        target="performance_class",
    )

    # Logging train/test split config
    log_split_config(
        split_strategy="train_test_split",
        test_size=0.2,
        random_seed=RANDOM_STATE,
    )

    # Base model
    model = RandomForestClassifier(
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
        scoring="f1_macro",
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
        "scoring": "f1_macro",
    })

    # Prediction
    y_pred = best_model.predict(X_test)

    # Metrics
    metrics = {
        "accuracy": accuracy_score(
            y_test,
            y_pred,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_test,
            y_pred,
        ),
        "f1_macro": f1_score(
            y_test,
            y_pred,
            average="macro",
        ),
        "f1_weighted": f1_score(
            y_test,
            y_pred,
            average="weighted",
        ),
    }

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
    )

    # Logging metrics
    log_metrics(metrics)

    # Logging best model and artifacts
    log_model_and_artifacts(
        best_model,
        cm,
    )
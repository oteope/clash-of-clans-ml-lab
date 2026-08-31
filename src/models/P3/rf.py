from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

#Configs
DATASET_PATH = "data/datasets/clan_war_performance_regression.parquet"
RANDOM_STATE = 42

#Loading the dataset
data = pd.read_parquet(DATASET_PATH)

#Defining the features and variables
X = data.drop(columns="")
y = data("war_success_rate")

#Train and test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = RANDOM_STATE,
)

#Mlflow experiment
experiment_name = get_experiment_name("p3")
configure_tracking()

with mlflow_run(
    experiment_name,
    run_name = "RF baseline"
):
    
    #Logging dataset context
    log_dataset_context(
        DATASET_PATH,
        row_count = len(data),
        feature_count = X.shape[1],
        target = "war_success_rate"
    )
    
    #Logging test and train split config
    log_split_config(
        split_strategy="train_test_split",
        test_size=0.2,
        random_seed=RANDOM_STATE,
    )
    
    #Model
    model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=1.0,
    bootstrap=True,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    )
    
    #Log model params
    log_model_params({
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": 1.0,
    "bootstrap": True,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    })
    
    #Training
    model.fit(
        X_train,
        y_train,
    )
    
    #Predict
    y_pred = model.predict(X_test)
    
    #Metrics
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
    }
        
    #Logging metrics
    log_metrics(metrics)
    
    #Logging the model and artifacts
    log_model_and_artifacts(model)
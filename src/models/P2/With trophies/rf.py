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

#Loading the dataset
DATASET_PATH = "data/datasets/clan_rank_regression_with_trophies.parquet"
data = pd.read_parquet(DATASET_PATH)

#Defining target and features
X = data.drop(columns=[
    "clan_rank",
    "player_tag",
    "clan_tag",
    "name",
    "league_name",
    "league_tier_name",
    "war_frequency",
    "war_league",
    "capital_league",
    "type",
])
y = data["clan_rank"]

#Preparing the train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42,)


#Mlflow experiment
configure_tracking()
experiment_name = get_experiment_name("p2")

with mlflow_run(
    experiment_name,
    run_name = "RF baseline - with trophies"
):
    #Logging the dataset context
    log_dataset_context(
        DATASET_PATH,
        row_count = len(data),
        feature_count = X.shape[1],
        target="clan_rank",
    )
    
    #Logging split config
    log_split_config(
    split_strategy="train_test_split",
    test_size=0.2,
    random_seed=42,
    )
    
    #Model
    model = RandomForestRegressor(random_state=42,
                                   n_estimators=100,)
    
    #Logging model parameters
    log_model_params({
        "random_state":42,
        "n_estimators":100,
    })
    
    
    #Training the model
    model.fit(X_train,
              y_train,)
    
    #Prediction
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
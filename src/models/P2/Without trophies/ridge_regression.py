from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

#Loading the dataset
DATASET_PATH = "data/datasets/clan_rank_regression_without_trophies.parquet"
data = pd.read_parquet(DATASET_PATH)

#Selecting the features
X = data.drop(
    columns=[
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
        "capital_contributions",
        "clan_mean_capital_contributions",
    ],
    errors="ignore",
)

y = data["clan_rank"]

#Spliting the dataset
X_train, X_test, y_train, y_test = train_test_split(
                                                    X,
                                                    y,
                                                    test_size = 0.2,
                                                    random_state = 42
                                                )

#Mlflow experiment
configure_tracking()
experiment_name = get_experiment_name("p2")

with mlflow_run(
    experiment_name,
    run_name = "v2 Ridge regression - without trophies"
):
    #Dataset information
    log_dataset_context(
        DATASET_PATH,
        row_count = len(data),
        feature_count = X.shape[1],
        target = "clan_rank",
    )
    
    #Split configuration
    log_split_config(
        split_strategy="train_test_split",
        test_size=0.2,
        random_seed=42,
        preprocessing_config={
            "scaler": "StandardScaler"
        },
    )
    
    #Model
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=10)),
        ])
    
    #Logging model parameters
    log_model_params({
        "imputer":"strategy=median",
        "scaler":"StandardScaler",
        "alpha":10,
    })
    
    #Training the model
    model.fit(X_train,y_train)
    
    #Prediction
    y_pred = model.predict(X_test)
    
    #Metrics
    metrics = {"mae": mean_absolute_error(y_test, y_pred),
               "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "r2": r2_score(y_test, y_pred)}
    
    #Logging the metrics
    log_metrics(metrics)
    
    #Logging model and artifacts
    log_model_and_artifacts(model)
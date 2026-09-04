from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
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
from sklearn.metrics import train_test_split
from mlflow_tracking.experiments import get_experiment_name

#Configs
DATASET_PATH = "data/datasets/player_clustering.parquet"
RANDOM_STATE = 42

#Read dataset
data = pd.read_parquet(DATASET_PATH)

#Feature selection
X = data.drop(colums ="")
y = data[""]

#Training and test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = RANDOM_STATE,
    ) 

#Mlflow experiment
experiment_name = get_experiment_name("p5")

configure_tracking()

with mlflow_run(
    experiment_name,
    run_name = "DBSCAN Baseline"
):
    #Log dataset context
    log_dataset_context(
        DATASET_PATH,
        row_count = len(data),
        feature_count = X.shape[1],
    )
    
    #Log split config
    log_split_config(
        split_strategy = "train_test_split",
        test_size = 0.2,
        random_state = RANDOM_STATE,
    )
    
    #Model
    model = DBSCAN(
        random_state = RANDOM_STATE,
    )
    
    #Training the model
    model.fit(
        X_train,
        y_train,
    )
    
    #Predictions
    y_pred = model.predict(X_test)
    
    #Metrics
    metrics = "Lo que toque pa este modelo"
    
    #Log metrics
    log_metrics(metrics)
    
    #Log model and artifacts
    log_model_and_artifacts(model)
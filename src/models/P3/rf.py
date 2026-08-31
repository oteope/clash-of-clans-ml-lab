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
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

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


# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_PATH = (
    "data/datasets/clan_war_performance_regression.parquet"
)
RANDOM_STATE = 42
TEST_SIZE = 0.2


# =============================================================================
# LOAD DATASET
# =============================================================================

data = pd.read_parquet(DATASET_PATH)


# =============================================================================
# FEATURES AND TARGET
# =============================================================================

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


# =============================================================================
# TRAIN / TEST SPLIT
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)


# =============================================================================
# MLFLOW
# =============================================================================

configure_tracking()

experiment_name = get_experiment_name("p3")


with mlflow_run(
    experiment_name,
    run_name="Linear regression baseline",
):

    # -------------------------------------------------------------------------
    # Dataset context
    # -------------------------------------------------------------------------

    log_dataset_context(
        DATASET_PATH,
        row_count=len(data),
        feature_count=X.shape[1],
        target="war_success_rate",
    )

    # -------------------------------------------------------------------------
    # Train / test split
    # -------------------------------------------------------------------------

    log_split_config(
        split_strategy="train_test_split",
        test_size=TEST_SIZE,
        random_seed=RANDOM_STATE,
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    model = LinearRegression()

    # -------------------------------------------------------------------------
    # Model parameters
    # -------------------------------------------------------------------------

    log_model_params({
        "fit_intercept": model.fit_intercept,
        "copy_X": model.copy_X,
        "n_jobs": model.n_jobs,
        "positive": model.positive,
    })

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    model.fit(
        X_train,
        y_train,
    )

    # -------------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------------

    y_pred = model.predict(
        X_test
    )

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    metrics = {
        "mae": mean_absolute_error(
            y_test,
            y_pred,
        ),

        "rmse": np.sqrt(
            mean_squared_error(
                y_test,
                y_pred,
            )
        ),

        "r2": r2_score(
            y_test,
            y_pred,
        ),
    }

    # -------------------------------------------------------------------------
    # Log metrics
    # -------------------------------------------------------------------------

    log_metrics(
        metrics
    )

    # -------------------------------------------------------------------------
    # Log model and artifacts
    # -------------------------------------------------------------------------

    log_model_and_artifacts(
        model
    )

import mlflow
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
)

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
    run_name="Linear Regression GridSearchCV",
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
    # Base model
    # -------------------------------------------------------------------------

    model = LinearRegression()

    # -------------------------------------------------------------------------
    # Hyperparameter grid
    # -------------------------------------------------------------------------

    param_grid = {
        "fit_intercept": [True, False],
        "positive": [True, False],
    }

    # -------------------------------------------------------------------------
    # Grid Search
    # -------------------------------------------------------------------------

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    search.fit(
        X_train,
        y_train,
    )

    # -------------------------------------------------------------------------
    # Best model
    # -------------------------------------------------------------------------

    best_model = search.best_estimator_

    best_params = search.best_params_

    # -------------------------------------------------------------------------
    # Log model parameters
    # -------------------------------------------------------------------------

    log_model_params({
        "fit_intercept": best_params["fit_intercept"],
        "positive": best_params["positive"],
        "copy_X": best_model.copy_X,
        "n_jobs": best_model.n_jobs,
        "tuning_method": "GridSearchCV",
        "cv": 3,
        "scoring": "neg_mean_absolute_error",
    })

    # -------------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------------

    y_pred = best_model.predict(
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
    # Log best model and artifacts
    # -------------------------------------------------------------------------

    log_model_and_artifacts(
        best_model
    )
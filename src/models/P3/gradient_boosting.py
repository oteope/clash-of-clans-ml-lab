import optuna
import mlflow
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
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

#Config
DATASET_PATH = "data\datasets\clan_war_performance_regression.parquet"
N_TRIALS = 30
RANDOM_STATE = 42

#Preparing the dataset
print("Loading dataset...")
data = pd.read_parquet(DATASET_PATH)

#Defining the features and targets
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

#Preparing the dataset
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = RANDOM_STATE,
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full,
    y_train_full,
    test_size = 0.2,
    random_state = RANDOM_STATE
)

#Mlflow experiment
configure_tracking()

experiment_name = get_experiment_name("p3")

#Optuna

def objective(trial):
    params = {
    "n_estimators": trial.suggest_int(
        "n_estimators",
        100,
        500,
        step=50,
    ),

    "learning_rate": trial.suggest_float(
        "learning_rate",
        0.01,
        0.20,
        log=True,
    ),

    "max_depth": trial.suggest_int(
        "max_depth",
        2,
        8,
    ),

    "min_samples_split": trial.suggest_int(
        "min_samples_split",
        2,
        20,
    ),

    "min_samples_leaf": trial.suggest_int(
        "min_samples_leaf",
        1,
        10,
    ),

    "subsample": trial.suggest_float(
        "subsample",
        0.6,
        1.0,
    ),

    "max_features": trial.suggest_categorical(
        "max_features",
        [None, "sqrt", "log2"],
    ),

    "random_state": RANDOM_STATE,

    }
    
    with mlflow_run(
        experiment_name,
        run_name = f"Gradient boosting v{trial.number}"
    ):
        log_model_params({
            **params,
            "optuna_trial": trial.number,
        
        })
        
        #Model
        model = GradientBoostingRegressor(
            **params,
        )
        
        #Training
        model.fit(X_train,
                y_train)
        
        #Prediction
        y_pred = model.predict(X_valid)
        
        #Metrics
        metrics = {
                "mae": mean_absolute_error(
                    y_valid,
                    y_pred,
                ),
            
                "rmse": np.sqrt(
                    mean_squared_error(
                         y_valid,
                        y_pred,
                    )
                ),
            
                 "r2": r2_score(
                    y_valid,
                    y_pred,
                ),
             }
        
        #Logging the metrics
        log_metrics(metrics)
        
        trial.set_user_attr(
                        "mae",
                        metrics["mae"],
                    )
            
        trial.set_user_attr(
                "rmse",
                 metrics["rmse"],
            )
            
        trial.set_user_attr(
                "r2",
                metrics["r2"],
             )
        
        # We minimize MAE
        return metrics["mae"]

# ============================================================
# Run Optuna
# ============================================================

print()
print("=" * 60)
print("Starting Optuna hyperparameter optimization")
print("=" * 60)
print()

study = optuna.create_study(
    direction="minimize",
    study_name="p3_gradient_boosting",
)


study.optimize(
    objective,
    n_trials=N_TRIALS,
)


# ============================================================
# Best parameters
# ============================================================

print()
print("=" * 60)
print("OPTUNA RESULTS")
print("=" * 60)

print(f"Best trial: {study.best_trial.number}")
print(f"Best MAE: {study.best_value}")

print()
print("Best parameters:")

for parameter, value in study.best_params.items():
    print(f"  {parameter}: {value}")


# ============================================================
# Final model
# ============================================================

print()
print("Training final model with best parameters...")


best_model = GradientBoostingRegressor(
    **study.best_params,
    random_state=RANDOM_STATE,
)


best_model.fit(
    X_train_full,
    y_train_full,
)


# ============================================================
# Final test evaluation
# ============================================================

y_pred_test = best_model.predict(X_test)


final_metrics = {
    "mae": mean_absolute_error(
        y_test,
        y_pred_test,
    ),

    "rmse": np.sqrt(
        mean_squared_error(
            y_test,
            y_pred_test,
        )
    ),

    "r2": r2_score(
        y_test,
        y_pred_test,
    ),
}


print()
print("=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

for metric, value in final_metrics.items():
    print(f"{metric}: {value}")
    
# ============================================================
# Log final champion model
# ============================================================

with mlflow_run(
    experiment_name,
    run_name = "Gradient Boosting Champion"
):
    log_dataset_context(
        DATASET_PATH,
        row_count = len(data),
        feature_count = X.shape[1],
        target = "war_success_rate"
    )
    
    log_split_config(
        split_strategy = "train_test_split",
        test_size = 0.2,
        random_seed = RANDOM_STATE,
    )
    
    log_model_params({
        **study.best_params,
        "optuna_trials":N_TRIALS,
        "model_type":"GradientBoostingRegressor",
        "tuning_method": "optuna",
    })
    
    log_metrics(final_metrics)
    
    log_model_and_artifacts(
        best_model,
    )
    
print()
print("Optimization completed successfully.")
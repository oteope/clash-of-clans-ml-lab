import optuna
import mlflow
import xgboost as xgb
import pandas as pd
import numpy as np

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
DATASET_PATH = "data/datasets/clan_rank_regression_without_trophies.parquet"
N_TRIALS = 20
RANDOM_STATE = 42

#Preparing the dataset
print("Loading dataset...")
data = pd.read_parquet(DATASET_PATH)


#Defining the features and target
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

#Preparing the train and test set
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = RANDOM_STATE,
)

X_train, X_valid, y_train, y_valid = train_test_split(X_train_full,
                                                    y_train_full,
                                                    test_size=0.2,
                                                    random_state=42,)

#Mlflow experiment
configure_tracking()

experiment_name = get_experiment_name("p2")

#Optuna

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            300,
            1000,
            step=100,
        ),
        
        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.10,
            log=True
        ),
        
        "max_depth": trial.suggest_int(
            "max_depth",
            3,
            8,
        ),
        
        "min_child_weight": trial.suggest_int(
            "min_child_weight",
            1,
            10,
        ),
        
        "subsample": trial.suggest_float(
            "subsample",
            0.7,
            1.0,
         ),
        
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.7,
            1.0,
         ),
        
            "random_state": RANDOM_STATE,
    }

    with mlflow_run(
        experiment_name,
        run_name = f"XGBoost Optuna Trial {trial.number} - without trophies"
    ):
    
        log_model_params({
            **params,
            "optuna_trial": trial.number,
        }) 
        #Model
        model = xgb.XGBRegressor(
                **params,
             )
    
        #Training
        model.fit(X_train,
              y_train
              )
    
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
    
        #Logging metrics
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
    study_name="p2_xgboost_regression",
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


best_model = xgb.XGBRegressor(
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
    run_name="Optuna Champion XGBoost - without trophies",
):

    log_dataset_context(
        DATASET_PATH,
        row_count=len(data),
        feature_count=X.shape[1],
        target="clan_rank",
    )

    log_split_config(
        split_strategy="train_test_split",
        test_size=0.20,
        random_seed=RANDOM_STATE,
    )

    log_model_params({
        **study.best_params,
        "optuna_trials": N_TRIALS,
        "model_type": "XGBoost",
        "tuning_method": "Optuna",
    })

    log_metrics(final_metrics)

    log_model_and_artifacts(
        best_model,
    )


print()
print("Optimization completed successfully.")
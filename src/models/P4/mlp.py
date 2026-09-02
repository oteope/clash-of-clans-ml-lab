import torch
import torch.nn as nn
import optuna
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
from sklearn.metrics import train_test_split


# ============================================================
# Configuration
# ============================================================

DATASET_PATH = "data/datasets/clan_performance_classification.parquet"
N_TRIALS = 50
RANDOM_STATE = 42

# ============================================================
# Load dataset
# ============================================================

print("Loading dataset...")

data = pd.read_parquet(DATASET_PATH)

# ============================================================
# Features / target
# ============================================================

X = data.drop(columns="performance_class").select_dtypes(include="number")

y = data["performance_class"]

# ============================================================
# Train / validation / test split
# ============================================================

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = RANDOM_STATE,
)

X_train, X_valid, y_train, y_valid = train_test_split (
    X_train_full,
    y_train_full,
    test_size = 0.2,
    random_state = RANDOM_STATE,
)

# ============================================================
# MLflow
# ============================================================

configure_tracking()

experiment_name = get_experiment_name("p4")

# ============================================================
# Optuna objective
# ============================================================
def objective(trial):
    params = {
        "hidden_layer_size": trial.suggest_int(
            "hiden_layer_size",
            1,
            100,
        ),
        
        "activation": trial.suggest_bool(
            "activation",
            "relu",
            "tahn",   
        ),
        
        "alpha": trial.suggest_float(
            "alpha",
            0.1,
            10,
        ),
        
        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.1,
            10,
        ),

        "solver": "adam",
        
        "max_iter_epochs": trial.suggest_int(
            "max_iter_epochs",
            50,
            200,
        ),
        
        "batch_size": trial.suggest_int(
            "batch_size",
            32,
            256,
        ),
        
        "random_state": RANDOM_STATE,
    }
    
    # --------------------------------------------------------
    # MLflow run for this Optuna trial
    # --------------------------------------------------------  
    
    with mlflow_run(
        experiment_name,
        run_name = f"MLP trial {trial.number}",
    ):
        
        #Log model params
        log_model_params({
            **params,
            "optuna_trial": trial.number,
        })
        
        #Model
        model = nn(**params)
        
        #Training
        model.fit(
            X_train,
            y_train,
        )
        
        # Validation metrics
        y_valid_pred = model.predict(X_valid)

        valid_metrics = {
            "accuracy": accuracy_score(
                y_valid,
                y_valid_pred,
            ),
            "balanced_accuracy": balanced_accuracy_score(
                y_valid,
                y_valid_pred,
            ),
            "f1_macro": f1_score(
                y_valid,
                y_valid_pred,
                average="macro",
            ),
            "f1_weighted": f1_score(
                y_valid,
                y_valid_pred,
                average="weighted",
            ),
}

        # Tell Optuna the objective value
        trial.set_user_attr(
            "accuracy",
            valid_metrics["accuracy"],
        )
        
        trial.set_user_attr(
            "balanced_accuracy",
            valid_metrics["balanced_accuracy"],
        )

        trial.set_user_attr(
            "f1_macro",
            valid_metrics["f1_macro"],
        )

        trial.set_user_attr(
            "f1_weighted",
            valid_metrics["f1_weighted"],
        )       

        # We maximize F1 Macro
        return valid_metrics["f1_macro"] 
    
# ============================================================
# Run Optuna
# ============================================================

print()
print("=" * 60)
print("Starting Optuna hyperparameter optimization")
print("=" * 60)
print()

study = optuna.create_study(
    direction="maximize",
    study_name="p4_mlp_classification",
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
print(f"Best F1 Macro: {study.best_value}")

print()
print("Best parameters:")

for parameter, value in study.best_params.items():
    print(f"  {parameter}: {value}")

# ============================================================
# Final model
# ============================================================

print()
print("Training final model with best parameters...")

best_model = xgb.XGBClassifier(
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

y_pred_test = best_model.predict(
    X_test
)

final_metrics = {
    "accuracy": accuracy_score(
        y_test,
        y_pred_test,
    ),

    "balanced_accuracy": balanced_accuracy_score(
        y_test,
        y_pred_test,
    ),

    "f1_macro": f1_score(
        y_test,
        y_pred_test,
        average="macro",
    ),

    "f1_weighted": f1_score(
        y_test,
        y_pred_test,
        average="weighted",
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
    run_name="Optuna Champion MLP",
):

    log_dataset_context(
        DATASET_PATH,
        row_count=len(data),
        feature_count=X.shape[1],
        target="performance_class",
    )

    log_split_config(
        split_strategy="train_test_split",
        test_size=0.2,
        random_seed=RANDOM_STATE,
    )

    log_model_params({
        **study.best_params,
        "optuna_trials": N_TRIALS,
        "model_type": "MLP",
        "tuning_method": "Optuna",
        "optimization_metric": "f1_macro",
    })
    
    cm = confusion_matrix(
        y_test,
        y_pred_test,
    )

    log_metrics(
        final_metrics
    )

    log_model_and_artifacts(
        best_model,
        cm,
    )

print()
print("Optimization completed successfully.")
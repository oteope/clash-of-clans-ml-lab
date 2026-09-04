import optuna
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

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

X = data.drop(columns="performance_class").select_dtypes(
    include=["number", "bool"]
)

y = data["performance_class"]


# ============================================================
# Encode target
# ============================================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

print("Target classes:")
for encoded_value, class_name in enumerate(label_encoder.classes_):
    print(f"  {encoded_value}: {class_name}")


# ============================================================
# Train / validation / test split
# ============================================================

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.2,
    random_state=RANDOM_STATE,
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

    # --------------------------------------------------------
    # Hyperparameters
    # --------------------------------------------------------

    hidden_layer_size = trial.suggest_int(
        "hidden_layer_size",
        16,
        128,
        step=16,
    )

    activation = trial.suggest_categorical(
        "activation",
        ["relu", "tanh"],
    )

    alpha = trial.suggest_float(
        "alpha",
        1e-5,
        1e-1,
        log=True,
    )

    learning_rate_init = trial.suggest_float(
        "learning_rate_init",
        1e-4,
        1e-2,
        log=True,
    )

    max_iter = trial.suggest_int(
        "max_iter",
        100,
        300,
        step=50,
    )

    batch_size = trial.suggest_categorical(
        "batch_size",
        [32, 64, 128, 256],
    )

    params = {
        "hidden_layer_sizes": (hidden_layer_size,),
        "activation": activation,
        "alpha": alpha,
        "learning_rate_init": learning_rate_init,
        "solver": "adam",
        "max_iter": max_iter,
        "batch_size": batch_size,
        "random_state": RANDOM_STATE,
    }


    # --------------------------------------------------------
    # MLflow run for this Optuna trial
    # --------------------------------------------------------

    with mlflow_run(
        experiment_name,
        run_name=f"MLP trial {trial.number}",
    ):

        # Log model parameters
        log_model_params({
            **params,
            "optuna_trial": trial.number,
        })


        # ----------------------------------------------------
        # Model pipeline
        # ----------------------------------------------------

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("mlp", MLPClassifier(**params)),
            ]
        )


        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train,
        )


        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        y_valid_pred = model.predict(
            X_valid,
        )


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


        # ----------------------------------------------------
        # Store validation metrics in Optuna
        # ----------------------------------------------------

        trial.set_user_attr(
            "accuracy",
            valid_metrics["accuracy"],
        )

        trial.set_user_attr(
            "balanced_accuracy",
            valid_metrics["balanced_accuracy"],
        )

        trial.set_user_attr(
            "f1_weighted",
            valid_metrics["f1_weighted"],
        )


        # ----------------------------------------------------
        # Optimization objective
        # ----------------------------------------------------

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
# Prepare final model parameters
# ============================================================

best_params = {
    "hidden_layer_sizes": (
        study.best_params["hidden_layer_size"],
    ),

    "activation": study.best_params["activation"],

    "alpha": study.best_params["alpha"],

    "learning_rate_init": (
        study.best_params["learning_rate_init"]
    ),

    "solver": "adam",

    "max_iter": study.best_params["max_iter"],

    "batch_size": study.best_params["batch_size"],

    "random_state": RANDOM_STATE,
}


# ============================================================
# Final model
# ============================================================

print()
print("Training final model with best parameters...")

best_model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(**best_params)),
    ]
)

best_model.fit(
    X_train_full,
    y_train_full,
)


# ============================================================
# Final test prediction
# ============================================================

y_pred_test = best_model.predict(
    X_test,
)


# ============================================================
# Decode target labels
# ============================================================

y_test_labels = label_encoder.inverse_transform(
    y_test
)

y_pred_test_labels = label_encoder.inverse_transform(
    y_pred_test.astype(int)
)


# ============================================================
# Final test evaluation
# ============================================================

final_metrics = {
    "accuracy": accuracy_score(
        y_test_labels,
        y_pred_test_labels,
    ),

    "balanced_accuracy": balanced_accuracy_score(
        y_test_labels,
        y_pred_test_labels,
    ),

    "f1_macro": f1_score(
        y_test_labels,
        y_pred_test_labels,
        average="macro",
    ),

    "f1_weighted": f1_score(
        y_test_labels,
        y_pred_test_labels,
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
# Confusion matrix
# ============================================================

cm = confusion_matrix(
    y_test_labels,
    y_pred_test_labels,
    labels=label_encoder.classes_,
)


print()
print("Confusion Matrix:")
print(cm)


# ============================================================
# MLflow class metadata
# ============================================================

class_mapping = {
    str(encoded_value): class_name
    for encoded_value, class_name
    in enumerate(label_encoder.classes_)
}


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
        **best_params,
        "optuna_trials": N_TRIALS,
        "model_type": "MLPClassifier",
        "tuning_method": "Optuna",
        "optimization_metric": "f1_macro",
        "label_mapping": str(class_mapping),
    })

    log_metrics(
        final_metrics,
    )

    log_model_and_artifacts(
    best_model,
    confusion_matrix=cm,
    class_names=label_encoder.classes_.tolist(),
    )


print()
print("Optimization completed successfully.")
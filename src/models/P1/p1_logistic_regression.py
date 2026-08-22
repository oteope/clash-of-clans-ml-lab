import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
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


# Loading the dataset
DATASET_PATH = "data/dataset/role_classification.parquet"
data = pd.read_parquet(DATASET_PATH)

# Defining features and target variable
X = data.drop(columns=["role"])
y = data["role"]

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# MLflow experiment
experiment_name = get_experiment_name("p1")

with mlflow_run(
    experiment_name,
    run_name="logistic_regression_baseline",
):

    # Dataset information
    log_dataset_context(
        DATASET_PATH,
        row_count=len(data),
        feature_count=X.shape[1],
        target="role",
    )

    # Split configuration
    log_split_config(
        "train_test_split",
        0.2,
        42,
        {},
    )

    # Model
    model = LogisticRegression(max_iter=1000)

    # Model hyperparameters
    log_model_params(
        {
            "max_iter": 1000,
        }
    )

    # Training
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
    }

    # Save metrics to MLflow
    log_metrics(metrics)

    # Save model
    log_model_and_artifacts(model)
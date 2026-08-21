# MLflow Tracking Setup

This repository uses local MLflow Tracking to manage experiments.

## Dependencies
Install dependencies with:
```powershell
pip install -r requirements.txt
```

## Start Tracking Server
Run the MLflow tracking server locally:
```powershell
mlflow server `
  --backend-store-uri sqlite:///mlflow/mlflow.db `
  --default-artifact-root ./mlflow/mlruns `
  --host 127.0.0.1 `
  --port 5000
```

The server will be available at http://127.0.0.1:5000.

## Configure Environment
Set the MLflow tracking URI so client code can connect:
```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
```

## Experiment Naming
Experiments are created automatically according to problem codes:
- p1 -> p1_role_classification
- p2 -> p2_clan_rank
- p3 -> p3_war_performance
- p4 -> p4_clan_performance_classification
- p5 -> p5_player_clustering

## Usage in Code
Import utilities:
```python
from mlflow_tracking.tracking_utils import configure_tracking, mlflow_run, log_dataset_context, log_split_config, log_model_params, log_metrics, log_model_and_artifacts
from mlflow_tracking.experiments import get_experiment_name
```

Example:
```python
configure_tracking()  # usa MLFLOW_TRACKING_URI o sqlite local por defecto

experiment_name = get_experiment_name("p1")
with mlflow_run(experiment_name, run_name="my_run"):
    log_dataset_context("path/to/data.parquet", row_count=1000, feature_count=20, target="label")
    log_split_config("GroupKFold", "clan_tag", 42, {"imputer": "median"})
    log_model_params({"learning_rate": 0.1})
    # ... train model, evaluate
    log_metrics({"accuracy": 0.95})
    log_model_and_artifacts(model, confusion_matrix=cm, class_names=["class0", "class1"])
```

## Smoke Test
Run the smoke test:
```powershell
python -m unittest tests.test_mlflow_smoke
```

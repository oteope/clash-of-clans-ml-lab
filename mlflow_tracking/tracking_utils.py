"""Reusable MLflow tracking utilities for local experiments."""
from __future__ import annotations

import hashlib
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Union

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

DEFAULT_ARTIFACT_ROOT = "mlflow/mlruns"
DEFAULT_BACKEND_STORE_URI = "sqlite:///mlflow/mlflow.db"

STANDARD_TAGS = {
    "project": "clash-of-clans-ml-lab",
    "tracking_version": "1.0.0",
}

def get_or_create_experiment(experiment_name: str) -> str:
    """Get existing experiment ID or create a new one."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_name)
    return experiment_id

def _hash_file(path: Union[str, Path], algo: str = "sha256") -> str:
    """Return hex digest of a file."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def log_dataset_context(
    dataset_path: Union[str, Path],
    row_count: int,
    feature_count: int,
    target: str,
    parquet_hash: Optional[str] = None,
    modification_time: Optional[str] = None,
) -> None:
    """Log dataset metadata as MLflow params/tags."""
    dataset_path = Path(dataset_path)
    if parquet_hash is None and dataset_path.exists():
        parquet_hash = _hash_file(dataset_path)
    if modification_time is None and dataset_path.exists():
        modification_time = str(pd.Timestamp(dataset_path.stat().st_mtime, unit="s"))
    mlflow.log_params({
        "row_count": row_count,
        "feature_count": feature_count,
        "target": target,
    })
    mlflow.set_tags({
        "dataset_path": str(dataset_path),
        "parquet_hash": parquet_hash or "N/A",
        "parquet_modified_time": modification_time or "N/A",
    })

def log_split_config(
    split_strategy: str,
    grouping_col: Optional[str],
    random_seed: int,
    preprocessing_config: Dict[str, Any],
) -> None:
    """Log data split and preprocessing configuration."""
    mlflow.log_params({
        "split_strategy": split_strategy,
        "grouping_col": grouping_col or "none",
        "random_seed": random_seed,
        "preprocessing_config": json.dumps(preprocessing_config, sort_keys=True),
    })

def log_model_params(params: Dict[str, Any]) -> None:
    """Log model hyperparameters."""
    for key, value in params.items():
        mlflow.log_param(f"model__{key}", value)

def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """Log evaluation metrics."""
    mlflow.log_metrics(metrics, step=step)

def log_model_and_artifacts(
    model: Any,
    artifact_path: str = "model",
    confusion_matrix: Optional[Any] = None,
    class_names: Optional[list] = None,
    extra_artifacts: Optional[Dict[str, str]] = None,
) -> None:
    """Log a trained model and optional artifacts (e.g., confusion matrix)."""
    mlflow.sklearn.log_model(model, artifact_path)
    if confusion_matrix is not None:
        try:
            import matplotlib.pyplot as plt
            if class_names is None:
                class_names = list(range(confusion_matrix.shape[0]))
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(confusion_matrix, cmap="Blues")
            ax.figure.colorbar(im, ax=ax)
            ax.set(
                xticks=range(len(class_names)),
                yticks=range(len(class_names)),
                xticklabels=class_names,
                yticklabels=class_names,
                ylabel="Actual",
                xlabel="Predicted",
            )
            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    ax.text(j, i, str(confusion_matrix[i, j]), ha="center", va="center", color="w")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                fig.savefig(tmp.name, bbox_inches="tight")
                plt.close(fig)
                mlflow.log_artifact(tmp.name, "artifacts/confusion_matrix.png")
        except ImportError:
            pass
    if extra_artifacts:
        for local_path, artifact_path in extra_artifacts.items():
            mlflow.log_artifact(local_path, artifact_path)

@contextmanager
def mlflow_run(experiment_name: str, run_name: Optional[str] = None):
    """Context manager for an MLflow run scoped to an experiment."""
    experiment_id = get_or_create_experiment(experiment_name)
    run = mlflow.start_run(run_name=run_name)
    try:
        # Set standard tags at run start
        mlflow.set_tags(STANDARD_TAGS)
        yield run
    finally:
        mlflow.end_run()

def log_cv_aggregated_metrics(metrics: Dict[str, float], prefix: str = "cv") -> None:
    """Log aggregated cross-validation metrics under a given prefix."""
    prefixed = {f"{prefix}__{k}": v for k, v in metrics.items()}
    mlflow.log_metrics(prefixed)

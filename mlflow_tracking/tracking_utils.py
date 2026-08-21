"""Reusable MLflow tracking utilities for local experiments."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Union
from urllib.parse import urlparse

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

def _validate_tracking_uri(uri: str) -> None:
    """Validate that the tracking URI is local and supported."""
    if not uri:
        raise ValueError("Tracking URI cannot be empty.")
    parsed = urlparse(uri)
    if not parsed.scheme:
        raise ValueError(f"Invalid tracking URI (missing scheme): {uri}")
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https", "sqlite", "file"):
        raise ValueError(f"Unsupported tracking URI scheme: {scheme}")
    if scheme in ("http", "https"):
        host = parsed.hostname or ""
        if host not in ("localhost", "127.0.0.1"):
            raise ValueError(
                "Remote HTTP tracking URIs are not allowed. Use localhost or 127.0.0.1."
            )
    # sqlite:/// and file:/// are inherently local; no extra validation needed.

def _ensure_sqlite_parent_directory(tracking_uri: str) -> None:
    """Ensure that the parent directory of a SQLite tracking URI exists."""
    parsed = urlparse(tracking_uri)
    if parsed.scheme.lower() != "sqlite":
        return

    path = parsed.path
    if path.startswith("//"):
        # sqlite:////absolute/path -> /absolute/path
        db_path = Path(path[2:])
    elif path.startswith("/"):
        # sqlite:///relative/path -> relative/path
        db_path = Path(path[1:])
    else:
        db_path = Path(path)

    parent = db_path.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)

def _ensure_artifact_root() -> None:
    """Ensure the default artifact root directory exists."""
    Path(DEFAULT_ARTIFACT_ROOT).mkdir(parents=True, exist_ok=True)

def configure_tracking(tracking_uri: Optional[str] = None) -> None:
    """
    Configure MLflow tracking explicitly.

    If ``tracking_uri`` is None, it reads the ``MLFLOW_TRACKING_URI``
    environment variable. If that is also unset, it falls back to the default
    local SQLite store.

    The URI is validated to ensure only local tracking backends are used.
    """
    if tracking_uri is None:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        tracking_uri = DEFAULT_BACKEND_STORE_URI
    _validate_tracking_uri(tracking_uri)

    # Ensure SQLite database parent directory exists before MLflow uses it.
    _ensure_sqlite_parent_directory(tracking_uri)

    mlflow.set_tracking_uri(tracking_uri)

def get_or_create_experiment(experiment_name: str) -> str:
    """Get existing experiment ID or create a new one."""
    _ensure_artifact_root()
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=str(Path(DEFAULT_ARTIFACT_ROOT).resolve()),
        )
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
    compute_hash: bool = False,
) -> None:
    """
    Log dataset metadata as MLflow params/tags.

    By default, only file size and modification timestamp are logged.
    SHA-256 hash is computed only when ``compute_hash`` is True or a
    pre-computed ``parquet_hash`` is provided.
    """
    dataset_path = Path(dataset_path)
    file_size: Optional[int] = None
    if dataset_path.exists():
        stat = dataset_path.stat()
        file_size = stat.st_size
        if modification_time is None:
            modification_time = str(pd.Timestamp(stat.st_mtime, unit="s"))

    if compute_hash and parquet_hash is None and dataset_path.exists():
        parquet_hash = _hash_file(dataset_path)

    mlflow.log_params({
        "row_count": row_count,
        "feature_count": feature_count,
        "target": target,
    })

    tags = {
        "dataset_path": str(dataset_path),
        "file_size_bytes": str(file_size) if file_size is not None else "N/A",
        "parquet_modified_time": modification_time or "N/A",
        "parquet_hash": parquet_hash or "N/A",
    }
    mlflow.set_tags(tags)

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
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required to log confusion matrices. "
                "Install it with 'pip install matplotlib'."
            ) from exc

        if class_names is None:
            class_names = list(range(confusion_matrix.shape[0]))

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(confusion_matrix, cmap="Blues")
        fig.colorbar(im, ax=ax)
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
                ax.text(
                    j,
                    i,
                    str(confusion_matrix[i, j]),
                    ha="center",
                    va="center",
                    color="w",
                )
        # Use log_figure to store the figure as a stable artifact file.
        mlflow.log_figure(fig, "confusion_matrix.png", artifact_path="artifacts")
        plt.close(fig)

    if extra_artifacts:
        for local_path, artifact_path in extra_artifacts.items():
            mlflow.log_artifact(local_path, artifact_path)

@contextmanager
def mlflow_run(experiment_name: str, run_name: Optional[str] = None):
    """Context manager for an MLflow run scoped to an experiment."""
    experiment_id = get_or_create_experiment(experiment_name)
    run = mlflow.start_run(run_name=run_name)
    try:
        mlflow.set_tags({
            **STANDARD_TAGS,
            "problem_name": experiment_name,
        })
        yield run
    finally:
        mlflow.end_run()

def log_cv_aggregated_metrics(metrics: Dict[str, float], prefix: str = "cv") -> None:
    """Log aggregated cross-validation metrics under a given prefix."""
    prefixed = {f"{prefix}__{k}": v for k, v in metrics.items()}
    mlflow.log_metrics(prefixed)

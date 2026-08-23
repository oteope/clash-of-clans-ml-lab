import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Ensure the repository root is importable when running directly from the
# repository root: python src/results_p1.py
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ---------------------------------------------------------------------------
# Existing project infrastructure
# ---------------------------------------------------------------------------
from mlflow_tracking.tracking_utils import configure_tracking
from mlflow_tracking.experiments import get_experiment_name
from src.features.problem1.build_role_dataset import (
    load_small_tables,
    assemble_role_dataset,
    PROCESSED_DIR,
)
from src.features.problem1.player_features import build_player_features_from_files
from src.features.problem1.player_clan_features import compute_clan_relative_features

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(ROOT_DIR) / "src" / "results" / "P1"
CLASS_ORDER = ["admin", "coLeader", "leader", "member"]
TARGET_COLUMN = "role"

# MLflow run IDs supplied by the project owner for the final models
XGBOOST_RUN_ID = "11a727b624134920896bfefffc5abc90"
RF_RUN_ID = "40c9a36cbdd044fe83899ac0f8694178"
LOGISTIC_RUN_NAME = "logistic_regression_tuned_v3"


def _find_run_id_by_name(name: str) -> str:
    """Return the run_id for a run whose MLflow run name is ``name``.

    This function uses the project's existing ``configure_tracking`` /
    ``get_experiment_name`` utilities to locate the correct experiment and run.
    """
    configure_tracking()

    # First try to narrow the search to the P1 experiment.
    try:
        experiment_name = get_experiment_name("p1")
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is not None:
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.mlflow.runName = '{name}'",
            )
            if not runs.empty:
                return str(runs.iloc[0]["run_id"])
    except Exception:
        pass

    # Fallback: search all accessible experiments.
    all_runs = mlflow.search_runs(filter_string=f"tags.mlflow.runName = '{name}'")
    if all_runs.empty:
        raise RuntimeError(f"Could not find an MLflow run with name: {name}")

    return str(all_runs.iloc[0]["run_id"])


def _load_model_from_run(run_id: str):
    """Load a model from an existing MLflow run using the run URI."""
    model_uri = f"runs:/{run_id}/model"
    return mlflow.pyfunc.load_model(model_uri)


def _parse_preprocessing_config(params: Dict[str, str]) -> Dict[str, Any]:
    """Extract and parse the preprocessing_config parameter from a run.

    The project logs this parameter via ``log_split_config`` as a JSON string.
    """
    raw = params.get("preprocessing_config")
    if raw is None:
        raise RuntimeError(
            "The MLflow run does not contain a 'preprocessing_config' parameter. "
            "Unable to recover the exact data preprocessing configuration."
        )

    try:
        config = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "The 'preprocessing_config' parameter is not valid JSON: " + str(raw)
        ) from exc

    if not isinstance(config, dict):
        raise RuntimeError(
            "The 'preprocessing_config' parameter must be a JSON object."
        )

    return config


def _get_test_size_from_config(config: Dict[str, Any]) -> float:
    """Return test_size from a preprocessing configuration dict.

    The config may store the value directly, inside a nested ``split`` dict,
    or through common aliases.  Raises if it cannot be found.
    """
    possible_direct_keys = ("test_size", "test_ratio", "test_fraction")
    for key in possible_direct_keys:
        if key in config:
            try:
                return float(config[key])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Invalid test_size value in preprocessing_config: {config[key]}"
                ) from exc

    # Check nested 'split' or 'data_split' sections.
    for nested_key in ("split", "data_split", "train_test_split"):
        nested = config.get(nested_key)
        if isinstance(nested, dict):
            for key in possible_direct_keys:
                if key in nested:
                    try:
                        return float(nested[key])
                    except (TypeError, ValueError) as exc:
                        raise RuntimeError(
                            f"Invalid test_size value in preprocessing_config "
                            f"under '{nested_key}': {nested[key]}"
                        ) from exc

    raise RuntimeError(
        "The 'preprocessing_config' parameter does not contain test_size. "
        "Unable to recover the exact train/test split."
    )


def _get_random_state_from_params(params: Dict[str, str]) -> int:
    """Return random_state from common MLflow parameter names."""
    for key in ("random_seed", "seed", "random_state", "random_state_seed"):
        if key in params:
            try:
                return int(params[key])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Invalid random_state value in parameter '{key}': {params[key]}"
                ) from exc

    raise RuntimeError(
        "The MLflow run does not contain random_seed/seed/random_state. "
        "Unable to recover the exact train/test split."
    )


def _get_consistent_run_config(run_ids: List[str]) -> Tuple[float, int, Dict[str, Any]]:
    """Retrieve and validate split/preprocessing config across all runs.

    Raises if any required parameter is missing or if the runs disagree.
    Returns (test_size, random_state, preprocessing_config).
    """
    client = mlflow.tracking.MlflowClient()

    test_sizes: List[float] = []
    random_states: List[int] = []
    configs: List[Dict[str, Any]] = []

    for rid in run_ids:
        run = client.get_run(rid)
        params = run.data.params

        random_state = _get_random_state_from_params(params)
        config = _parse_preprocessing_config(params)
        test_size = _get_test_size_from_config(config)

        test_sizes.append(test_size)
        random_states.append(random_state)
        configs.append(config)

    if len(set(test_sizes)) != 1:
        raise RuntimeError(
            "The MLflow runs contain inconsistent test_size values: "
            f"{test_sizes}"
        )

    if len(set(random_states)) != 1:
        raise RuntimeError(
            "The MLflow runs contain inconsistent random_state values: "
            f"{random_states}"
        )

    # Ensure the relevant parts of the configs match; use the first run's
    # config as canonical.
    canonical_config = configs[0]
    for idx, cfg in enumerate(configs[1:], start=1):
        if cfg != canonical_config:
            # Allow differences in non-relevant keys, but any difference that
            # affects preprocessing or split must be caught above.
            # Here we compare the full dicts to be conservative.
            raise RuntimeError(
                "The MLflow runs contain different preprocessing_config values."
            )

    return test_sizes[0], random_states[0], canonical_config


def _build_dataset() -> pd.DataFrame:
    """Build the P1 dataset using the existing project pipeline.

    This function deliberately reuses the project's own functions and file
    layout.  No new split is created here.
    """
    small_tables = load_small_tables()
    clans_df = small_tables["clans"]
    clan_members_df = small_tables["clan_members"]

    # Reuse the same feature construction that the original project uses.
    pf = build_player_features_from_files(PROCESSED_DIR, batch_size=100_000)
    pcf = compute_clan_relative_features(clan_members_df, pf)

    # Assemble the final role classification dataset.
    dataset = assemble_role_dataset(
        pf=pf,
        pcf=pcf,
        clan_members_df=clan_members_df,
        clans_df=clans_df,
    )

    return dataset


def _select_categorical_columns(
    features: pd.DataFrame,
    config: Dict[str, Any],
) -> List[str]:
    """Return columns that need categorical encoding.

    Prefer explicit values from the preprocessing config.  If absent, fall
    back to all object columns in the feature matrix.
    """
    for key in ("categorical_cols", "categorical_features", "categorical_columns"):
        if key in config:
            raw_cols = config[key]
            if isinstance(raw_cols, list):
                # Keep only existing columns.
                return [c for c in raw_cols if c in features.columns]
            raise RuntimeError(
                f"'{key}' in preprocessing_config must be a list of column names."
            )

    # Fallback: all object columns that are not identifiers.
    return list(features.select_dtypes(include="object").columns)


def _get_train_test_split(
    test_size: float,
    random_state: int,
    preprocessing_config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return X_train, X_test, y_train, y_test.

    The split is performed after applying the same categorical encoding used
    during training, as recovered from MLflow.  Identifiers are removed before
    encoding.
    """
    dataset = _build_dataset()

    target = dataset[TARGET_COLUMN]
    features = dataset.drop(columns=[TARGET_COLUMN])

    # Identifiers must not be used as modelling features.
    for id_col in ("player_tag", "clan_tag"):
        if id_col in features.columns:
            features = features.drop(columns=[id_col])

    # Determine and apply categorical encoding.
    categorical_cols = _select_categorical_columns(features, preprocessing_config)
    if categorical_cols:
        features = pd.get_dummies(
            features,
            columns=categorical_cols,
            drop_first=True,
        )

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    return X_train, X_test, y_train, y_test


def _plot_class_distribution(y_test: pd.Series, output_path: Path) -> None:
    """Plot the distribution of the four target classes."""
    counts = y_test.value_counts().reindex(CLASS_ORDER, fill_value=0)

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar", color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    plt.title("Test set class distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of samples")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    output_path: Path,
) -> None:
    """Plot a confusion matrix for a single model."""
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion matrix - {model_name}")
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_ORDER))
    plt.xticks(tick_marks, CLASS_ORDER, rotation=45)
    plt.yticks(tick_marks, CLASS_ORDER)
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_global_metrics_comparison(
    metrics_df: pd.DataFrame, output_path: Path
) -> None:
    """Plot a grouped bar chart comparing global metrics for each model."""
    metrics_to_plot = [
        "accuracy",
        "f1_weighted",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
    ]

    plot_df = metrics_df[["model"] + metrics_to_plot].set_index("model")
    plot_df.plot(kind="bar", figsize=(10, 6), rot=0)
    plt.title("Global metrics comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_f1_by_class(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """Plot F1 score for each class and model."""
    class_cols = [f"{c}_f1" for c in CLASS_ORDER]
    plot_df = metrics_df[["model"] + class_cols].set_index("model")
    plot_df.plot(kind="bar", figsize=(10, 6), rot=0)
    plt.title("F1 score by class")
    plt.ylabel("F1")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_leader_metrics(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """Plot leader-specific precision, recall, and F1 for each model."""
    leader_cols = ["leader_precision", "leader_recall", "leader_f1"]
    plot_df = metrics_df[["model"] + leader_cols].set_index("model")
    plot_df.plot(kind="bar", figsize=(9, 6), rot=0)
    plt.title("Leader-specific metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _extract_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute all metrics required for the final comparison table."""
    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "f1_macro": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "precision_weighted": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    }

    for cls in CLASS_ORDER:
        cls_report = report.get(cls, {})
        metrics[f"{cls}_f1"] = cls_report.get("f1-score", 0.0)
        metrics[f"{cls}_precision"] = cls_report.get("precision", 0.0)
        metrics[f"{cls}_recall"] = cls_report.get("recall", 0.0)

    return metrics


def main() -> None:
    """Generate the final P1 analysis without creating new MLflow runs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    configure_tracking()

    # Determine the logistic regression run ID from its MLflow run name.
    logistic_run_id = _find_run_id_by_name(LOGISTIC_RUN_NAME)

    run_ids = [XGBOOST_RUN_ID, RF_RUN_ID, logistic_run_id]

    # Load the three final models from their existing MLflow runs.
    xgb_model = _load_model_from_run(XGBOOST_RUN_ID)
    rf_model = _load_model_from_run(RF_RUN_ID)
    logreg_model = _load_model_from_run(logistic_run_id)

    # Recover the exact split and preprocessing configuration from MLflow.
    test_size, random_state, preprocessing_config = _get_consistent_run_config(run_ids)

    # Reuse the project's existing pipeline and split.
    _, X_test, _, y_test = _get_train_test_split(
        test_size, random_state, preprocessing_config
    )

    models = {
        "xgboost": xgb_model,
        "random_forest": rf_model,
        "logistic_regression": logreg_model,
    }
    model_tags = {
        "xgboost": "Xgboost_tuned_v8",
        "random_forest": "rf_tuned_v6",
        "logistic_regression": "logistic_regression_tuned_v3",
    }

    all_metrics = []

    # Class distribution plot -------------------------------------------------
    _plot_class_distribution(
        y_test, RESULTS_DIR / "01_class_distribution.png"
    )

    # Confusion matrices and per-model metrics --------------------------------
    for key, model in models.items():
        y_pred = model.predict(X_test)

        # Convert possible array-like outputs to a clean 1D numpy array.
        if hasattr(y_pred, "to_numpy"):
            y_pred = y_pred.to_numpy()
        y_pred = np.asarray(y_pred).ravel()

        metrics = _extract_metrics(y_test, y_pred)
        metrics["model"] = model_tags[key]
        all_metrics.append(metrics)

        # Confusion matrix files with the requested naming convention.
        matrix_filename = {
            "xgboost": "02_confusion_matrix_xgboost.png",
            "random_forest": "02_confusion_matrix_random_forest.png",
            "logistic_regression": "02_confusion_matrix_logistic_regression.png",
        }[key]
        _plot_confusion_matrix(
            y_test,
            y_pred,
            model_name=model_tags[key],
            output_path=RESULTS_DIR / matrix_filename,
        )

    # Build final metrics table ------------------------------------------------
    metrics_df = pd.DataFrame(all_metrics)

    # Soft, reproducible ordering by macro F1.
    metrics_df = metrics_df.sort_values("f1_macro", ascending=False)

    # Create the final CSV with the exact columns requested by the project.
    csv_columns = [
        "model",
        "accuracy",
        "f1_weighted",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "admin_f1",
        "coLeader_f1",
        "leader_f1",
        "member_f1",
    ]
    final_csv_df = metrics_df[csv_columns]
    final_csv_df.to_csv(RESULTS_DIR / "final_model_comparison.csv", index=False)

    # Print table with 4 decimal places.
    with pd.option_context("display.precision", 4):
        print(final_csv_df.to_string(index=False))

    # Remaining plots ----------------------------------------------------------
    _plot_global_metrics_comparison(
        metrics_df, RESULTS_DIR / "03_global_metrics_comparison.png"
    )
    _plot_f1_by_class(metrics_df, RESULTS_DIR / "04_f1_by_class.png")
    _plot_leader_metrics(metrics_df, RESULTS_DIR / "05_leader_metrics.png")


if __name__ == "__main__":
    main()

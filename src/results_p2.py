import sys
import os
import json
import gc
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, unquote

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split


# =============================================================================
# REPOSITORY ROOT
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# =============================================================================
# PROJECT INFRASTRUCTURE
# =============================================================================

from mlflow_tracking.tracking_utils import configure_tracking


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_RUN_IDS = {
    "with_trophies": {
        "xgboost": "83396bc7df4946b3bfd388b32afe8cb5",
        "random_forest": "aa9291a0474b451b8d36b25e226aed5d",
        "ridge": "83532504b72142088051285c008ae4f0",
    },

    "without_trophies": {
        "xgboost": "bcb7d9873163449697ca75b2c1cbbba5",
        "random_forest": "8eef95c1839443e2840c07737d2e2f4c",
        "ridge": "575905e3512f4729acfb2b654e816de5",
    },
}


# =============================================================================
# GENERAL CONFIGURATION
# =============================================================================

RESULTS_DIR = ROOT_DIR / "src" / "results"

DATASET_WITH_TROPHIES = (
    ROOT_DIR
    / "data"
    / "datasets"
    / "clan_rank_regression_with_trophies.parquet"
)

DATASET_WITHOUT_TROPHIES = (
    ROOT_DIR
    / "data"
    / "datasets"
    / "clan_rank_regression_without_trophies.parquet"
)

TARGET_COLUMN = "clan_rank"

DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42


# =============================================================================
# MODEL DISPLAY NAMES
# =============================================================================

MODEL_DISPLAY_NAMES = {
    "xgboost": "xgboost",
    "random_forest": "random_forest",
    "ridge": "ridge",
}


# =============================================================================
# VALIDATE RUN CONFIGURATION
# =============================================================================

def _validate_run_configuration() -> None:

    valid_variants = {
        "with_trophies",
        "without_trophies",
    }

    valid_models = {
        "xgboost",
        "random_forest",
        "ridge",
    }

    for variant, models in MODEL_RUN_IDS.items():

        if variant not in valid_variants:
            raise ValueError(
                f"Invalid dataset variant in MODEL_RUN_IDS: {variant}"
            )

        if not isinstance(models, dict):
            raise ValueError(
                f"Configuration for '{variant}' must be a dictionary."
            )

        for model_name, run_id in models.items():

            if model_name not in valid_models:
                raise ValueError(
                    f"Invalid model '{model_name}' "
                    f"in variant '{variant}'."
                )

            if run_id is None:
                continue

            if not isinstance(run_id, str):
                raise ValueError(
                    f"Run ID for '{variant}/{model_name}' "
                    f"must be a string or None."
                )

            if not run_id.strip():
                raise ValueError(
                    f"Run ID for '{variant}/{model_name}' "
                    f"cannot be empty."
                )


# =============================================================================
# DATASET LOADING
# =============================================================================

def _load_dataset(path: Path) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    data = pd.read_parquet(path)

    if TARGET_COLUMN not in data.columns:
        raise RuntimeError(
            f"Target column '{TARGET_COLUMN}' "
            f"not found in dataset: {path}"
        )

    return data


# =============================================================================
# FEATURE PREPARATION
# =============================================================================
#
# NOTE: this stays as it was. It intentionally does NOT hardcode the
# training-time column drops (league_name, war_frequency, capital_league,
# etc.) beyond the obvious ID/target columns. Those extra columns are
# non-numeric (clan/player league and category labels from the Clash of
# Clans API), so `pd.to_numeric(errors="coerce")` turns them into all-NaN
# columns, and the all-NaN cleanup below removes them -- reproducing the
# same effective feature set the training scripts build explicitly,
# without duplicating that column list here (and risking it drifting out
# of sync with the six training scripts again).
#
# What this function does NOT do is decide which of the *numeric* columns
# (e.g. capital_contributions / clan_mean_capital_contributions) a given
# model was trained with -- the six P2 training scripts are not all
# consistent with each other about those two columns. That's handled
# per-model in `_prepare_model_input`, using each loaded model's own
# `feature_names_in_` / booster feature names as the source of truth,
# rather than duplicating a second guess here.
# =============================================================================

def _prepare_features(
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:

    y = data[TARGET_COLUMN].copy()

    X = data.drop(
        columns=[TARGET_COLUMN]
    ).copy()

    columns_to_remove = [
        "player_tag",
        "clan_tag",
        "name",
    ]

    existing_to_remove = [
        column
        for column in columns_to_remove
        if column in X.columns
    ]

    if existing_to_remove:
        X = X.drop(
            columns=existing_to_remove
        )

    X = X.apply(
        pd.to_numeric,
        errors="coerce",
    )

    bool_columns = X.select_dtypes(
        include=["bool"]
    ).columns.tolist()

    for column in bool_columns:
        X[column] = (
            X[column]
            .fillna(False)
            .astype(np.int8)
        )

    all_nan_columns = X.columns[
        X.isna().all()
    ].tolist()

    if all_nan_columns:
        X = X.drop(
            columns=all_nan_columns
        )

    for column in X.columns:

        if pd.api.types.is_float_dtype(
            X[column]
        ):
            X[column] = X[column].astype(
                np.float32
            )

        elif pd.api.types.is_integer_dtype(
            X[column]
        ):
            X[column] = X[column].astype(
                np.float32
            )

    return X, y


# =============================================================================
# TEST SPLIT
# =============================================================================

def _get_test_indices(
    n_samples: int,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> np.ndarray:

    indices = np.arange(
        n_samples,
        dtype=np.int32,
    )

    _, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=None,
    )

    return test_indices


# =============================================================================
# MLflow 3 MODEL LOADING
# =============================================================================
#
# Diagnosis (see chat for the full writeup and the GitHub issues this
# matches: mlflow/mlflow#16429 and #16501):
#
# Reproduced directly against a real local MLflow 3.15.2 install: calling
# mlflow.sklearn.log_model(model, artifact_path="model", ...) -- exactly
# what log_model_and_artifacts() in this project does -- no longer writes
# anything under the run's own artifact directory. runs:/<run_id>/model
# has nothing to find, so "please ensure the path is correct" is a
# correct error, not a fluke. The MLmodel/model.pkl files only exist
# under a separate LoggedModel-scoped location.
#
# Also reproduced: mlflow.search_logged_models(filter_string=f"source_run_id='{run_id}'")
# returns 0 results even when a matching row genuinely exists in the
# logged_models table with that exact source_run_id. That search/filter
# path is not reliable here -- which is exactly the "Found 0 Logged
# Model(s)" you were seeing.
#
# What *does* reliably work, confirmed against a live run: asking the RUN
# itself which LoggedModel(s) it produced, via run.outputs.model_outputs,
# then loading models:/<model_id> with the flavor-native loader. This is
# strategy 1 below.
#
# Ridge/XGBoost currently succeeding via the old runs:/<run_id>/model path
# most likely means those specific runs were logged under a different
# MLflow install than whatever was active for the Random Forest runs --
# requirements.txt only pins mlflow>=2.10.0, so a `pip install` run at a
# different time could easily have resolved to a different 2.x/3.x
# version. Strategies 2-4 below keep that path working for whichever
# runs it already works for, while fixing the ones it doesn't.
# =============================================================================

def _file_uri_to_local_path(uri: str) -> Path:
    """
    Convert a file:// artifact URI (as stored by MLflow -- percent-encoded,
    e.g. spaces as %20) into a real local Path. Handles the Windows
    leading-slash-before-drive-letter case that a bare urlparse().path
    would otherwise leave in the string (producing "/C:/Users/..." instead
    of "C:/Users/...").
    """
    parsed = urlparse(uri)
    path = unquote(parsed.path)
    if os.name == "nt" and len(path) > 2 and path[0] == "/" and path[2] == ":":
        path = path[1:]
    return Path(path)


def _find_mlmodel_dir(root: Path) -> Optional[Path]:
    """Search a local artifact directory for an MLmodel flavor file and
    return the directory containing it, or None if there isn't one."""
    if not root.exists():
        return None
    if (root / "MLmodel").exists():
        return root
    matches = list(root.rglob("MLmodel"))
    return matches[0].parent if matches else None


def _load_native_flavor(model_uri_or_path: str, model_key: str) -> Any:
    """
    Load with the flavor-native loader instead of mlflow.pyfunc.load_model,
    so we get the raw estimator/booster object back directly (a plain
    RandomForestRegressor / Ridge / XGBRegressor) rather than a PyFuncModel
    wrapper. This is both simpler downstream (feature_names_in_, .predict()
    work with no unwrapping) and matches exactly how log_model_and_artifacts()
    chose the flavor when logging.
    """
    if model_key == "xgboost":
        return mlflow.xgboost.load_model(model_uri_or_path)
    return mlflow.sklearn.load_model(model_uri_or_path)


def _load_model_from_run(
    run_id: str,
    model_key: str,
) -> Any:
    """
    Load a model logged with MLflow 3 via log_model_and_artifacts().

    Strategy, in order (each one printed clearly so a failure shows the
    real error and the exact path/location that was tried):

      1. run.outputs.model_outputs -> models:/<model_id>
         The correct MLflow-3-native lookup. Confirmed working in testing.
      2. Legacy runs:/<run_id>/model and runs:/<run_id>/modelo
         Kept for any run still logged the classic (pre-3.x-behavior) way.
      3. Direct local artifact inspection: resolve the run's and/or the
         LoggedModel's artifact_location file:// URI to a real path,
         list what's actually there, and load straight from that local
         directory -- bypassing runs:/ and models:/ URI resolution
         entirely.
      4. Raw pickle.load() on model.pkl next to whatever MLmodel file
         strategy 3 found, matching serialization_format="pickle" used
         at training time. Absolute last resort.

    Returns the raw flavor object, never a pyfunc wrapper.
    """

    client = MlflowClient()

    # -------------------------------------------------------------------
    # Strategy 1: run -> LoggedModel(s) -> models:/<model_id>
    # -------------------------------------------------------------------

    run = None

    try:
        run = client.get_run(run_id)
        model_outputs = (
            run.outputs.model_outputs
            if run.outputs is not None
            else []
        )

        print(
            f"    run.outputs.model_outputs: {len(model_outputs)} entry(ies)"
        )

        for output in model_outputs:

            model_id = output.model_id

            try:
                logged_model = client.get_logged_model(model_id)
                print(
                    f"      - model_id={model_id} "
                    f"name={logged_model.name} "
                    f"status={logged_model.status}"
                )
            except Exception:
                pass

            model_uri = f"models:/{model_id}"

            try:
                print(f"    Loading MLflow 3 Logged Model: {model_uri}")
                loaded_model = _load_native_flavor(model_uri, model_key)
                print("    \u2713 Loaded via models:/<model_id>")
                return loaded_model
            except Exception as exc:
                print(f"    {model_uri} failed: {exc}")

    except Exception as exc:
        print(f"    Could not inspect run outputs: {exc}")

    # -------------------------------------------------------------------
    # Strategy 2: legacy runs:/ paths
    # -------------------------------------------------------------------

    for fallback_uri in (
        f"runs:/{run_id}/model",
        f"runs:/{run_id}/modelo",
    ):
        try:
            print(f"    Trying fallback: {fallback_uri}")
            loaded_model = _load_native_flavor(fallback_uri, model_key)
            print(f"    \u2713 Loaded using fallback: {fallback_uri}")
            return loaded_model
        except Exception as exc:
            print(f"    Fallback failed: {exc}")

    # -------------------------------------------------------------------
    # Strategy 3: direct local artifact inspection
    # -------------------------------------------------------------------

    print("    Falling back to direct local artifact inspection...")

    candidate_locations: List[Tuple[str, str]] = []

    try:
        if run is None:
            run = client.get_run(run_id)

        candidate_locations.append(
            ("run artifact root", run.info.artifact_uri)
        )

        model_outputs = (
            run.outputs.model_outputs
            if run.outputs is not None
            else []
        )

        for output in model_outputs:
            try:
                logged_model = client.get_logged_model(output.model_id)
                candidate_locations.append(
                    (
                        f"logged model {output.model_id}",
                        logged_model.artifact_location,
                    )
                )
            except Exception:
                pass

    except Exception as exc:
        print(f"    Could not enumerate candidate locations: {exc}")

    for label, uri in candidate_locations:

        try:
            local_root = _file_uri_to_local_path(uri)
        except Exception as exc:
            print(f"    Could not resolve {label} ({uri}): {exc}")
            continue

        print(f"    Inspecting {label} on disk: {local_root}")

        if not local_root.exists():
            print("      (path does not exist)")
            continue

        found_files = sorted(
            str(p.relative_to(local_root))
            for p in local_root.rglob("*")
            if p.is_file()
        )

        print(
            f"      Files found: {found_files if found_files else '(empty)'}"
        )

        mlmodel_dir = _find_mlmodel_dir(local_root)

        if mlmodel_dir is None:
            continue

        print(f"    Found MLmodel at: {mlmodel_dir}")

        try:
            loaded_model = _load_native_flavor(str(mlmodel_dir), model_key)
            print(f"    \u2713 Loaded directly from local path: {mlmodel_dir}")
            return loaded_model
        except Exception as exc:
            print(f"    Native flavor load from local path failed: {exc}")

            # ---------------------------------------------------------------
            # Strategy 4: raw pickle, last resort
            # ---------------------------------------------------------------

            pkl_path = mlmodel_dir / "model.pkl"

            if pkl_path.exists():
                try:
                    print(f"    Trying raw pickle.load on: {pkl_path}")
                    with open(pkl_path, "rb") as f:
                        loaded_model = pickle.load(f)
                    print("    \u2713 Loaded via raw pickle.load")
                    return loaded_model
                except Exception as exc2:
                    print(f"    Raw pickle.load failed: {exc2}")

    raise RuntimeError(
        f"Could not load model from run {run_id}.\n"
        f"Tried: MLflow 3 Logged Model (models:/<model_id>), legacy "
        f"runs:/ paths, and direct local artifact inspection. See the "
        f"diagnostic output above for the real error and exact paths "
        f"tried at each step."
    )


# =============================================================================
# MODEL INPUT
# =============================================================================

def _prepare_model_input(
    model_key: str,
    model: Any,
    X_test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align X_test to exactly the columns the loaded model was trained on.

    Uses feature *names* (feature_names_in_ for sklearn-flavor models,
    including XGBRegressor's sklearn API; the booster's feature_names as a
    fallback for xgboost), not just a feature count. This matters here
    specifically because the six P2 training scripts are not all
    consistent with each other about two columns (capital_contributions,
    clan_mean_capital_contributions) -- with_trophies always drops them,
    without_trophies keeps them for random_forest/ridge but drops them for
    xgboost. Selecting by name handles all six combinations correctly
    without hardcoding any of them here.
    """

    X_model = X_test.copy()

    expected_features: Optional[List[str]] = None

    feature_names_in = getattr(model, "feature_names_in_", None)

    if feature_names_in is not None:
        expected_features = list(feature_names_in)

    if expected_features is None:
        try:
            booster = model.get_booster()
            booster_names = getattr(booster, "feature_names", None)
            if booster_names:
                expected_features = list(booster_names)
        except Exception:
            pass

    if expected_features is not None:

        missing = [
            column
            for column in expected_features
            if column not in X_model.columns
        ]

        if missing:
            raise RuntimeError(
                f"Feature mismatch for {model_key}.\n"
                f"Model expects columns not present in the prepared "
                f"dataset: {missing}"
            )

        extra = [
            column
            for column in X_model.columns
            if column not in expected_features
        ]

        if extra:
            print(
                f"    Dropping {len(extra)} column(s) not used by "
                f"this model: {extra}"
            )

        X_model = X_model[expected_features]

        print(
            f"    Feature count validated: {len(expected_features)}"
        )

    else:

        print(
            "    Model does not expose feature names."
        )

        print(
            f"    Using dataset feature matrix directly: "
            f"{X_model.shape}"
        )

    return X_model


# =============================================================================
# METRICS
# =============================================================================

def _extract_metrics(
    y_true: pd.Series,
    y_pred: Any,
) -> Dict[str, float]:

    y_true = np.asarray(
        y_true
    ).ravel()

    y_pred = np.asarray(
        y_pred
    ).ravel()

    if len(y_true) != len(y_pred):

        raise RuntimeError(
            "y_true and y_pred have different lengths: "
            f"{len(y_true)} vs {len(y_pred)}"
        )

    mse = mean_squared_error(
        y_true,
        y_pred,
    )

    return {

        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),

        "mse": float(
            mse
        ),

        "rmse": float(
            np.sqrt(mse)
        ),

        "r2": float(
            r2_score(
                y_true,
                y_pred,
            )
        ),
    }


# =============================================================================
# ACTUAL VS PREDICTED
# =============================================================================

def _plot_actual_vs_predicted(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    variant_name: str,
    output_path: Path,
) -> None:

    plt.figure(
        figsize=(8, 7)
    )

    plt.scatter(
        y_true,
        y_pred,
        alpha=0.25,
        s=8,
    )

    min_value = min(
        float(np.min(y_true)),
        float(np.min(y_pred)),
    )

    max_value = max(
        float(np.max(y_true)),
        float(np.max(y_pred)),
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
        linewidth=2,
    )

    plt.xlabel(
        "Actual clan rank"
    )

    plt.ylabel(
        "Predicted clan rank"
    )

    plt.title(
        f"Actual vs predicted - "
        f"{model_name} - {variant_name}"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


# =============================================================================
# RESIDUAL PLOT
# =============================================================================

def _plot_residuals(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    variant_name: str,
    output_path: Path,
) -> None:

    residuals = (
        np.asarray(y_true).ravel()
        - np.asarray(y_pred).ravel()
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.scatter(
        y_pred,
        residuals,
        alpha=0.25,
        s=8,
    )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=2,
    )

    plt.xlabel(
        "Predicted clan rank"
    )

    plt.ylabel(
        "Residual (actual - predicted)"
    )

    plt.title(
        f"Residuals - "
        f"{model_name} - {variant_name}"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


# =============================================================================
# RESIDUAL DISTRIBUTION
# =============================================================================

def _plot_residual_distribution(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    variant_name: str,
    output_path: Path,
) -> None:

    residuals = (
        np.asarray(y_true).ravel()
        - np.asarray(y_pred).ravel()
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.hist(
        residuals,
        bins=80,
        alpha=0.8,
    )

    plt.axvline(
        0,
        linestyle="--",
        linewidth=2,
    )

    plt.xlabel(
        "Residual"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        f"Residual distribution - "
        f"{model_name} - {variant_name}"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


# =============================================================================
# GLOBAL MODEL COMPARISON
# =============================================================================

def _plot_global_metrics_comparison(
    metrics_df: pd.DataFrame,
    output_path: Path,
) -> None:

    plot_df = metrics_df.copy()

    plot_df["model_variant"] = (
        plot_df["variant"]
        + " - "
        + plot_df["model"]
    )

    # -------------------------------------------------------------------------
    # MAE / RMSE
    # -------------------------------------------------------------------------

    error_df = (
        plot_df[
            [
                "model_variant",
                "mae",
                "rmse",
            ]
        ]
        .set_index("model_variant")
    )

    error_df.plot(
        kind="bar",
        figsize=(13, 7),
        rot=45,
    )

    plt.title(
        "MAE and RMSE comparison"
    )

    plt.ylabel(
        "Error"
    )

    plt.tight_layout()

    error_output = (
        output_path.parent
        / "06_mae_rmse_comparison.png"
    )

    plt.savefig(
        error_output,
        dpi=150,
    )

    plt.close()

    # -------------------------------------------------------------------------
    # R2
    # -------------------------------------------------------------------------

    r2_df = (
        plot_df[
            [
                "model_variant",
                "r2",
            ]
        ]
        .set_index("model_variant")
    )

    r2_df.plot(
        kind="bar",
        figsize=(13, 7),
        rot=45,
        legend=False,
    )

    plt.title(
        "R\u00b2 comparison"
    )

    plt.ylabel(
        "R\u00b2"
    )

    plt.tight_layout()

    r2_output = (
        output_path.parent
        / "07_r2_comparison.png"
    )

    plt.savefig(
        r2_output,
        dpi=150,
    )

    plt.close()


# =============================================================================
# EVALUATE ONE MODEL
# =============================================================================

def _evaluate_model(
    model_key: str,
    model: Any,
    model_name: str,
    variant_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Dict[str, float], np.ndarray]:

    print(
        f"  Evaluating {model_name}..."
    )

    X_test_model = _prepare_model_input(
        model_key=model_key,
        model=model,
        X_test=X_test,
    )

    print(
        "    Running predictions..."
    )

    y_pred = model.predict(
        X_test_model
    )

    if hasattr(
        y_pred,
        "to_numpy",
    ):
        y_pred = y_pred.to_numpy()

    y_pred = np.asarray(
        y_pred
    ).ravel()

    metrics = _extract_metrics(
        y_test,
        y_pred,
    )

    metrics["model"] = model_name
    metrics["variant"] = variant_name

    print(
        f"    MAE  = {metrics['mae']:.4f}"
    )

    print(
        f"    MSE  = {metrics['mse']:.4f}"
    )

    print(
        f"    RMSE = {metrics['rmse']:.4f}"
    )

    print(
        f"    R\u00b2   = {metrics['r2']:.4f}"
    )

    return (
        metrics,
        y_pred,
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print(
        "\n"
        + "=" * 80
    )

    print(
        "P2 FINAL RESULTS - CLAN RANK REGRESSION"
    )

    print(
        "=" * 80
        + "\n"
    )

    # -------------------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------------------

    print(
        "[1/6] Validating MLflow Run ID configuration..."
    )

    _validate_run_configuration()

    print(
        "\u2713 Run configuration valid\n"
    )

    # -------------------------------------------------------------------------
    # MLflow
    # -------------------------------------------------------------------------

    print(
        "[2/6] Configuring MLflow..."
    )

    configure_tracking()

    print(
        "\u2713 MLflow configured\n"
    )

    # -------------------------------------------------------------------------
    # Show runs
    # -------------------------------------------------------------------------

    print(
        "Configured runs:"
    )

    for variant, models in MODEL_RUN_IDS.items():

        print(
            f"\n{variant}:"
        )

        configured_count = 0

        for model_name, run_id in models.items():

            if run_id is None:

                print(
                    f"  {model_name}: SKIPPED"
                )

            else:

                print(
                    f"  {model_name}: {run_id}"
                )

                configured_count += 1

        print(
            f"  Configured models: {configured_count}"
        )

    print()

    # -------------------------------------------------------------------------
    # Datasets
    # -------------------------------------------------------------------------

    print(
        "[3/6] Loading datasets..."
    )

    datasets = {
        "with_trophies": DATASET_WITH_TROPHIES,
        "without_trophies": DATASET_WITHOUT_TROPHIES,
    }

    all_metrics = []

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    print(
        "[4/6] Evaluating models...\n"
    )

    for variant_name, dataset_path in datasets.items():

        print(
            "\n"
            + "-" * 80
        )

        print(
            f"DATASET: {variant_name}"
        )

        print(
            "-" * 80
        )

        # ---------------------------------------------------------------------
        # Load dataset
        # ---------------------------------------------------------------------

        data = _load_dataset(
            dataset_path
        )

        print(
            f"Dataset shape: {data.shape}"
        )

        # ---------------------------------------------------------------------
        # Features
        # ---------------------------------------------------------------------

        X, y = _prepare_features(
            data
        )

        print(
            f"Feature matrix shape: {X.shape}"
        )

        print(
            f"Target size: {len(y)}"
        )

        print(
            f"Number of available features: {X.shape[1]}"
        )

        # ---------------------------------------------------------------------
        # Recreate exact test split
        # ---------------------------------------------------------------------

        test_indices = _get_test_indices(
            n_samples=len(X),
            test_size=DEFAULT_TEST_SIZE,
            random_state=DEFAULT_RANDOM_STATE,
        )

        X_test = X.iloc[
            test_indices
        ].copy()

        y_test = y.iloc[
            test_indices
        ].copy()

        print(
            f"Test shape: {X_test.shape}"
        )

        print(
            f"Test target size: {len(y_test)}\n"
        )

        # ---------------------------------------------------------------------
        # Free full data
        # ---------------------------------------------------------------------

        del X
        del y
        del data
        del test_indices

        gc.collect()

        # ---------------------------------------------------------------------
        # Output directory
        # ---------------------------------------------------------------------

        variant_results_dir = (
            RESULTS_DIR
            / variant_name
        )

        variant_results_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ---------------------------------------------------------------------
        # Models
        # ---------------------------------------------------------------------

        variant_runs = MODEL_RUN_IDS.get(
            variant_name,
            {},
        )

        for model_key, run_id in variant_runs.items():

            if run_id is None:

                print(
                    f"  Skipping {model_key}: "
                    f"no Run ID configured."
                )

                continue

            model = None
            y_pred = None

            # -----------------------------------------------------------------
            # Load
            # -----------------------------------------------------------------

            try:

                print(
                    f"\n  Loading {model_key}..."
                )

                model = _load_model_from_run(
                    run_id=run_id,
                    model_key=model_key,
                )

                print(
                    f"  \u2713 {model_key} loaded"
                )

            except Exception as exc:

                print(
                    f"\nWARNING: Could not load "
                    f"{model_key} for {variant_name}."
                )

                print(
                    f"  Run ID: {run_id}"
                )

                print(
                    f"  Error: {exc}"
                )

                continue

            model_display_name = MODEL_DISPLAY_NAMES.get(
                model_key,
                model_key,
            )

            # -----------------------------------------------------------------
            # Evaluate
            # -----------------------------------------------------------------

            try:

                metrics, y_pred = _evaluate_model(
                    model_key=model_key,
                    model=model,
                    model_name=model_display_name,
                    variant_name=variant_name,
                    X_test=X_test,
                    y_test=y_test,
                )

            except Exception as exc:

                print(
                    f"\nWARNING: Could not evaluate "
                    f"{model_key} for {variant_name}."
                )

                print(
                    f"  Run ID: {run_id}"
                )

                print(
                    f"  Error: {exc}"
                )

                del model
                model = None

                gc.collect()

                continue

            metrics["run_id"] = run_id

            all_metrics.append(
                metrics
            )

            # -----------------------------------------------------------------
            # Actual vs predicted
            # -----------------------------------------------------------------

            _plot_actual_vs_predicted(
                y_true=y_test,
                y_pred=y_pred,
                model_name=model_display_name,
                variant_name=variant_name,
                output_path=(
                    variant_results_dir
                    / f"01_actual_vs_predicted_{model_key}.png"
                ),
            )

            # -----------------------------------------------------------------
            # Residuals
            # -----------------------------------------------------------------

            _plot_residuals(
                y_true=y_test,
                y_pred=y_pred,
                model_name=model_display_name,
                variant_name=variant_name,
                output_path=(
                    variant_results_dir
                    / f"02_residuals_{model_key}.png"
                ),
            )

            # -----------------------------------------------------------------
            # Residual distribution
            # -----------------------------------------------------------------

            _plot_residual_distribution(
                y_true=y_test,
                y_pred=y_pred,
                model_name=model_display_name,
                variant_name=variant_name,
                output_path=(
                    variant_results_dir
                    / f"03_residual_distribution_{model_key}.png"
                ),
            )

            print(
                f"  \u2713 {model_key} completed"
            )

            # -----------------------------------------------------------------
            # Memory cleanup
            # -----------------------------------------------------------------

            del y_pred
            del model

            model = None
            y_pred = None

            gc.collect()

        # ---------------------------------------------------------------------
        # Free test data
        # ---------------------------------------------------------------------

        del X_test
        del y_test

        gc.collect()

    # =========================================================================
    # FINAL RESULTS
    # =========================================================================

    print(
        "[5/6] Building final comparison tables..."
    )

    if not all_metrics:

        raise RuntimeError(
            "No models were successfully evaluated."
        )

    metrics_df = pd.DataFrame(
        all_metrics
    )

    metrics_df = metrics_df.sort_values(
        "rmse",
        ascending=True,
    )

    # -------------------------------------------------------------------------
    # Main CSV
    # -------------------------------------------------------------------------

    csv_columns = [
        "variant",
        "model",
        "mae",
        "mse",
        "rmse",
        "r2",
        "run_id",
    ]

    final_csv_df = metrics_df[
        csv_columns
    ].copy()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_csv_df.to_csv(
        RESULTS_DIR
        / "final_model_comparison.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Console comparison
    # -------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL MODEL COMPARISON"
    )

    print(
        "=" * 80
        + "\n"
    )

    print(
        final_csv_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()

    # -------------------------------------------------------------------------
    # Best model per variant
    # -------------------------------------------------------------------------

    best_models = (
        metrics_df
        .sort_values(
            "rmse",
            ascending=True,
        )
        .groupby(
            "variant",
            as_index=False,
        )
        .first()
    )

    best_models.to_csv(
        RESULTS_DIR
        / "best_model_by_variant.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Global comparison plots
    # -------------------------------------------------------------------------

    _plot_global_metrics_comparison(
        metrics_df,
        RESULTS_DIR
        / "05_global_metrics_comparison.png",
    )

    # -------------------------------------------------------------------------
    # Summary JSON
    # -------------------------------------------------------------------------

    summary = {

        "datasets": {

            "with_trophies": str(
                DATASET_WITH_TROPHIES
            ),

            "without_trophies": str(
                DATASET_WITHOUT_TROPHIES
            ),
        },

        "test_size": DEFAULT_TEST_SIZE,

        "random_state": DEFAULT_RANDOM_STATE,

        "models": MODEL_RUN_IDS,

        "results": metrics_df[
            [
                "variant",
                "model",
                "mae",
                "mse",
                "rmse",
                "r2",
            ]
        ].to_dict(
            orient="records"
        ),
    }

    with open(
        RESULTS_DIR
        / "results_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # -------------------------------------------------------------------------
    # Final output
    # -------------------------------------------------------------------------

    print(
        "[6/6] Results generated successfully!"
    )

    print(
        "\nOutput directory:"
    )

    print(
        RESULTS_DIR
    )

    print(
        "\nGenerated files:"
    )

    for file in sorted(
        RESULTS_DIR.rglob("*")
    ):

        if file.is_file():

            print(
                f"  - {file.relative_to(RESULTS_DIR)}"
            )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "P2 RESULTS COMPLETED"
    )

    print(
        "=" * 80
        + "\n"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
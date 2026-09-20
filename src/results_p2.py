import sys
import os
import json
import gc
import pickle
import traceback
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
        "random_forest": "7c4018cff56846bd8cd591d6624aaed9",
        "ridge": "407b15aa60a74dd0bdf879a55f012d01",
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

# How many features the two GLOBAL feature importance charts show
# (09_feature_importance_global_comparison.png and
# 10_feature_importance_global_heatmap.png).
GLOBAL_IMPORTANCE_TOP_N = 15


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

    # Detected BEFORE numeric coercion: a bool column with no missing
    # values keeps dtype "bool" straight through pd.to_numeric below, but
    # a bool column that DOES have missing values is already dtype
    # "object" by this point (pandas can't hold NaN in a plain bool
    # column) -- coercion still correctly turns its True/False into
    # 1.0/0.0, but leaves the actual missing entries as NaN rather than
    # False, and a dtype=="bool" check done AFTER coercion never finds
    # that column at all, so those NaNs would otherwise silently ride
    # through to .predict(). Checking here, before coercion, catches
    # both cases the same way.
    bool_like_columns = []

    for column in X.columns:

        if X[column].dtype == bool:
            bool_like_columns.append(column)
            continue

        if X[column].dtype == object:
            non_null = X[column].dropna()
            if len(non_null) > 0 and non_null.isin([True, False]).all():
                bool_like_columns.append(column)

    X = X.apply(
        pd.to_numeric,
        errors="coerce",
    )

    for column in bool_like_columns:
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


def _describe_local_artifacts(uri: str, label: str) -> None:
    """
    Resolve a file:// artifact URI to a local path and print what's there,
    WITH FILE SIZES, before any load is attempted. This runs unconditionally
    (not just on failure) specifically so a large model.pkl is visible up
    front -- a bare `except Exception as exc: print(exc)` can print an
    EMPTY string for some exceptions (MemoryError raised with no message
    does exactly this), which looks like "it just silently failed" when
    what actually happened is the file is too big to fit in memory. Seeing
    the size here removes the guessing.
    """
    try:
        local_root = _file_uri_to_local_path(uri)
    except Exception as exc:
        print(f"    Could not resolve {label} ({uri}): {type(exc).__name__}: {exc}")
        return

    if not local_root.exists():
        print(f"    {label}: {local_root} (does not exist)")
        return

    files = sorted(p for p in local_root.rglob("*") if p.is_file())

    if not files:
        print(f"    {label}: {local_root} (empty)")
        return

    print(f"    {label}: {local_root}")

    for file_path in files:
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(
            f"      - {file_path.relative_to(local_root)}: {size_mb:,.1f} MB"
        )


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

    Every exception is printed as "ExceptionType: message" rather than
    just "message" -- some exceptions (a bare MemoryError in particular)
    stringify to an EMPTY string, which otherwise looks exactly like a
    silent, unexplained failure instead of what it actually is. File
    sizes for whatever's found on disk are also printed up front, before
    any load is attempted, so a too-large-to-load model.pkl is visible
    immediately rather than inferred after the fact.

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
            logged_model = None

            try:
                logged_model = client.get_logged_model(model_id)
                print(
                    f"      - model_id={model_id} "
                    f"name={logged_model.name} "
                    f"status={logged_model.status}"
                )
            except Exception:
                pass

            if logged_model is not None:
                _describe_local_artifacts(
                    logged_model.artifact_location,
                    f"logged model {model_id} on disk",
                )

            model_uri = f"models:/{model_id}"

            try:
                print(f"    Loading MLflow 3 Logged Model: {model_uri}")
                loaded_model = _load_native_flavor(model_uri, model_key)
                print("    \u2713 Loaded via models:/<model_id>")
                return loaded_model
            except Exception as exc:
                print(
                    f"    {model_uri} failed: "
                    f"{type(exc).__name__}: {exc}"
                )

    except Exception as exc:
        print(
            f"    Could not inspect run outputs: "
            f"{type(exc).__name__}: {exc}"
        )

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
            print(
                f"    Fallback failed: {type(exc).__name__}: {exc}"
            )

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
        print(
            f"    Could not enumerate candidate locations: "
            f"{type(exc).__name__}: {exc}"
        )

    for label, uri in candidate_locations:

        try:
            local_root = _file_uri_to_local_path(uri)
        except Exception as exc:
            print(
                f"    Could not resolve {label} ({uri}): "
                f"{type(exc).__name__}: {exc}"
            )
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

        pkl_path = mlmodel_dir / "model.pkl"

        if pkl_path.exists():
            size_mb = pkl_path.stat().st_size / (1024 * 1024)
            print(f"    model.pkl size: {size_mb:,.1f} MB")

            if size_mb > 1500:
                print(
                    "    NOTE: this is a large file. If the next two "
                    "attempts fail with an EMPTY error message below, "
                    "that blank is very likely a bare MemoryError -- "
                    "not the path being wrong -- i.e. this process ran "
                    "out of available RAM while deserializing it, not "
                    "a location problem."
                )

        try:
            loaded_model = _load_native_flavor(str(mlmodel_dir), model_key)
            print(f"    \u2713 Loaded directly from local path: {mlmodel_dir}")
            return loaded_model
        except Exception as exc:
            print(
                f"    Native flavor load from local path failed: "
                f"{type(exc).__name__}: {exc}"
            )
            if not str(exc):
                print(
                    "      (empty message above; see NOTE on file size, "
                    "or the exception type itself, for what this means)"
                )

            # ---------------------------------------------------------------
            # Strategy 4: raw pickle, last resort
            # ---------------------------------------------------------------

            if pkl_path.exists():
                try:
                    print(f"    Trying raw pickle.load on: {pkl_path}")
                    with open(pkl_path, "rb") as f:
                        loaded_model = pickle.load(f)
                    print("    \u2713 Loaded via raw pickle.load")
                    return loaded_model
                except Exception as exc2:
                    print(
                        f"    Raw pickle.load failed: "
                        f"{type(exc2).__name__}: {exc2}"
                    )
                    if not str(exc2):
                        print(
                            "      (empty message again -- same "
                            "underlying cause as the attempt above)"
                        )
                    print(
                        "    Full traceback for this last attempt:"
                    )
                    print(
                        "    "
                        + traceback.format_exc().replace("\n", "\n    ")
                    )

    raise RuntimeError(
        f"Could not load model from run {run_id}.\n"
        f"Tried: MLflow 3 Logged Model (models:/<model_id>), legacy "
        f"runs:/ paths, and direct local artifact inspection. See the "
        f"diagnostic output above for the real error, the exception "
        f"TYPE (not just its message -- some exceptions like a bare "
        f"MemoryError stringify to nothing), file sizes, and exact "
        f"paths tried at each step."
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
# FEATURE IMPORTANCE (XGBoost / Random Forest)
# =============================================================================

def _extract_feature_importance(
    model: Any,
    feature_names: List[str],
) -> Optional[pd.Series]:
    """
    Returns feature_importances_ indexed by name, or None if the model
    doesn't expose it (Ridge doesn't -- handled separately below).

    Prefers the model's OWN feature_names_in_ over the shared
    feature_names list built from X_test. Confirmed the hard way in P4:
    if this specific model was trained on a different column subset than
    the other models (exactly the kind of per-model inconsistency
    already found between the six P2 training scripts --
    capital_contributions present for some, absent for others),
    feature_importances_ comes back shorter than the shared list, and
    indexing it with the wrong names either raises ValueError or, worse,
    would silently mislabel importances if the lengths matched by
    coincidence.
    """

    importances = getattr(
        model,
        "feature_importances_",
        None,
    )

    if importances is None:
        return None

    own_feature_names = getattr(
        model,
        "feature_names_in_",
        None,
    )

    if own_feature_names is not None:

        own_feature_names = list(own_feature_names)

        if len(own_feature_names) == len(importances):

            if set(own_feature_names) != set(feature_names):
                print(
                    f"    NOTE: this model's features differ from the "
                    f"shared feature set ({len(own_feature_names)} vs "
                    f"{len(feature_names)} columns) -- using this "
                    f"model's own feature names for its importance "
                    f"values."
                )

            return pd.Series(
                importances,
                index=own_feature_names,
            )

    if len(importances) == len(feature_names):
        return pd.Series(
            importances,
            index=feature_names,
        )

    print(
        f"    Skipping feature importance: feature_importances_ has "
        f"{len(importances)} entries, which matches neither this "
        f"model's own feature_names_in_ nor the {len(feature_names)} "
        f"shared feature columns."
    )

    return None


def _extract_ridge_importance(
    model: Any,
    X_reference: pd.DataFrame,
    feature_names: List[str],
) -> Optional[pd.Series]:
    """
    Ridge has no feature_importances_ -- its coefficients are the
    equivalent, but a RAW coefficient conflates effect size with that
    feature's natural scale (a coefficient on "trophies", which ranges in
    the thousands, is not comparable to one on a 0-1 ratio just because
    both are "a coefficient"). Multiplying each coefficient by that
    feature's own standard deviation (computed from X_reference, i.e.
    X_test -- an unbiased sample of the same distribution the model sees
    at evaluation time) rescales every feature onto the same basis: "how
    much does a real one-standard-deviation change in this feature move
    the prediction". That is what actually makes it comparable to
    RF/XGBoost's impurity-based importances at all, whether or not Ridge
    itself was fit on pre-standardized features.

    Returns a DataFrame with both the signed standardized coefficient
    (which direction the feature pushes the prediction) and its absolute
    value (the magnitude used for the cross-model importance comparison,
    to match RF/XGBoost's non-negative importances).
    """

    coefficients = getattr(
        model,
        "coef_",
        None,
    )

    if coefficients is None:
        return None

    coefficients = np.asarray(coefficients).ravel()

    own_feature_names = getattr(model, "feature_names_in_", None)

    if own_feature_names is not None and len(list(own_feature_names)) == len(coefficients):
        names = list(own_feature_names)
    elif len(coefficients) == len(feature_names):
        names = feature_names
    else:
        print(
            f"    Skipping Ridge coefficients: length {len(coefficients)} "
            f"matches neither feature_names_in_ nor the shared feature "
            f"columns."
        )
        return None

    missing = [n for n in names if n not in X_reference.columns]
    if missing:
        print(
            f"    Skipping Ridge coefficients: {missing} not present in "
            f"the evaluation data, can't compute feature std for scaling."
        )
        return None

    feature_stds = X_reference[names].std().replace(0, np.nan)

    standardized = coefficients * feature_stds.values

    return pd.DataFrame({
        "coefficient_standardized": standardized,
        "coefficient_standardized_abs": np.abs(standardized),
    }, index=names)


def _normalize_importance_for_comparison(
    series: pd.Series,
) -> pd.Series:
    """
    Scales an importance series to sum to 1 -- matching the convention
    RF/XGBoost's own feature_importances_ already follow natively. Ridge's
    standardized-coefficient magnitudes are on a completely different
    numeric scale (they carry the target's own units), so without this
    they would either dwarf or vanish next to RF/XGBoost's 0-1 values on
    the same chart. This is a *relative-ranking* view for the plot only
    -- "how does this feature rank against this model's own most
    important features" -- not a claim that the underlying units are the
    same, which they aren't and shouldn't be presented as.
    """

    total = series.sum()

    if total == 0 or pd.isna(total):
        return series * 0.0

    return series / total


def _plot_feature_importance_comparison(
    importance_by_model: Dict[str, pd.Series],
    output_path: Path,
    top_n: int = 15,
) -> None:
    """
    Grouped bar chart comparing normalized feature importance across
    however many of {xgboost, random_forest, ridge} loaded successfully
    for this variant. Each model's column is independently normalized to
    sum to 1 (see _normalize_importance_for_comparison) before plotting
    -- RF/XGBoost's impurity-based fraction and Ridge's standardized
    |coefficient| are different units by construction; this puts them on
    a common 0-1 relative-ranking scale for visual comparison only. The
    saved CSV (built alongside this plot) keeps each model's native,
    unnormalized values so nothing is lost to that rescaling.
    """

    if not importance_by_model:
        return

    normalized = {
        model_key: _normalize_importance_for_comparison(series)
        for model_key, series in importance_by_model.items()
    }

    combined = pd.DataFrame(normalized).fillna(0.0)

    ranking = combined.mean(axis=1).sort_values(ascending=False)
    top_features = ranking.head(top_n).index

    plot_df = combined.loc[top_features]

    plot_df.plot(
        kind="barh",
        figsize=(10, 8),
    )

    plt.gca().invert_yaxis()

    plt.title(
        f"Feature importance comparison (top {len(top_features)})\n"
        f"(each model's own values normalized to sum to 1 -- relative "
        f"ranking, not the same units)"
    )

    plt.xlabel("Normalized importance")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


# =============================================================================
# GLOBAL FEATURE IMPORTANCE COMPARISON (both variants, all models)
# =============================================================================
#
# The per-variant comparison above (08_feature_importance_comparison.png)
# only puts the models of ONE variant side by side. The functions below
# put every model that loaded successfully -- with_trophies AND
# without_trophies -- into a single table and two charts, so the effect
# of removing the trophy features can be read model by model (the
# with/without pair of the same model sits together) next to the
# differences between the models themselves.
#
# Same convention as the per-variant comparison: every (variant, model)
# column is normalized to sum to 1 over that model's OWN feature set, so
# columns read as "share of this model's total importance", not as the
# same unit (Ridge's standardized coefficients and the trees' impurity
# importances never are). One consequence to keep in mind when reading
# with vs without: removing the trophy features hands their share to the
# remaining features, so a feature gaining share in without_trophies is
# expected -- what the comparison shows is WHICH features absorb it.
#
# A feature that is not in a model's feature set at all (e.g. the trophy
# columns in a without_trophies model) stays NaN in the table -- NOT 0 --
# so "this model never saw the feature" remains distinguishable from
# "this model saw it and it carries no importance". The bar chart draws
# it as an empty bar; the heatmap draws it as a grey "n/a" cell.
# =============================================================================

def _build_global_importance_table(
    all_feature_importance: Dict[str, Dict[str, pd.Series]],
) -> pd.DataFrame:
    """
    Builds ONE table with the normalized importance of every model that
    loaded successfully, in both dataset variants.

    Index: feature name (union of every model's own feature set).
    Columns: MultiIndex (variant, model_key), grouped by model first and
    variant second -- so with_trophies / without_trophies of the SAME
    model sit next to each other, which is what makes the effect of
    removing the trophy features readable at a glance.

    NaN means "this feature is not in that model's feature set". It is
    deliberately NOT filled with 0 here.
    """

    column_keys: List[Tuple[str, str]] = []
    column_series: List[pd.Series] = []

    for model_key in MODEL_DISPLAY_NAMES:

        for variant_name in MODEL_RUN_IDS:

            series = all_feature_importance.get(
                variant_name,
                {},
            ).get(model_key)

            if series is None:
                continue

            # A NaN INSIDE a model's own series means "in the feature set
            # but no measurable importance" (e.g. a zero-variance feature
            # has an undefined standardized Ridge coefficient), so it
            # counts as 0. Features the model never had at all only turn
            # into NaN when the columns are aligned below.
            series = series.astype(float).fillna(0.0)

            column_keys.append(
                (variant_name, model_key)
            )

            column_series.append(
                _normalize_importance_for_comparison(series)
            )

    if not column_series:
        return pd.DataFrame()

    table = pd.concat(
        column_series,
        axis=1,
    )

    table.columns = pd.MultiIndex.from_tuples(
        column_keys,
        names=["variant", "model"],
    )

    table.index.name = "feature"

    return table


def _rank_features_by_mean_importance(
    table: pd.DataFrame,
) -> pd.Series:
    """
    Mean normalized importance across every (variant, model) column,
    treating "feature not in this model" as 0, highest first. Used for
    the CSV order and to pick the top features of both charts, so the
    CSV and the two charts always agree on which features come first.
    """

    return (
        table
        .fillna(0.0)
        .mean(axis=1)
        .sort_values(ascending=False)
    )


def _plot_global_importance_bars(
    table: pd.DataFrame,
    output_path: Path,
    top_n: int = 15,
) -> None:
    """
    Grouped horizontal bars: one group per feature (top_n by mean
    importance), one bar per (variant, model) column. Colour = model and
    hatched = without_trophies, so the two bars of the same colour inside
    each group are exactly "this model with vs without trophies". A
    feature that is not in a model's feature set is drawn as 0.
    """

    ranking = _rank_features_by_mean_importance(table)

    top_features = list(
        ranking.head(top_n).index
    )

    plot_df = table.loc[top_features].fillna(0.0)

    n_series = plot_df.shape[1]

    bar_height = 0.8 / n_series

    positions = np.arange(
        len(top_features)
    )

    model_keys = list(MODEL_DISPLAY_NAMES.keys())
    variant_names = list(MODEL_RUN_IDS.keys())

    # No hatch = first variant (with_trophies); hatched = the others.
    hatch_patterns = ["", "//", "..", "xx"]

    fig, ax = plt.subplots(
        figsize=(
            13,
            max(8.0, 0.9 * len(top_features)),
        )
    )

    for column_index, (variant_name, model_key) in enumerate(
        plot_df.columns
    ):

        offset = (
            column_index
            - (n_series - 1) / 2.0
        ) * bar_height

        ax.barh(
            positions + offset,
            plot_df[(variant_name, model_key)].to_numpy(),
            height=bar_height,
            color=f"C{model_keys.index(model_key) % 10}",
            hatch=hatch_patterns[
                variant_names.index(variant_name)
                % len(hatch_patterns)
            ],
            edgecolor="black",
            linewidth=0.5,
            label=(
                f"{variant_name} - "
                f"{MODEL_DISPLAY_NAMES.get(model_key, model_key)}"
            ),
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(top_features)
    ax.invert_yaxis()

    ax.set_xlabel("Normalized importance")

    ax.set_title(
        f"Global feature importance comparison "
        f"(top {len(top_features)})\n"
        f"(each model normalized to sum to 1 over its own features "
        f"-- relative ranking, not the same units)"
    )

    ax.legend(loc="lower right")

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
    )

    plt.close(fig)


def _plot_global_importance_heatmap(
    table: pd.DataFrame,
    output_path: Path,
    top_n: int = 15,
) -> None:
    """
    Heatmap of the same top_n features x every (variant, model) column,
    with the exact normalized value written in each cell. A grey "n/a"
    cell means the feature is not part of that model's feature set (the
    trophy columns in without_trophies), as opposed to a pale cell,
    which means it IS in the model and simply carries little importance.
    """

    ranking = _rank_features_by_mean_importance(table)

    top_features = list(
        ranking.head(top_n).index
    )

    heat = table.loc[top_features]

    values = heat.to_numpy(dtype=float)

    finite_values = values[np.isfinite(values)]

    vmax = (
        float(finite_values.max())
        if finite_values.size > 0
        else 1.0
    )

    if vmax <= 0.0:
        vmax = 1.0

    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(color="#d9d9d9")

    n_rows, n_cols = values.shape

    fig, ax = plt.subplots(
        figsize=(
            max(9.0, 1.6 * n_cols + 5.0),
            max(6.0, 0.5 * n_rows + 2.5),
        )
    )

    image = ax.imshow(
        np.ma.masked_invalid(values),
        aspect="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
    )

    ax.set_xticks(np.arange(n_cols))

    ax.set_xticklabels(
        [
            f"{MODEL_DISPLAY_NAMES.get(model_key, model_key)}\n{variant_name}"
            for variant_name, model_key in heat.columns
        ]
    )

    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(top_features)

    for row in range(n_rows):

        for col in range(n_cols):

            value = values[row, col]

            if np.isnan(value):

                ax.text(
                    col,
                    row,
                    "n/a",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#666666",
                )

            else:

                ax.text(
                    col,
                    row,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=(
                        "white"
                        if value > 0.6 * vmax
                        else "black"
                    ),
                )

    # A dark line wherever the model changes, so each model's
    # with_trophies / without_trophies pair reads as one block.
    column_models = [
        model_key
        for _, model_key in heat.columns
    ]

    for col in range(1, n_cols):

        if column_models[col] != column_models[col - 1]:

            ax.axvline(
                col - 0.5,
                color="black",
                linewidth=1.5,
            )

    fig.colorbar(
        image,
        ax=ax,
        label="Normalized importance",
    )

    ax.set_title(
        f"Global feature importance heatmap "
        f"(top {len(top_features)})\n"
        f"(each model normalized to sum to 1 over its own features; "
        f"grey n/a = feature not used by that model)",
        fontsize=11,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
    )

    plt.close(fig)


def _save_global_feature_importance_comparison(
    all_feature_importance: Dict[str, Dict[str, pd.Series]],
    results_dir: Path,
    top_n: int = 15,
) -> None:
    """
    Builds the global table from the per-variant importance collected in
    main() and writes, all under results_dir (next to the other global
    comparison files):

      - feature_importance_global_comparison.csv
      - 09_feature_importance_global_comparison.png  (grouped bars)
      - 10_feature_importance_global_heatmap.png     (heatmap)

    The CSV keeps "feature not in that model's feature set" as an empty
    cell and adds mean_importance (the ranking behind the CSV order and
    the top-N features of both charts). Each model's NATIVE, unnormalized
    values stay in the per-variant feature_importance_comparison.csv.
    """

    table = _build_global_importance_table(
        all_feature_importance
    )

    if table.empty:

        print(
            "  No feature importance available for any model; "
            "skipping the global comparison."
        )

        return

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_df = table.copy()

    csv_df.columns = [
        f"{variant_name}__{model_key}"
        for variant_name, model_key in table.columns
    ]

    csv_df["mean_importance"] = _rank_features_by_mean_importance(
        table
    )

    csv_df = csv_df.sort_values(
        "mean_importance",
        ascending=False,
    )

    csv_df.to_csv(
        results_dir
        / "feature_importance_global_comparison.csv"
    )

    _plot_global_importance_bars(
        table,
        results_dir
        / "09_feature_importance_global_comparison.png",
        top_n=top_n,
    )

    _plot_global_importance_heatmap(
        table,
        results_dir
        / "10_feature_importance_global_heatmap.png",
        top_n=top_n,
    )

    included = ", ".join(
        f"{variant_name}/{model_key}"
        for variant_name, model_key in table.columns
    )

    print(
        f"  Global feature importance comparison saved: "
        f"{table.shape[1]} model(s) [{included}], "
        f"{table.shape[0]} feature(s) in total"
    )


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

    # Per-variant feature importance (variant -> model_key -> Series),
    # kept so the global with/without comparison can be built once every
    # variant has been evaluated.
    all_feature_importance: Dict[str, Dict[str, pd.Series]] = {}

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

        feature_importance_records: Dict[str, pd.Series] = {}
        ridge_signed_coefficients: Optional[pd.Series] = None

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

            # -----------------------------------------------------------------
            # Feature importance / Ridge coefficients (lightweight -- only
            # the small resulting Series is kept, not the model, so this
            # doesn't affect the "one model in memory at a time" rule).
            # -----------------------------------------------------------------

            feature_names = list(X_test.columns)

            if model_key == "ridge":

                ridge_result = _extract_ridge_importance(
                    model=model,
                    X_reference=X_test,
                    feature_names=feature_names,
                )

                if ridge_result is not None:
                    feature_importance_records[model_key] = ridge_result[
                        "coefficient_standardized_abs"
                    ]
                    ridge_signed_coefficients = ridge_result[
                        "coefficient_standardized"
                    ]

            else:

                importance = _extract_feature_importance(
                    model=model,
                    feature_names=feature_names,
                )

                if importance is not None:
                    feature_importance_records[model_key] = importance

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
        # Feature importance comparison (XGBoost / Random Forest / Ridge),
        # for whichever of the three loaded successfully in this variant.
        # ---------------------------------------------------------------------

        if feature_importance_records:

            all_feature_importance[variant_name] = dict(
                feature_importance_records
            )

            importance_table = pd.DataFrame(feature_importance_records)
            importance_table.index.name = "feature"

            rename_map = {
                key: (
                    "ridge_coefficient_abs_standardized"
                    if key == "ridge"
                    else f"{key}_importance"
                )
                for key in importance_table.columns
            }
            importance_table = importance_table.rename(columns=rename_map)

            if ridge_signed_coefficients is not None:
                importance_table["ridge_coefficient_signed"] = ridge_signed_coefficients

            importance_table.to_csv(
                variant_results_dir / "feature_importance_comparison.csv"
            )

            print(
                f"\n  Feature importance comparison saved: "
                f"{len(feature_importance_records)} model(s) "
                f"({', '.join(feature_importance_records.keys())})"
            )

            _plot_feature_importance_comparison(
                feature_importance_records,
                variant_results_dir / "08_feature_importance_comparison.png",
            )

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
    # Global feature importance comparison (with_trophies + without_trophies)
    # -------------------------------------------------------------------------

    print(
        "\nBuilding global feature importance comparison..."
    )

    try:

        _save_global_feature_importance_comparison(
            all_feature_importance=all_feature_importance,
            results_dir=RESULTS_DIR,
            top_n=GLOBAL_IMPORTANCE_TOP_N,
        )

    except Exception as exc:

        # Extra output only: whatever happens here, the metrics tables,
        # the per-model plots and results_summary.json are not affected.
        print(
            f"\nWARNING: Could not build the global feature importance "
            f"comparison: {type(exc).__name__}: {exc}"
        )

        print(
            traceback.format_exc()
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
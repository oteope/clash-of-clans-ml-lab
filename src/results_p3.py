import sys
import os
import json
import gc
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, unquote

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
    max_error,
)

from mlflow_tracking.tracking_utils import configure_tracking


# =============================================================================
# REPOSITORY ROOT
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# =============================================================================
# CONFIGURATION
# =============================================================================
#
# Fill these in manually with the real Run IDs from your MLflow experiment.
# Left as None on purpose -- a model with run_id=None is skipped cleanly
# (see main()) instead of being attempted with a made-up ID.
# =============================================================================

MODEL_RUN_IDS = {
    "linear_regression": "b6335fa7af0d497ea722eb96255eb151",
    "random_forest": "b812bc69c0444d84a51d861612e61cb6",
    "gradient_boosting": "fdeb45c2be044680b24fecdecdc202e5",
}


# =============================================================================
# GENERAL CONFIGURATION
# =============================================================================

RESULTS_DIR = ROOT_DIR / "src" / "results" / "P3"

DATASET_PATH = (
    ROOT_DIR
    / "data"
    / "datasets"
    / "clan_war_performance_regression.parquet"
)

TARGET_COLUMN = "war_success_rate"

DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42

FEATURE_IMPORTANCE_TOP_N = 15


# =============================================================================
# VALIDATE RUN CONFIGURATION
# =============================================================================

def _validate_run_configuration() -> None:

    valid_models = {
        "linear_regression",
        "random_forest",
        "gradient_boosting",
    }

    for model_name, run_id in MODEL_RUN_IDS.items():

        if model_name not in valid_models:
            raise ValueError(
                f"Invalid model '{model_name}' in MODEL_RUN_IDS."
            )

        if run_id is None:
            continue

        if not isinstance(run_id, str):
            raise ValueError(
                f"Run ID for '{model_name}' "
                f"must be a string or None."
            )

        if not run_id.strip():
            raise ValueError(
                f"Run ID for '{model_name}' "
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
# Only player_tag / clan_tag / name are explicitly dropped (checked for
# existence first, same as P2). Nothing else is removed arbitrarily.
# Everything else is coerced to numeric; columns that are entirely
# non-numeric (pure-NaN after coercion) are dropped as a consequence of
# that coercion, not by name.
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
    )

    return test_indices


# =============================================================================
# TARGET SCALE DETECTION
# =============================================================================
#
# war_success_rate is a bounded rate, but whether it's stored as a 0-1
# fraction or a 0-100 percentage isn't specified anywhere in the request.
# Rather than assume one, this infers it from y_test itself, so the
# tolerance-band and out-of-bounds metrics below use sensible,
# scale-appropriate thresholds either way.
# =============================================================================

def _infer_target_scale(
    y_test: pd.Series,
) -> Dict[str, Any]:

    upper_observed = float(
        np.max(y_test)
    )

    if upper_observed <= 1.5:

        return {

            "lower_bound": 0.0,
            "upper_bound": 1.0,

            "tol_tight": 0.05,
            "tol_loose": 0.10,

            "unit": "fraction (0-1)",
        }

    return {

        "lower_bound": 0.0,
        "upper_bound": 100.0,

        "tol_tight": 5.0,
        "tol_loose": 10.0,

        "unit": "percentage points (0-100)",
    }


# =============================================================================
# MLflow 3 MODEL LOADING
# =============================================================================
#
# Same strategy as results_p2.py, validated directly against a real local
# MLflow 3 install (see that file's comments / the chat for the full
# writeup, and mlflow/mlflow#16429 and #16501 upstream):
# mlflow.sklearn.log_model(model, artifact_path="model", ...) does not
# write anything under the run's own classic artifact directory in
# current MLflow 3 -- runs:/<run_id>/model legitimately has nothing to
# find. run.outputs.model_outputs -> models:/<model_id> is the reliable
# path; legacy runs:/ and a direct local-artifact fallback are kept for
# defense in depth (e.g. a run logged under an older MLflow install).
#
# All three P3 models (LinearRegression, RandomForestRegressor,
# GradientBoostingRegressor) are logged via mlflow.sklearn.log_model() by
# log_model_and_artifacts() -- none of them are xgboost -- so this loads
# with mlflow.sklearn.load_model() only, no flavor dispatch needed.
# =============================================================================

def _file_uri_to_local_path(uri: str) -> Path:
    """
    Convert a file:// artifact URI (as stored by MLflow -- percent-encoded,
    e.g. spaces as %20) into a real local Path. Handles the Windows
    leading-slash-before-drive-letter case that a bare urlparse().path
    would otherwise leave in the string.
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


def _load_model_from_run(
    run_id: str,
    model_name: str,
) -> Any:
    """
    Load a model logged with MLflow 3 via log_model_and_artifacts().

    Strategy, in order (each step prints exactly what it tried and what
    it found, so a failure shows the real error and the real path, not a
    generic message):

      1. run.outputs.model_outputs -> models:/<model_id>, loaded with
         mlflow.sklearn.load_model(). The reliable MLflow-3-native path.
      2. Legacy runs:/<run_id>/model and runs:/<run_id>/modelo.
      3. Direct local artifact inspection: resolve the run's and/or the
         LoggedModel's artifact_location file:// URI to a real path,
         list what's actually there, and load straight from whatever
         local directory actually contains an MLmodel file.
      4. Raw pickle.load() on model.pkl next to that MLmodel file, as an
         absolute last resort (matches serialization_format="pickle").

    Returns the raw sklearn estimator, never a pyfunc wrapper, so
    n_features_in_ / feature_importances_ / coef_ and .predict() all
    work directly with no unwrapping.

    Raises the underlying exception (never swallows it) if every
    strategy fails, so the caller can log model / Run ID / method /
    full error and move on to the next model.
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
                loaded_model = mlflow.sklearn.load_model(model_uri)
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
            loaded_model = mlflow.sklearn.load_model(fallback_uri)
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
            loaded_model = mlflow.sklearn.load_model(str(mlmodel_dir))
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
        f"Could not load model '{model_name}' from run {run_id}.\n"
        f"Tried: MLflow 3 Logged Model (models:/<model_id>), legacy "
        f"runs:/ paths, and direct local artifact inspection. See the "
        f"diagnostic output above for the real error and exact paths "
        f"tried at each step."
    )


# =============================================================================
# MODEL INPUT
# =============================================================================

def _prepare_model_input(
    model_name: str,
    model: Any,
    X_test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate X_test against the model's expected feature COUNT via
    n_features_in_ (not exclusively feature names -- MLflow-loaded models
    don't expose feature_names_in_ consistently across flavors/versions).
    If the model doesn't expose n_features_in_ at all, proceed without
    validation rather than guessing.
    """

    expected_count = getattr(
        model,
        "n_features_in_",
        None,
    )

    if expected_count is not None:

        actual_count = X_test.shape[1]

        if actual_count != expected_count:
            raise RuntimeError(
                f"Feature count mismatch for {model_name}.\n"
                f"Model expects: {expected_count}. "
                f"Dataset provides: {actual_count}."
            )

        print(
            f"    Feature count validated: {expected_count}"
        )

    else:

        print(
            "    Model does not expose feature count "
            "(n_features_in_ not available)."
        )

        print(
            f"    Using dataset feature matrix directly: {X_test.shape}"
        )

    return X_test


# =============================================================================
# FEATURE IMPORTANCE / COEFFICIENTS
# =============================================================================
#
# Extracted right after a model loads successfully, and only the small
# resulting array is kept -- not the model itself -- so the "one model in
# memory at a time" rule downstream isn't affected by wanting a
# cross-model comparison later.
# =============================================================================

def _extract_feature_importance(
    model: Any,
    feature_names: List[str],
) -> Optional[pd.Series]:

    importances = getattr(
        model,
        "feature_importances_",
        None,
    )

    if importances is None:
        return None

    return pd.Series(
        importances,
        index=feature_names,
    )


def _extract_linear_coefficients(
    model: Any,
    feature_names: List[str],
) -> Optional[pd.Series]:

    coefficients = getattr(
        model,
        "coef_",
        None,
    )

    if coefficients is None:
        return None

    coefficients = np.asarray(
        coefficients
    ).ravel()

    if len(coefficients) != len(feature_names):
        return None

    return pd.Series(
        coefficients,
        index=feature_names,
    )


# =============================================================================
# METRICS
# =============================================================================
#
# Beyond MAE/MSE/RMSE/R², these add:
#   - medae            median absolute error -- robust to outlier clans,
#                       complements MAE when the error distribution is
#                       skewed (a few very poorly-predicted clans
#                       shouldn't dominate the headline number).
#   - explained_variance  differs from R2 when a model is biased but well
#                       correlated with the target -- seeing the two
#                       diverge is itself informative.
#   - max_error         worst single prediction -- tail risk that an
#                       average-based metric hides entirely.
#   - mean_bias         signed mean residual -- is the model
#                       systematically over- or under-predicting war
#                       success rate, not just "how far off on average".
#   - within_tight_pct / within_loose_pct
#                       % of test clans predicted within a tight/loose
#                       tolerance band (scale-adjusted, see
#                       _infer_target_scale) -- a directly interpretable
#                       "how often is this actually close enough" number,
#                       which RMSE alone doesn't give you.
#   - pct_out_of_bounds a plain regressor has no constraint keeping
#                       predictions inside a valid rate's bounds; this
#                       flags how often it produces a nonsensical value
#                       (e.g. a success rate below 0 or above 100%).
# =============================================================================

def _extract_metrics(
    y_true: pd.Series,
    y_pred: Any,
    scale_info: Dict[str, Any],
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

    residuals = y_true - y_pred

    abs_errors = np.abs(
        residuals
    )

    mse = mean_squared_error(
        y_true,
        y_pred,
    )

    tol_tight = scale_info["tol_tight"]
    tol_loose = scale_info["tol_loose"]

    lower_bound = scale_info["lower_bound"]
    upper_bound = scale_info["upper_bound"]

    out_of_bounds = (
        (y_pred < lower_bound)
        | (y_pred > upper_bound)
    )

    return {

        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),

        "medae": float(
            median_absolute_error(
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

        "explained_variance": float(
            explained_variance_score(
                y_true,
                y_pred,
            )
        ),

        "max_error": float(
            max_error(
                y_true,
                y_pred,
            )
        ),

        "mean_bias": float(
            np.mean(residuals)
        ),

        "within_tight_pct": float(
            np.mean(abs_errors <= tol_tight) * 100.0
        ),

        "within_loose_pct": float(
            np.mean(abs_errors <= tol_loose) * 100.0
        ),

        "pct_out_of_bounds": float(
            np.mean(out_of_bounds) * 100.0
        ),
    }


# =============================================================================
# ACTUAL VS PREDICTED
# =============================================================================

def _plot_actual_vs_predicted(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
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
        "Actual war_success_rate"
    )

    plt.ylabel(
        "Predicted war_success_rate"
    )

    plt.title(
        f"Actual vs predicted - {model_name}"
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
        "Predicted war_success_rate"
    )

    plt.ylabel(
        "Residual (actual - predicted)"
    )

    plt.title(
        f"Residuals - {model_name}"
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
        f"Residual distribution - {model_name}"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


# =============================================================================
# GLOBAL MODEL COMPARISON - ERROR METRICS
# =============================================================================

def _plot_global_metrics_comparison(
    metrics_df: pd.DataFrame,
    output_dir: Path,
) -> None:

    plot_df = metrics_df.set_index(
        "model"
    )

    # -------------------------------------------------------------------------
    # MAE / MedAE / RMSE
    # -------------------------------------------------------------------------

    plot_df[
        [
            "mae",
            "medae",
            "rmse",
        ]
    ].plot(
        kind="bar",
        figsize=(10, 7),
        rot=30,
    )

    plt.title(
        "MAE, MedAE and RMSE comparison"
    )

    plt.ylabel(
        "Error"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / "06_mae_rmse_comparison.png",
        dpi=150,
    )

    plt.close()

    # -------------------------------------------------------------------------
    # R2 / Explained variance
    # -------------------------------------------------------------------------

    plot_df[
        [
            "r2",
            "explained_variance",
        ]
    ].plot(
        kind="bar",
        figsize=(10, 7),
        rot=30,
    )

    plt.title(
        "R\u00b2 and explained variance comparison"
    )

    plt.ylabel(
        "Score"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / "07_r2_comparison.png",
        dpi=150,
    )

    plt.close()


# =============================================================================
# GLOBAL MODEL COMPARISON - ERROR TOLERANCE
# =============================================================================

def _plot_error_tolerance_comparison(
    metrics_df: pd.DataFrame,
    scale_info: Dict[str, Any],
    output_path: Path,
) -> None:

    plot_df = metrics_df.set_index(
        "model"
    )[
        [
            "within_tight_pct",
            "within_loose_pct",
        ]
    ]

    plot_df.columns = [
        f"within \u00b1{scale_info['tol_tight']:g}",
        f"within \u00b1{scale_info['tol_loose']:g}",
    ]

    plot_df.plot(
        kind="bar",
        figsize=(10, 7),
        rot=30,
    )

    plt.title(
        "Predictions within tolerance "
        f"({scale_info['unit']})"
    )

    plt.ylabel(
        "% of test predictions"
    )

    plt.ylim(0, 100)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


# =============================================================================
# GLOBAL MODEL COMPARISON - FEATURE IMPORTANCE
# =============================================================================

def _plot_feature_importance_comparison(
    importance_by_model: Dict[str, pd.Series],
    output_path: Path,
    top_n: int = FEATURE_IMPORTANCE_TOP_N,
) -> None:

    if not importance_by_model:
        return

    combined = pd.DataFrame(
        importance_by_model
    ).fillna(0.0)

    ranking = combined.mean(
        axis=1
    ).sort_values(
        ascending=False
    )

    top_features = ranking.head(
        top_n
    ).index

    plot_df = combined.loc[
        top_features
    ]

    plot_df.plot(
        kind="barh",
        figsize=(10, 8),
    )

    plt.gca().invert_yaxis()

    plt.title(
        f"Feature importance comparison (top {len(top_features)})"
    )

    plt.xlabel(
        "Importance"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


# =============================================================================
# EVALUATE ONE MODEL
# =============================================================================

def _evaluate_model(
    model_name: str,
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scale_info: Dict[str, Any],
) -> Tuple[Dict[str, float], np.ndarray]:

    print(
        f"\nEvaluating {model_name}..."
    )

    X_test_model = _prepare_model_input(
        model_name=model_name,
        model=model,
        X_test=X_test,
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
        scale_info,
    )

    print(
        f"MAE   = {metrics['mae']:.4f}"
    )

    print(
        f"MedAE = {metrics['medae']:.4f}"
    )

    print(
        f"MSE   = {metrics['mse']:.4f}"
    )

    print(
        f"RMSE  = {metrics['rmse']:.4f}"
    )

    print(
        f"R\u00b2    = {metrics['r2']:.4f}"
    )

    print(
        f"Explained variance = {metrics['explained_variance']:.4f}"
    )

    print(
        f"Max error   = {metrics['max_error']:.4f}"
    )

    print(
        f"Mean bias   = {metrics['mean_bias']:.4f} "
        f"({'over' if metrics['mean_bias'] < 0 else 'under'}-predicting on average)"
    )

    print(
        f"Within \u00b1{scale_info['tol_tight']:g} "
        f"({scale_info['unit']}): {metrics['within_tight_pct']:.1f}%"
    )

    print(
        f"Within \u00b1{scale_info['tol_loose']:g} "
        f"({scale_info['unit']}): {metrics['within_loose_pct']:.1f}%"
    )

    if metrics["pct_out_of_bounds"] > 0:

        print(
            f"WARNING: {metrics['pct_out_of_bounds']:.1f}% of predictions "
            f"fall outside the valid "
            f"[{scale_info['lower_bound']:g}, {scale_info['upper_bound']:g}] range"
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
        "P3 FINAL RESULTS - WAR SUCCESS RATE REGRESSION"
    )

    print(
        "=" * 80
        + "\n"
    )

    # -------------------------------------------------------------------------
    # [1/6]
    # -------------------------------------------------------------------------

    print(
        "[1/6] Validating MLflow Run ID configuration..."
    )

    _validate_run_configuration()

    print(
        "\u2713 Run configuration valid\n"
    )

    for model_name, run_id in MODEL_RUN_IDS.items():

        if run_id is None:
            print(f"  {model_name}: SKIPPED (no Run ID configured)")
        else:
            print(f"  {model_name}: {run_id}")

    print()

    # -------------------------------------------------------------------------
    # [2/6]
    # -------------------------------------------------------------------------

    print(
        "[2/6] Configuring MLflow..."
    )

    configure_tracking()

    print(
        "\u2713 MLflow configured\n"
    )

    # -------------------------------------------------------------------------
    # [3/6]
    # -------------------------------------------------------------------------

    print(
        "[3/6] Loading dataset..."
    )

    data = _load_dataset(
        DATASET_PATH
    )

    print(
        f"Dataset shape: {data.shape}"
    )

    X, y = _prepare_features(
        data
    )

    print(
        f"Feature matrix shape: {X.shape}"
    )

    print(
        f"Number of available features: {X.shape[1]}"
    )

    del data
    gc.collect()

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

    feature_names = list(
        X_test.columns
    )

    print(
        f"Test shape: {X_test.shape}"
    )

    print(
        f"Test target size: {len(y_test)}"
    )

    scale_info = _infer_target_scale(
        y_test
    )

    print(
        f"Detected target scale: {scale_info['unit']}, "
        f"bounds [{scale_info['lower_bound']:g}, {scale_info['upper_bound']:g}], "
        f"tolerances \u00b1{scale_info['tol_tight']:g} / "
        f"\u00b1{scale_info['tol_loose']:g}\n"
    )

    del X
    del y
    del test_indices

    gc.collect()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # [4/6]
    # -------------------------------------------------------------------------

    print(
        "[4/6] Evaluating models...\n"
    )

    all_metrics = []

    feature_importance_records: Dict[str, pd.Series] = {}
    linear_coefficient_records: Dict[str, pd.Series] = {}

    for model_name, run_id in MODEL_RUN_IDS.items():

        if run_id is None:

            print(
                f"Skipping {model_name}: no Run ID configured."
            )

            continue

        model = None
        y_pred = None

        # -----------------------------------------------------------------
        # Load
        # -----------------------------------------------------------------

        print(
            f"\nLoading {model_name}..."
        )

        try:

            model = _load_model_from_run(
                run_id=run_id,
                model_name=model_name,
            )

            print(
                f"\u2713 {model_name} loaded"
            )

        except Exception as exc:

            print(
                f"\nWARNING: Could not load {model_name}."
            )

            print(
                f"  Model: {model_name}"
            )

            print(
                f"  Run ID: {run_id}"
            )

            print(
                f"  Load method attempted: MLflow 3 Logged Model "
                f"(models:/<model_id>), legacy runs:/, direct local "
                f"artifact inspection"
            )

            print(
                f"  Error: {exc}"
            )

            continue

        # -----------------------------------------------------------------
        # Feature importance / coefficients (lightweight -- keep the
        # small array, not the model, for the cross-model comparison
        # built after every model has been freed).
        # -----------------------------------------------------------------

        importance = _extract_feature_importance(
            model=model,
            feature_names=feature_names,
        )

        if importance is not None:
            feature_importance_records[model_name] = importance

        linear_coefficients = _extract_linear_coefficients(
            model=model,
            feature_names=feature_names,
        )

        if linear_coefficients is not None:
            linear_coefficient_records[model_name] = linear_coefficients

        # -----------------------------------------------------------------
        # Evaluate
        # -----------------------------------------------------------------

        try:

            metrics, y_pred = _evaluate_model(
                model_name=model_name,
                model=model,
                X_test=X_test,
                y_test=y_test,
                scale_info=scale_info,
            )

        except Exception as exc:

            print(
                f"\nWARNING: Could not evaluate {model_name}."
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

        metrics["model"] = model_name
        metrics["run_id"] = run_id

        all_metrics.append(
            metrics
        )

        # -----------------------------------------------------------------
        # Plots
        # -----------------------------------------------------------------

        _plot_actual_vs_predicted(
            y_true=y_test,
            y_pred=y_pred,
            model_name=model_name,
            output_path=(
                RESULTS_DIR
                / f"01_actual_vs_predicted_{model_name}.png"
            ),
        )

        _plot_residuals(
            y_true=y_test,
            y_pred=y_pred,
            model_name=model_name,
            output_path=(
                RESULTS_DIR
                / f"02_residuals_{model_name}.png"
            ),
        )

        _plot_residual_distribution(
            y_true=y_test,
            y_pred=y_pred,
            model_name=model_name,
            output_path=(
                RESULTS_DIR
                / f"03_residual_distribution_{model_name}.png"
            ),
        )

        print(
            f"\u2713 {model_name} completed"
        )

        # -----------------------------------------------------------------
        # Memory cleanup
        # -----------------------------------------------------------------

        del model
        del y_pred

        model = None
        y_pred = None

        gc.collect()

    del X_test
    del y_test

    gc.collect()

    # =========================================================================
    # [5/6]
    # =========================================================================

    print(
        "\n[5/6] Building final comparison tables..."
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

    csv_columns = [
        "model",
        "mae",
        "medae",
        "mse",
        "rmse",
        "r2",
        "explained_variance",
        "max_error",
        "mean_bias",
        "within_tight_pct",
        "within_loose_pct",
        "pct_out_of_bounds",
        "run_id",
    ]

    final_csv_df = metrics_df[
        csv_columns
    ].copy()

    final_csv_df.to_csv(
        RESULTS_DIR
        / "final_model_comparison.csv",
        index=False,
    )

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

    best_model_df = final_csv_df.head(1)

    best_model_df.to_csv(
        RESULTS_DIR
        / "best_model.csv",
        index=False,
    )

    _plot_global_metrics_comparison(
        metrics_df,
        RESULTS_DIR,
    )

    _plot_error_tolerance_comparison(
        metrics_df,
        scale_info,
        RESULTS_DIR / "09_error_tolerance_comparison.png",
    )

    if feature_importance_records:

        _plot_feature_importance_comparison(
            feature_importance_records,
            RESULTS_DIR / "08_feature_importance_comparison.png",
        )

        print(
            "Feature importance (tree-based models) -- "
            f"top {FEATURE_IMPORTANCE_TOP_N} saved to "
            "08_feature_importance_comparison.png\n"
        )

    if linear_coefficient_records:

        print(
            "Linear Regression -- largest-magnitude coefficients "
            "(raw, unstandardized: reflects effect size AND feature "
            "scale, not directly comparable to tree feature_importances_):"
        )

        for model_name, coefficients in linear_coefficient_records.items():

            ranked = coefficients.reindex(
                coefficients.abs().sort_values(
                    ascending=False
                ).index
            ).head(10)

            print(
                f"  {model_name}:"
            )

            for feature_name, value in ranked.items():

                print(
                    f"    {feature_name}: {value:.4f}"
                )

        print()

    summary = {

        "dataset": str(
            DATASET_PATH
        ),

        "target": TARGET_COLUMN,

        "test_size": DEFAULT_TEST_SIZE,

        "random_state": DEFAULT_RANDOM_STATE,

        "target_scale": scale_info,

        "models": MODEL_RUN_IDS,

        "results": metrics_df[
            csv_columns
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

    # =========================================================================
    # [6/6]
    # =========================================================================

    print(
        "\n[6/6] Results generated successfully!"
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
        "P3 RESULTS COMPLETED"
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
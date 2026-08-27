import sys
import json
import gc
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
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

RESULTS_DIR = ROOT_DIR / "src" / "results" / "P2"

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

def _load_dataset(
    path: Path,
) -> pd.DataFrame:

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

def _prepare_features(
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:

    y = data[TARGET_COLUMN].copy()

    X = data.drop(
        columns=[TARGET_COLUMN]
    ).copy()

    # -------------------------------------------------------------------------
    # Remove identifiers
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Convert numeric-looking columns
    # -------------------------------------------------------------------------

    X = X.apply(
        pd.to_numeric,
        errors="coerce",
    )

    # -------------------------------------------------------------------------
    # Boolean features
    #
    # Keep them exactly as modelling columns, converting them to integers.
    # -------------------------------------------------------------------------

    bool_columns = X.select_dtypes(
        include=["bool"]
    ).columns.tolist()

    for column in bool_columns:
        X[column] = (
            X[column]
            .fillna(False)
            .astype(np.int8)
        )

    # -------------------------------------------------------------------------
    # Columns that are completely NaN
    #
    # For the generated datasets, these are not useful modelling columns.
    # -------------------------------------------------------------------------

    all_nan_columns = X.columns[
        X.isna().all()
    ].tolist()

    if all_nan_columns:
        X = X.drop(
            columns=all_nan_columns
        )

    # -------------------------------------------------------------------------
    # Reduce memory usage.
    #
    # The models accept float32 input and this considerably reduces RAM
    # consumption on the 837k-row datasets.
    # -------------------------------------------------------------------------

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
    """
    Reproduce sklearn train_test_split without materializing X_train.

    This is important for P2 because the datasets contain ~837k rows.
    """

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
# MODEL LOADING
# =============================================================================

def _load_model_from_run(
    run_id: str,
):
    """
    Same basic MLflow loading strategy used by P1.

    First:
        runs:/<run_id>/model

    Fallback:
        runs:/<run_id>/modelo
    """

    model_uri = f"runs:/{run_id}/model"

    try:
        return mlflow.pyfunc.load_model(
            model_uri
        )

    except Exception as first_error:

        fallback_uri = f"runs:/{run_id}/modelo"

        try:
            return mlflow.pyfunc.load_model(
                fallback_uri
            )

        except Exception as second_error:

            raise RuntimeError(
                f"Could not load model from run {run_id}.\n"
                f"Tried:\n"
                f"  {model_uri}\n"
                f"  {fallback_uri}\n"
                f"Original error: {first_error}\n"
                f"Fallback error: {second_error}"
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
    Prepare model input.

    IMPORTANT:
    P2 datasets are already generated in the exact feature order used
    by the regression training pipeline.

    Unlike the previous implementation, this function does NOT attempt
    to recover feature names from the MLflow model.

    P1 follows the same philosophy: the evaluation dataset is rebuilt
    using the project's feature pipeline and then passed directly to
    model.predict().
    """

    X_model = X_test.copy()

    # -------------------------------------------------------------------------
    # Validate model feature count when the underlying model exposes it.
    # This does not require feature names.
    # -------------------------------------------------------------------------

    expected_count = None

    # PyFunc -> underlying sklearn/xgboost implementation
    candidates = [model]

    try:
        python_model = model.unwrap_python_model()

        candidates.append(
            python_model
        )

        inner_model = getattr(
            python_model,
            "model",
            None,
        )

        if inner_model is not None:
            candidates.append(
                inner_model
            )

    except Exception:
        pass

    try:
        impl = getattr(
            model,
            "_model_impl",
            None,
        )

        if impl is not None:
            candidates.append(
                impl
            )

            inner_model = getattr(
                impl,
                "model",
                None,
            )

            if inner_model is not None:
                candidates.append(
                    inner_model
                )

    except Exception:
        pass

    for candidate in candidates:

        try:
            n_features = getattr(
                candidate,
                "n_features_in_",
                None,
            )

            if n_features is not None:
                expected_count = int(
                    n_features
                )
                break

        except Exception:
            pass

        try:
            booster = candidate.get_booster()

            feature_names = getattr(
                booster,
                "feature_names",
                None,
            )

            if feature_names:
                expected_count = len(
                    feature_names
                )
                break

        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Count validation only.
    #
    # We deliberately do NOT use model feature names because the stored
    # MLflow models do not expose them reliably.
    # -------------------------------------------------------------------------

    if expected_count is not None:

        actual_count = X_model.shape[1]

        if actual_count != expected_count:

            raise RuntimeError(
                f"Feature count mismatch for {model_key}.\n"
                f"Model expects: {expected_count}\n"
                f"Dataset provides: {actual_count}"
            )

        print(
            f"    Feature count validated: "
            f"{actual_count}"
        )

    else:

        print(
            f"    Model does not expose feature count."
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
        "R² comparison"
    )

    plt.ylabel(
        "R²"
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

    # -------------------------------------------------------------------------
    # Prepare input
    # -------------------------------------------------------------------------

    X_test_model = _prepare_model_input(
        model_key=model_key,
        model=model,
        X_test=X_test,
    )

    # -------------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

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
        f"    R²   = {metrics['r2']:.4f}"
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
    # Validate configuration
    # -------------------------------------------------------------------------

    print(
        "[1/6] Validating MLflow Run ID configuration..."
    )

    _validate_run_configuration()

    print(
        "✓ Run configuration valid\n"
    )

    # -------------------------------------------------------------------------
    # MLflow
    # -------------------------------------------------------------------------

    print(
        "[2/6] Configuring MLflow..."
    )

    configure_tracking()

    print(
        "✓ MLflow configured\n"
    )

    # -------------------------------------------------------------------------
    # Show configured runs
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
    # Evaluate
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
        # Prepare features
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
        # Recreate exact test split WITHOUT building X_train
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
        # Free full feature matrix
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
        #
        # IMPORTANT:
        # Load ONE model at a time.
        #
        # This prevents XGBoost + Random Forest + Ridge from simultaneously
        # occupying memory.
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
                    run_id
                )

                print(
                    f"  ✓ {model_key} loaded"
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

                # Free model before continuing
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
                f"  ✓ {model_key} completed"
            )

            # -----------------------------------------------------------------
            # CRITICAL MEMORY CLEANUP
            # -----------------------------------------------------------------

            del y_pred
            del model

            model = None
            y_pred = None

            gc.collect()

        # ---------------------------------------------------------------------
        # Free test data before next dataset
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

    # Lower RMSE is better.

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
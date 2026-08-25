import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
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
from mlflow_tracking.experiments import get_experiment_name


# =============================================================================
# CONSTANTS
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
# MODEL DISCOVERY
# =============================================================================
#
# El script intenta localizar automáticamente los últimos runs de cada modelo.
#
# Si tus nombres de MLflow contienen estas palabras, no necesitas introducir
# manualmente los run IDs.
#
# Se buscan:
#
#   XGBoost
#   Random Forest
#   Ridge
#   Linear Regression
#
# para cada una de las dos variantes del dataset.
#
# =============================================================================

MODEL_PATTERNS = {
    "xgboost": [
        "xgboost",
        "xgb",
    ],
    "random_forest": [
        "random_forest",
        "random forest",
        "randomforest",
        "rf",
    ],
    "ridge": [
        "ridge",
    ],
    "linear_regression": [
        "linear_regression",
        "linear regression",
        "linearregression",
        "ols",
    ],
}


# =============================================================================
# DATASET VARIANT IDENTIFICATION
# =============================================================================

WITH_TROPHIES_PATTERNS = [
    "with_trophies",
    "with trophies",
    "with-trophies",
    "trophies",
]

WITHOUT_TROPHIES_PATTERNS = [
    "without_trophies",
    "without trophies",
    "without-trophies",
    "no_trophies",
    "no trophies",
    "notrophies",
    "trophy_free",
    "trophy-free",
]


# =============================================================================
# FIND MLflow RUNS
# =============================================================================

def _get_all_runs() -> pd.DataFrame:
    """
    Retrieve all MLflow runs.

    The project uses MLflow tracking infrastructure configured through
    configure_tracking().
    """

    configure_tracking()

    return mlflow.search_runs(
        output_format="pandas",
    )


def _normalise_text(value: Any) -> str:
    """
    Convert arbitrary MLflow metadata to lowercase text.
    """

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return str(value).lower()


def _run_text(run: pd.Series) -> str:
    """
    Build a searchable text representation of an MLflow run.

    Run name, tags, parameters and experiment name are included.
    """

    pieces = []

    for column in run.index:

        if (
            "run_name" in column
            or "tags." in column
            or "params." in column
            or column == "experiment_name"
        ):

            pieces.append(
                _normalise_text(run[column])
            )

    return " ".join(pieces)


def _contains_any(
    text: str,
    patterns: List[str],
) -> bool:
    """
    Return True if any pattern occurs in text.
    """

    return any(
        pattern.lower() in text
        for pattern in patterns
    )


def _identify_dataset_variant(
    run: pd.Series,
) -> Optional[str]:
    """
    Identify whether an MLflow run belongs to the trophy or trophy-free
    dataset variant.

    Returns:
        'with_trophies'
        'without_trophies'
        None
    """

    text = _run_text(run)

    # Check trophy-free FIRST because it also contains "trophies".
    if _contains_any(
        text,
        WITHOUT_TROPHIES_PATTERNS,
    ):
        return "without_trophies"

    if _contains_any(
        text,
        WITH_TROPHIES_PATTERNS,
    ):
        return "with_trophies"

    return None


def _identify_model(
    run: pd.Series,
) -> Optional[str]:
    """
    Identify model family from MLflow metadata.
    """

    text = _run_text(run)

    # More specific patterns first.
    for model_name, patterns in MODEL_PATTERNS.items():

        if _contains_any(
            text,
            patterns,
        ):
            return model_name

    return None


def _find_latest_model_runs() -> Dict[str, Dict[str, str]]:
    """
    Automatically find the latest MLflow run for every model/dataset
    combination.

    Returns:

        {
            "with_trophies": {
                "xgboost": "...",
                "random_forest": "...",
                ...
            },
            "without_trophies": {
                ...
            }
        }

    Runs are ordered by start_time and the newest matching run is selected.
    """

    runs = _get_all_runs()

    if runs.empty:
        raise RuntimeError(
            "No MLflow runs were found."
        )

    if "start_time" in runs.columns:
        runs = runs.sort_values(
            "start_time",
            ascending=False,
        )

    selected = {
        "with_trophies": {},
        "without_trophies": {},
    }

    for _, run in runs.iterrows():

        variant = _identify_dataset_variant(
            run
        )

        if variant is None:
            continue

        model_name = _identify_model(
            run
        )

        if model_name is None:
            continue

        run_id = run.get(
            "run_id"
        )

        if not run_id:
            continue

        # Because runs are sorted newest -> oldest,
        # the first matching run is the newest one.
        if model_name not in selected[variant]:

            selected[variant][model_name] = str(
                run_id
            )

    return selected


# =============================================================================
# DATASET LOADING
# =============================================================================

def _load_dataset(
    path: Path,
) -> pd.DataFrame:
    """
    Load one P2 regression dataset.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    data = pd.read_parquet(
        path
    )

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
    """
    Prepare X and y for P2 regression.

    Excluded:
        - target
        - player_tag
        - clan_tag
        - name
        - object/string columns
        - boolean columns

    Numeric features are retained.

    """

    y = data[
        TARGET_COLUMN
    ].copy()

    X = data.drop(
        columns=[
            TARGET_COLUMN
        ]
    ).copy()

    # -------------------------------------------------------------------------
    # Identifiers / non-modelling columns
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
    # Remove object/string columns
    # -------------------------------------------------------------------------

    object_columns = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    if object_columns:

        X = X.drop(
            columns=object_columns
        )

    # -------------------------------------------------------------------------
    # Remove boolean columns
    # -------------------------------------------------------------------------

    bool_columns = X.select_dtypes(
        include=["bool"]
    ).columns.tolist()

    if bool_columns:

        X = X.drop(
            columns=bool_columns
        )

    # -------------------------------------------------------------------------
    # Convert remaining columns to numeric
    # -------------------------------------------------------------------------

    X = X.apply(
        pd.to_numeric,
        errors="coerce",
    )

    # -------------------------------------------------------------------------
    # Remove columns containing only NaN
    # -------------------------------------------------------------------------

    all_nan_columns = X.columns[
        X.isna().all()
    ].tolist()

    if all_nan_columns:

        X = X.drop(
            columns=all_nan_columns
        )

    return X, y


# =============================================================================
# TRAIN / TEST SPLIT
# =============================================================================

def _get_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
):
    """
    Recreate the project train/test split.
    """

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=None,
    )


# =============================================================================
# MODEL LOADING
# =============================================================================

def _load_model_from_run(
    run_id: str,
):
    """
    Load a model from an MLflow run.

    The training scripts log the model under the 'model' artifact name.
    """

    model_uri = (
        f"runs:/{run_id}/model"
    )

    try:

        return mlflow.pyfunc.load_model(
            model_uri
        )

    except Exception as first_error:

        # Some project runs may have logged the model using the older
        # 'modelo' artifact name.
        fallback_uri = (
            f"runs:/{run_id}/modelo"
        )

        try:

            return mlflow.pyfunc.load_model(
                fallback_uri
            )

        except Exception:

            raise RuntimeError(
                f"Could not load model from run "
                f"{run_id}.\n"
                f"Tried:\n"
                f"  {model_uri}\n"
                f"  {fallback_uri}\n"
                f"Original error: {first_error}"
            )


# =============================================================================
# METRICS
# =============================================================================

def _extract_metrics(
    y_true: pd.Series,
    y_pred: Any,
) -> Dict[str, float]:
    """
    Calculate regression metrics.
    """

    y_true = np.asarray(
        y_true
    ).ravel()

    y_pred = np.asarray(
        y_pred
    ).ravel()

    return {
        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),

        "mse": float(
            mean_squared_error(
                y_true,
                y_pred,
            )
        ),

        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
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
    """
    Plot actual vs predicted clan rank.
    """

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
    """
    Plot residuals against predicted values.
    """

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
# GLOBAL MODEL COMPARISON
# =============================================================================

def _plot_global_metrics_comparison(
    metrics_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Compare MAE, RMSE and R2 across all models and variants.

    MAE/RMSE are plotted separately from R2 because they have different
    scales and directions.
    """

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
# RESIDUAL DISTRIBUTION
# =============================================================================

def _plot_residual_distribution(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    variant_name: str,
    output_path: Path,
) -> None:
    """
    Plot residual distribution.
    """

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
    """
    Evaluate one regression model.
    """

    print(
        f"  Evaluating {model_name}..."
    )

    y_pred = model.predict(
        X_test
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

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # MLflow
    # -------------------------------------------------------------------------

    print(
        "[1/6] Configuring MLflow..."
    )

    configure_tracking()

    print(
        "✓ MLflow configured\n"
    )

    # -------------------------------------------------------------------------
    # Find models
    # -------------------------------------------------------------------------

    print(
        "[2/6] Discovering P2 MLflow model runs..."
    )

    model_runs = _find_latest_model_runs()

    print(
        "\nDiscovered runs:"
    )

    for variant, models in model_runs.items():

        print(
            f"\n{variant}:"
        )

        if not models:

            print(
                "  No models discovered."
            )

        for model_name, run_id in models.items():

            print(
                f"  {model_name}: {run_id}"
            )

    print()

    # -------------------------------------------------------------------------
    # Load datasets
    # -------------------------------------------------------------------------

    print(
        "[3/6] Loading datasets..."
    )

    datasets = {

        "with_trophies": (
            DATASET_WITH_TROPHIES
        ),

        "without_trophies": (
            DATASET_WITHOUT_TROPHIES
        ),
    }

    all_metrics = []

    # -------------------------------------------------------------------------
    # Evaluate variants
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
        # Build X / y
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
            f"Number of features: {X.shape[1]}"
        )

        # ---------------------------------------------------------------------
        # Split
        # ---------------------------------------------------------------------

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = _get_split(
            X,
            y,
        )

        print(
            f"Train shape: {X_train.shape}"
        )

        print(
            f"Test shape:  {X_test.shape}\n"
        )

        # ---------------------------------------------------------------------
        # Output directory for this variant
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
        # Evaluate models
        # ---------------------------------------------------------------------

        variant_runs = model_runs.get(
            variant_name,
            {},
        )

        for model_key, run_id in variant_runs.items():

            try:

                model = _load_model_from_run(
                    run_id
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

            model_display_name = (
                model_key
            )

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
                    f"  Error: {exc}"
                )

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
                f"  ✓ {model_key} completed\n"
            )

    # -------------------------------------------------------------------------
    # Build final DataFrame
    # -------------------------------------------------------------------------

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

    # Best model first:
    # lower RMSE is better.
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

        "models": model_runs,

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
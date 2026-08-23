import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

from src.features.problem1.build_role_dataset import (
    load_small_tables,
    assemble_role_dataset,
    PROCESSED_DIR,
)

from src.features.problem1.player_features import (
    build_player_features_from_files,
)

from src.features.problem1.player_clan_features import (
    compute_clan_relative_features,
)


# =============================================================================
# CONSTANTS
# =============================================================================

RESULTS_DIR = ROOT_DIR / "src" / "results" / "P1"

CLASS_ORDER = [
    "admin",
    "coLeader",
    "leader",
    "member",
]

TARGET_COLUMN = "role"

# Existing MLflow runs supplied for the final models
XGBOOST_RUN_ID = "11a727b624134920896bfefffc5abc90"
RF_RUN_ID = "40c9a36cbdd044fe83899ac0f8694178"

LOGISTIC_RUN_NAME = "logistic_regression_tuned_v4"


# =============================================================================
# LABEL NORMALIZATION
# =============================================================================

def _normalize_predictions(
    y_pred: Any,
    model_name: str = "",
) -> np.ndarray:
    """
    Normalize model predictions to the project's string class labels.

    Some MLflow models return:
        0, 1, 2, 3

    while the dataset target contains:
        admin, coLeader, leader, member

    This function supports both representations.
    """

    y_pred = np.asarray(y_pred).ravel()

    if y_pred.size == 0:
        return y_pred.astype(str)

    # -------------------------------------------------------------------------
    # Case 1: predictions are already strings
    # -------------------------------------------------------------------------

    if np.issubdtype(y_pred.dtype, np.str_) or y_pred.dtype == object:

        normalized = []

        for value in y_pred:

            # Already a valid class label
            if isinstance(value, str):
                if value in CLASS_ORDER:
                    normalized.append(value)
                    continue

                # Sometimes numbers arrive as strings: "0", "1", ...
                try:
                    numeric_value = int(value)

                    if 0 <= numeric_value < len(CLASS_ORDER):
                        normalized.append(CLASS_ORDER[numeric_value])
                        continue

                except (ValueError, TypeError):
                    pass

            # Numeric values inside object arrays
            try:
                numeric_value = int(value)

                if 0 <= numeric_value < len(CLASS_ORDER):
                    normalized.append(CLASS_ORDER[numeric_value])
                    continue

            except (ValueError, TypeError):
                pass

            raise ValueError(
                f"Unknown prediction label from {model_name!r}: {value!r}. "
                f"Expected one of {CLASS_ORDER} or numeric labels "
                f"0-{len(CLASS_ORDER) - 1}."
            )

        return np.asarray(normalized, dtype=str)

    # -------------------------------------------------------------------------
    # Case 2: numeric predictions
    # -------------------------------------------------------------------------

    if np.issubdtype(y_pred.dtype, np.number):

        numeric_predictions = y_pred.astype(int)

        invalid = (
            (numeric_predictions < 0)
            | (numeric_predictions >= len(CLASS_ORDER))
        )

        if invalid.any():
            invalid_values = np.unique(numeric_predictions[invalid])

            raise ValueError(
                f"Unknown numeric prediction labels from {model_name!r}: "
                f"{invalid_values.tolist()}. "
                f"Expected labels 0-{len(CLASS_ORDER) - 1}."
            )

        return np.asarray(
            [CLASS_ORDER[int(value)] for value in numeric_predictions],
            dtype=str,
        )

    raise ValueError(
        f"Unsupported prediction dtype from {model_name!r}: "
        f"{y_pred.dtype}"
    )


# =============================================================================
# FIND MLflow RUN
# =============================================================================

def _find_run_id_by_name(name: str) -> str:
    """
    Find an MLflow run by its run name.
    """

    configure_tracking()

    # First try P1 experiment
    try:

        experiment_name = get_experiment_name("p1")

        experiment = mlflow.get_experiment_by_name(
            experiment_name
        )

        if experiment is not None:

            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=(
                    f"tags.mlflow.runName = '{name}'"
                ),
            )

            if not runs.empty:
                return str(runs.iloc[0]["run_id"])

    except Exception:
        pass

    # Fallback: search all experiments
    all_runs = mlflow.search_runs(
        filter_string=f"tags.mlflow.runName = '{name}'"
    )

    if all_runs.empty:
        raise RuntimeError(
            f"Could not find an MLflow run with name: {name}"
        )

    return str(all_runs.iloc[0]["run_id"])


# =============================================================================
# LOAD MODEL
# =============================================================================

def _load_model_from_run(run_id: str):
    """
    Load model from an existing MLflow run.
    """

    model_uri = f"runs:/{run_id}/model"

    return mlflow.pyfunc.load_model(model_uri)


# =============================================================================
# PREPROCESSING CONFIG
# =============================================================================

def _parse_preprocessing_config(
    params: Dict[str, str],
) -> Dict[str, Any]:
    """
    Recover preprocessing configuration stored in MLflow.
    """

    raw = params.get("preprocessing_config")

    if raw is None:
        # Older runs may have logged an empty config.
        return {}

    try:
        config = json.loads(raw)

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "The 'preprocessing_config' parameter is not valid JSON: "
            + str(raw)
        ) from exc

    if not isinstance(config, dict):
        raise RuntimeError(
            "The 'preprocessing_config' parameter must be a JSON object."
        )

    return config


# =============================================================================
# TEST SIZE
# =============================================================================

def _get_test_size_from_config(
    config: Dict[str, Any],
) -> float:
    """
    Recover test_size from preprocessing configuration.
    """

    possible_direct_keys = (
        "test_size",
        "test_ratio",
        "test_fraction",
    )

    for key in possible_direct_keys:

        if key in config:

            try:
                return float(config[key])

            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Invalid test_size value: {config[key]}"
                ) from exc

    for nested_key in (
        "split",
        "data_split",
        "train_test_split",
    ):

        nested = config.get(nested_key)

        if isinstance(nested, dict):

            for key in possible_direct_keys:

                if key in nested:

                    try:
                        return float(nested[key])

                    except (TypeError, ValueError) as exc:
                        raise RuntimeError(
                            f"Invalid test_size value under "
                            f"'{nested_key}': {nested[key]}"
                        ) from exc

    # Current project uses 0.2
    return 0.2


# =============================================================================
# RANDOM STATE
# =============================================================================

def _get_random_state_from_params(
    params: Dict[str, str],
) -> int:
    """
    Recover random_state from MLflow parameters.
    """

    for key in (
        "random_seed",
        "seed",
        "random_state",
        "random_state_seed",
    ):

        if key in params:

            try:
                return int(params[key])

            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Invalid random_state value in parameter "
                    f"'{key}': {params[key]}"
                ) from exc

    # Current project uses 42
    return 42


# =============================================================================
# RECOVER CONSISTENT RUN CONFIG
# =============================================================================

def _get_consistent_run_config(
    run_ids: List[str],
) -> Tuple[float, int, Dict[str, Any]]:
    """
    Recover train/test configuration from MLflow.

    All final model runs must use the same split.
    """

    client = mlflow.tracking.MlflowClient()

    test_sizes = []
    random_states = []
    configs = []

    for run_id in run_ids:

        run = client.get_run(run_id)

        params = run.data.params

        random_state = _get_random_state_from_params(
            params
        )

        config = _parse_preprocessing_config(
            params
        )

        test_size = _get_test_size_from_config(
            config
        )

        test_sizes.append(test_size)
        random_states.append(random_state)
        configs.append(config)

    # Validate split consistency
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

    # Keep first config as canonical.
    #
    # Empty preprocessing configs are valid for this project because
    # categorical columns are not used in the current training pipeline.
    canonical_config = configs[0]

    return (
        test_sizes[0],
        random_states[0],
        canonical_config,
    )


# =============================================================================
# BUILD DATASET
# =============================================================================

def _build_dataset() -> pd.DataFrame:
    """
    Rebuild the exact P1 role classification dataset.
    """

    small_tables = load_small_tables()

    clans_df = small_tables["clans"]
    clan_members_df = small_tables["clan_members"]

    # Build player-level features
    pf = build_player_features_from_files(
        PROCESSED_DIR,
        batch_size=100_000,
    )

    # Build player-clan relative features
    pcf = compute_clan_relative_features(
        clan_members_df,
        pf,
    )

    # Assemble final dataset
    dataset = assemble_role_dataset(
        pf=pf,
        pcf=pcf,
        clan_members_df=clan_members_df,
        clans_df=clans_df,
    )

    return dataset


# =============================================================================
# CATEGORICAL COLUMNS
# =============================================================================

def _select_categorical_columns(
    features: pd.DataFrame,
    config: Dict[str, Any],
) -> List[str]:
    """
    Determine categorical columns.

    Current logistic regression pipeline does not use categorical
    features, so this normally returns an empty list.
    """

    for key in (
        "categorical_cols",
        "categorical_features",
        "categorical_columns",
    ):

        if key in config:

            raw_cols = config[key]

            if isinstance(raw_cols, list):

                return [
                    c
                    for c in raw_cols
                    if c in features.columns
                ]

            raise RuntimeError(
                f"'{key}' must be a list of column names."
            )

    # Only object columns are categorical.
    return list(
        features.select_dtypes(
            include="object"
        ).columns
    )


# =============================================================================
# TRAIN / TEST SPLIT
# =============================================================================

def _get_train_test_split(
    test_size: float,
    random_state: int,
    preprocessing_config: Dict[str, Any],
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Recreate the same train/test split used during training.
    """

    dataset = _build_dataset()

    target = dataset[TARGET_COLUMN]

    features = dataset.drop(
        columns=[TARGET_COLUMN]
    )

    # -------------------------------------------------------------------------
    # IMPORTANT:
    # Identifiers are NOT modelling features.
    # -------------------------------------------------------------------------

    for id_col in (
        "player_tag",
        "clan_tag",
    ):

        if id_col in features.columns:

            features = features.drop(
                columns=[id_col]
            )

    # -------------------------------------------------------------------------
    # Remove the same non-feature categorical/context columns that were
    # removed in the logistic regression training pipeline.
    #
    # This is particularly important because the training code explicitly
    # removed these columns.
    # -------------------------------------------------------------------------

    columns_to_remove = [
        "war_frequency",
        "war_league",
        "capital_league",
        "type",
        "is_family_friendly",
    ]

    for column in columns_to_remove:

        if column in features.columns:

            features = features.drop(
                columns=[column]
            )

    # -------------------------------------------------------------------------
    # Apply categorical encoding only if the preprocessing configuration
    # explicitly contains categorical columns.
    # -------------------------------------------------------------------------

    categorical_cols = _select_categorical_columns(
        features,
        preprocessing_config,
    )

    if categorical_cols:

        features = pd.get_dummies(
            features,
            columns=categorical_cols,
            drop_first=True,
        )

    # -------------------------------------------------------------------------
    # EXACT PROJECT SPLIT
    # -------------------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=None,
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# =============================================================================
# CLASS DISTRIBUTION
# =============================================================================

def _plot_class_distribution(
    y_test: pd.Series,
    output_path: Path,
) -> None:

    counts = (
        y_test
        .value_counts()
        .reindex(
            CLASS_ORDER,
            fill_value=0,
        )
    )

    plt.figure(figsize=(8, 5))

    counts.plot(
        kind="bar",
        color=[
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
        ],
    )

    plt.title(
        "Test set class distribution"
    )

    plt.xlabel("Class")
    plt.ylabel("Number of samples")

    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


# =============================================================================
# CONFUSION MATRIX
# =============================================================================

def _plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    output_path: Path,
) -> None:

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
    )

    plt.figure(figsize=(7, 6))

    plt.imshow(
        cm,
        interpolation="nearest",
        cmap=plt.cm.Blues,
    )

    plt.title(
        f"Confusion matrix - {model_name}"
    )

    plt.colorbar()

    tick_marks = np.arange(
        len(CLASS_ORDER)
    )

    plt.xticks(
        tick_marks,
        CLASS_ORDER,
        rotation=45,
    )

    plt.yticks(
        tick_marks,
        CLASS_ORDER,
    )

    threshold = cm.max() / 2.0

    for i, j in np.ndindex(cm.shape):

        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            horizontalalignment="center",
            color=(
                "white"
                if cm[i, j] > threshold
                else "black"
            ),
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


# =============================================================================
# GLOBAL METRICS PLOT
# =============================================================================

def _plot_global_metrics_comparison(
    metrics_df: pd.DataFrame,
    output_path: Path,
) -> None:

    metrics_to_plot = [
        "accuracy",
        "f1_weighted",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
    ]

    plot_df = (
        metrics_df[
            ["model"] + metrics_to_plot
        ]
        .set_index("model")
    )

    plot_df.plot(
        kind="bar",
        figsize=(10, 6),
        rot=0,
    )

    plt.title(
        "Global metrics comparison"
    )

    plt.ylabel("Score")

    plt.ylim(0, 1)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


# =============================================================================
# F1 BY CLASS
# =============================================================================

def _plot_f1_by_class(
    metrics_df: pd.DataFrame,
    output_path: Path,
) -> None:

    class_cols = [
        f"{c}_f1"
        for c in CLASS_ORDER
    ]

    plot_df = (
        metrics_df[
            ["model"] + class_cols
        ]
        .set_index("model")
    )

    plot_df.plot(
        kind="bar",
        figsize=(10, 6),
        rot=0,
    )

    plt.title(
        "F1 score by class"
    )

    plt.ylabel("F1")

    plt.ylim(0, 1)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


# =============================================================================
# LEADER METRICS
# =============================================================================

def _plot_leader_metrics(
    metrics_df: pd.DataFrame,
    output_path: Path,
) -> None:

    leader_cols = [
        "leader_precision",
        "leader_recall",
        "leader_f1",
    ]

    plot_df = (
        metrics_df[
            ["model"] + leader_cols
        ]
        .set_index("model")
    )

    plot_df.plot(
        kind="bar",
        figsize=(9, 6),
        rot=0,
    )

    plt.title(
        "Leader-specific metrics"
    )

    plt.ylabel("Score")

    plt.ylim(0, 1)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


# =============================================================================
# EXTRACT METRICS
# =============================================================================

def _extract_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> Dict[str, float]:

    # -------------------------------------------------------------------------
    # VERY IMPORTANT:
    # At this point both arrays must use the same string labels.
    # -------------------------------------------------------------------------

    y_true = np.asarray(
        y_true
    ).ravel()

    y_pred = np.asarray(
        y_pred
    ).ravel()

    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )

    metrics = {

        # Global
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        # Weighted
        "f1_weighted": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),

        "f1_macro": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),

        "precision_weighted": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),

        "recall_weighted": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
    }

    # Per-class metrics
    for cls in CLASS_ORDER:

        cls_report = report.get(
            cls,
            {},
        )

        metrics[
            f"{cls}_f1"
        ] = cls_report.get(
            "f1-score",
            0.0,
        )

        metrics[
            f"{cls}_precision"
        ] = cls_report.get(
            "precision",
            0.0,
        )

        metrics[
            f"{cls}_recall"
        ] = cls_report.get(
            "recall",
            0.0,
        )

    return metrics


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print(
        "\n"
        + "=" * 70
    )

    print(
        "P1 FINAL RESULTS"
    )

    print(
        "=" * 70
        + "\n"
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # MLflow
    # -------------------------------------------------------------------------

    configure_tracking()

    print(
        "[1/7] Finding Logistic Regression MLflow run..."
    )

    logistic_run_id = _find_run_id_by_name(
        LOGISTIC_RUN_NAME
    )

    print(
        f"Logistic Regression run: "
        f"{logistic_run_id}\n"
    )

    run_ids = [
        XGBOOST_RUN_ID,
        RF_RUN_ID,
        logistic_run_id,
    ]

    # -------------------------------------------------------------------------
    # Load models
    # -------------------------------------------------------------------------

    print(
        "[2/7] Loading models from MLflow..."
    )

    xgb_model = _load_model_from_run(
        XGBOOST_RUN_ID
    )

    print(
        "✓ XGBoost loaded"
    )

    rf_model = _load_model_from_run(
        RF_RUN_ID
    )

    print(
        "✓ Random Forest loaded"
    )

    logreg_model = _load_model_from_run(
        logistic_run_id
    )

    print(
        "✓ Logistic Regression loaded\n"
    )

    # -------------------------------------------------------------------------
    # Recover split
    # -------------------------------------------------------------------------

    print(
        "[3/7] Recovering train/test split configuration..."
    )

    (
        test_size,
        random_state,
        preprocessing_config,
    ) = _get_consistent_run_config(
        run_ids
    )

    print(
        f"test_size    = {test_size}"
    )

    print(
        f"random_state = {random_state}"
    )

    print(
        f"preprocessing_config = "
        f"{preprocessing_config}\n"
    )

    # -------------------------------------------------------------------------
    # Rebuild dataset
    # -------------------------------------------------------------------------

    print(
        "[4/7] Rebuilding dataset and test split..."
    )

    (
        _,
        X_test,
        _,
        y_test,
    ) = _get_train_test_split(
        test_size,
        random_state,
        preprocessing_config,
    )

    print(
        f"X_test shape = {X_test.shape}"
    )

    print(
        f"y_test size  = {len(y_test)}\n"
    )

    # -------------------------------------------------------------------------
    # Models
    # -------------------------------------------------------------------------

    models = {

        "xgboost": (
            xgb_model,
            "Xgboost_tuned_v8",
        ),

        "random_forest": (
            rf_model,
            "rf_tuned_v6",
        ),

        "logistic_regression": (
            logreg_model,
            "logistic_regression_tuned_v4",
        ),
    }

    all_metrics = []

    print(
        "[5/7] Generating evaluation results...\n"
    )

    # -------------------------------------------------------------------------
    # Class distribution
    # -------------------------------------------------------------------------

    _plot_class_distribution(
        y_test,
        RESULTS_DIR
        / "01_class_distribution.png",
    )

    # -------------------------------------------------------------------------
    # Evaluate each model
    # -------------------------------------------------------------------------

    for key, (
        model,
        model_name,
    ) in models.items():

        print(
            f"Evaluating {model_name}..."
        )

        # Prediction
        y_pred = model.predict(
            X_test
        )

        # Some model wrappers can return tensors.
        if hasattr(
            y_pred,
            "to_numpy",
        ):
            y_pred = y_pred.to_numpy()

        # ---------------------------------------------------------------------
        # FIX:
        #
        # XGBoost may return:
        #
        #     [0, 1, 2, 3]
        #
        # while y_test contains:
        #
        #     ["admin", "coLeader", "leader", "member"]
        #
        # Normalize before sklearn metrics.
        # ---------------------------------------------------------------------

        y_pred = _normalize_predictions(
            y_pred,
            model_name=model_name,
        )

        # Ensure y_true is also a clean string array
        y_true = np.asarray(
            y_test
        ).ravel().astype(str)

        # Debug information
        print(
            "  Prediction labels:",
            np.unique(y_pred),
        )

        print(
            "  True labels:",
            np.unique(y_true),
        )

        # ---------------------------------------------------------------------
        # Metrics
        # ---------------------------------------------------------------------

        metrics = _extract_metrics(
            y_true,
            y_pred,
        )

        metrics["model"] = model_name

        all_metrics.append(
            metrics
        )

        # ---------------------------------------------------------------------
        # Confusion matrix
        # ---------------------------------------------------------------------

        matrix_filename = {

            "xgboost":
                "02_confusion_matrix_xgboost.png",

            "random_forest":
                "02_confusion_matrix_random_forest.png",

            "logistic_regression":
                "02_confusion_matrix_logistic_regression.png",
        }[key]

        _plot_confusion_matrix(
            y_true,
            y_pred,
            model_name=model_name,
            output_path=(
                RESULTS_DIR
                / matrix_filename
            ),
        )

        print(
            f"✓ {model_name} evaluated\n"
        )

    # -------------------------------------------------------------------------
    # Final metrics DataFrame
    # -------------------------------------------------------------------------

    print(
        "[6/7] Building final comparison table..."
    )

    metrics_df = pd.DataFrame(
        all_metrics
    )

    # Sort by macro F1
    metrics_df = metrics_df.sort_values(
        "f1_macro",
        ascending=False,
    )

    # -------------------------------------------------------------------------
    # CSV
    # -------------------------------------------------------------------------

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

    final_csv_df = metrics_df[
        csv_columns
    ]

    final_csv_df.to_csv(
        RESULTS_DIR
        / "final_model_comparison.csv",
        index=False,
    )

    print()

    print(
        final_csv_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()

    # -------------------------------------------------------------------------
    # Remaining plots
    # -------------------------------------------------------------------------

    _plot_global_metrics_comparison(
        metrics_df,
        RESULTS_DIR
        / "03_global_metrics_comparison.png",
    )

    _plot_f1_by_class(
        metrics_df,
        RESULTS_DIR
        / "04_f1_by_class.png",
    )

    _plot_leader_metrics(
        metrics_df,
        RESULTS_DIR
        / "05_leader_metrics.png",
    )

    print(
        "[7/7] Results generated successfully!"
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
        RESULTS_DIR.iterdir()
    ):

        if file.is_file():

            print(
                f"  - {file.name}"
            )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "P1 RESULTS COMPLETED"
    )

    print(
        "=" * 70
        + "\n"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
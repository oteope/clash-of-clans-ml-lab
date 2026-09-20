import sys
import os
import json
import gc
import pickle
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, unquote

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
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
    "random_forest": "c5fb3f9cd6404e55aab6c81da2d6e7a0",
    "xgboost":"3cfa3696adf9476888419a9ef8d53ac0" ,
    "mlp": "c85903718da0498b95b3d8b75e02f194",
}


# =============================================================================
# GENERAL CONFIGURATION
# =============================================================================

RESULTS_DIR = ROOT_DIR / "src" / "results" / "P4"

DATASET_PATH = (
    ROOT_DIR
    / "data"
    / "datasets"
    / "clan_performance_classification.parquet"
)

TARGET_COLUMN = "performance_class"

# Matches src/features/problem4/build_performance_classification_dataset.py,
# which is the only place this project defines these three labels.
CLASS_ORDER = [
    "low",
    "medium",
    "high",
]

# The class this project's classification results scripts single out for a
# dedicated metrics chart (results_p1.py does this for "leader"). "high"
# performing clans are the most likely to be the actionable class here --
# change this if that's not the right read for how you'll use this.
KEY_CLASS = "high"

DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42

FEATURE_IMPORTANCE_TOP_N = 15


# =============================================================================
# VALIDATE RUN CONFIGURATION
# =============================================================================

def _validate_run_configuration() -> None:

    valid_models = {
        "random_forest",
        "xgboost",
        "mlp",
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

    unexpected_classes = (
        set(data[TARGET_COLUMN].dropna().unique())
        - set(CLASS_ORDER)
    )

    if unexpected_classes:
        raise RuntimeError(
            f"Found class labels not in CLASS_ORDER: "
            f"{sorted(unexpected_classes)}. "
            f"Expected only: {CLASS_ORDER}."
        )

    return data


# =============================================================================
# FEATURE PREPARATION
# =============================================================================
#
# Same approach as results_p2.py / results_p3.py: only player_tag /
# clan_tag / name are explicitly dropped (checked for existence first).
# Everything else is coerced to numeric; non-numeric columns disappear as
# a consequence of that coercion (all-NaN after coercion), not by name.
#
# build_performance_classification_dataset.py already strips the
# war-outcome columns that would leak the target (war_success_rate,
# win_rate, etc.) before this dataset is ever written to disk, so that
# doesn't need to be repeated here.
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
    # "object" by this point -- coercion still turns its True/False into
    # 1.0/0.0 correctly, but leaves actual missing entries as NaN rather
    # than False, and a dtype=="bool" check done AFTER coercion never
    # finds that column at all. Checking here, before coercion, catches
    # both cases.
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

    # stratify=None to match the same convention already used in
    # results_p1.py / results_p2.py / results_p3.py for this project. If
    # your P4 training actually split with stratify=y, tell me and this
    # changes -- an unstratified split here would then be evaluating on
    # different rows than training held out, not just a different split
    # style.
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
# PREDICTION LABEL NORMALIZATION
# =============================================================================
#
# RandomForestClassifier was fit on the raw string labels directly and
# predicts strings back -- confirmed empirically, and it never touches
# the numeric-decoding branch below at all, so nothing here can affect it.
#
# XGBClassifier does NOT accept string y -- it raises ValueError:
# "Invalid classes inferred from unique values of y" (confirmed
# empirically). MLP was ALSO fit on LabelEncoder-encoded integers in this
# project's actual training pipeline (confirmed against this project's
# own MLflow metrics: XGBoost/MLP's training-time accuracy only matches
# results_p4.py's output once decoded with the mapping below, not the
# CLASS_ORDER-indexed one this file used previously). Both predict
# integers back, and BOTH need decoding through the SAME mapping a bare
# sklearn LabelEncoder actually produces.
#
# LabelEncoder.fit() sorts unique classes LEXICOGRAPHICALLY before
# assigning 0, 1, 2, ... -- confirmed directly:
#   LabelEncoder().fit(["high","low","medium"]).classes_
#   -> array(['high', 'low', 'medium'])   # alphabetical, i.e. high=0, low=1, medium=2
#
# This is DELIBERATELY kept separate from CLASS_ORDER, which stays
# ["low", "medium", "high"] everywhere else in this file (confusion
# matrix axes, plots, CSV/JSON column order) -- only the numeric decode
# step below needs the alphabetical mapping, because that's what the
# encoder actually produced, not a display preference to match.
LABEL_ENCODER_CLASS_ORDER = sorted(CLASS_ORDER)


def _find_label_encoder_classes(run_id: str) -> Optional[List[str]]:
    """
    Best-effort recovery of the true class order from the run itself,
    tried BEFORE falling back to LABEL_ENCODER_CLASS_ORDER below: once a
    model is fit on bare integers, the fitted object itself retains no
    trace of the original strings at all (confirmed empirically --
    XGBClassifier/MLPClassifier.classes_ shows [0, 1, 2], nothing else on
    the object references them either), so this can only ever come from
    something logged separately alongside the run -- a param recording
    the mapping, or a saved LabelEncoder/classes artifact. If nothing
    like that is logged, returns None, which is the expected outcome
    here: LABEL_ENCODER_CLASS_ORDER is the actual, correct mapping for
    this project's models, not a guess to be replaced once something
    better is found.
    """

    client = MlflowClient()

    try:
        run = client.get_run(run_id)
    except Exception:
        return None

    param_candidates = (
        "label_encoder_classes", "le_classes", "class_order",
        "classes_", "label_classes", "encoder_classes",
    )

    for key, value in run.data.params.items():
        if key.lower() in param_candidates:
            try:
                parsed = json.loads(value.replace("'", '"'))
                if isinstance(parsed, list) and set(parsed) == set(CLASS_ORDER):
                    print(f"    Recovered class order from run param '{key}': {parsed}")
                    return parsed
            except Exception:
                pass

    try:
        artifacts = client.list_artifacts(run_id)
    except Exception:
        artifacts = []

    artifact_name_hints = ("label_encoder", "labelencoder", "classes")

    for artifact in artifacts:
        name_lower = artifact.path.lower()
        if any(hint in name_lower for hint in artifact_name_hints):
            try:
                local_path = mlflow.artifacts.download_artifacts(
                    run_id=run_id, artifact_path=artifact.path
                )
                if local_path.endswith((".pkl", ".pickle")):
                    with open(local_path, "rb") as f:
                        obj = pickle.load(f)
                    classes = list(getattr(obj, "classes_", obj))
                elif local_path.endswith(".json"):
                    with open(local_path) as f:
                        classes = json.load(f)
                else:
                    continue
                if set(classes) == set(CLASS_ORDER):
                    print(f"    Recovered class order from artifact '{artifact.path}': {classes}")
                    return list(classes)
            except Exception:
                pass

    return None


def _normalize_predictions(
    y_pred: Any,
    model_name: str = "",
    numeric_class_order: Optional[List[str]] = None,
) -> np.ndarray:

    class_order_for_numeric = numeric_class_order or LABEL_ENCODER_CLASS_ORDER

    y_pred = np.asarray(y_pred).ravel()

    if y_pred.size == 0:
        return y_pred.astype(str)

    if np.issubdtype(y_pred.dtype, np.str_) or y_pred.dtype == object:

        normalized = []

        for value in y_pred:

            if isinstance(value, str) and value in CLASS_ORDER:
                normalized.append(value)
                continue

            try:
                numeric_value = int(value)
                if 0 <= numeric_value < len(class_order_for_numeric):
                    normalized.append(class_order_for_numeric[numeric_value])
                    continue
            except (ValueError, TypeError):
                pass

            raise ValueError(
                f"Unknown prediction label from {model_name!r}: "
                f"{value!r}. Expected one of {CLASS_ORDER} or "
                f"numeric labels 0-{len(class_order_for_numeric) - 1}."
            )

        return np.asarray(normalized, dtype=str)

    if np.issubdtype(y_pred.dtype, np.number):

        numeric_predictions = y_pred.astype(int)

        invalid = (
            (numeric_predictions < 0)
            | (numeric_predictions >= len(class_order_for_numeric))
        )

        if invalid.any():
            raise ValueError(
                f"Unknown numeric prediction labels from "
                f"{model_name!r}: "
                f"{np.unique(numeric_predictions[invalid]).tolist()}. "
                f"Expected labels 0-{len(class_order_for_numeric) - 1}."
            )

        return np.asarray(
            [class_order_for_numeric[int(value)] for value in numeric_predictions],
            dtype=str,
        )

    raise ValueError(
        f"Unsupported prediction dtype from {model_name!r}: "
        f"{y_pred.dtype}"
    )


# =============================================================================
# MLflow 3 MODEL LOADING
# =============================================================================
#
# Same strategy as results_p2.py / results_p3.py, validated directly
# against a real local MLflow 3 install (see those files' comments / the
# chat for the full writeup, and mlflow/mlflow#16429 and #16501
# upstream): mlflow.<flavor>.log_model(model, artifact_path="model", ...)
# does not write anything under the run's own classic artifact directory
# in current MLflow 3 -- runs:/<run_id>/model legitimately has nothing to
# find. run.outputs.model_outputs -> models:/<model_id> is the reliable
# path; legacy runs:/ and a direct local-artifact fallback are kept for
# defense in depth.
#
# log_model_and_artifacts() routes xgboost models through
# mlflow.xgboost.log_model() and everything else (random_forest, mlp)
# through mlflow.sklearn.log_model() -- so loading needs the same
# per-flavor dispatch as results_p2.py, unlike results_p3.py which never
# had an xgboost model to worry about.
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


def _describe_local_artifacts(uri: str, label: str) -> None:
    """
    Resolve a file:// artifact URI to a local path and print what's there,
    WITH FILE SIZES, before any load is attempted. Some exceptions (a bare
    MemoryError in particular) stringify to an EMPTY string, which looks
    like a silent, unexplained failure unless the size was already
    visible up front.
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
    so we get the raw estimator back directly (RandomForestClassifier /
    XGBClassifier / MLPClassifier) rather than a PyFuncModel wrapper --
    simpler downstream (n_features_in_, .predict(), .predict_proba() all
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

    Strategy, in order (each step prints exactly what it tried and what
    it found):

      1. run.outputs.model_outputs -> models:/<model_id>. The reliable
         MLflow-3-native path.
      2. Legacy runs:/<run_id>/model and runs:/<run_id>/modelo.
      3. Direct local artifact inspection: resolve the run's and/or the
         LoggedModel's artifact_location file:// URI to a real path,
         list what's actually there (with file sizes), and load straight
         from whatever local directory actually contains an MLmodel file.
      4. Raw pickle.load() on model.pkl next to that MLmodel file, as an
         absolute last resort (matches serialization_format="pickle").

    Every exception is printed as "ExceptionType: message" rather than
    just "message" -- some exceptions (a bare MemoryError in particular,
    which is what a too-large-to-deserialize model tends to raise) print
    as an EMPTY string otherwise.

    Returns the raw flavor object, never a pyfunc wrapper. Raises the
    underlying exception if every strategy fails, so the caller can log
    model / Run ID / method / full error and move on to the next model.
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
                    "that blank is very likely a bare MemoryError, not "
                    "the path being wrong."
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
                    print("    Full traceback for this last attempt:")
                    print(
                        "    "
                        + traceback.format_exc().replace("\n", "\n    ")
                    )

    raise RuntimeError(
        f"Could not load model from run {run_id}.\n"
        f"Tried: MLflow 3 Logged Model (models:/<model_id>), legacy "
        f"runs:/ paths, and direct local artifact inspection. See the "
        f"diagnostic output above for the real error, the exception "
        f"TYPE, file sizes, and exact paths tried at each step."
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
    Validate X_test against the model's expected feature COUNT via
    n_features_in_ (not exclusively feature names -- MLflow-loaded models
    don't expose feature_names_in_ consistently across flavors).
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
                f"Feature count mismatch for {model_key}.\n"
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
# FEATURE IMPORTANCE
# =============================================================================

def _extract_feature_importance(
    model: Any,
    feature_names: List[str],
) -> Optional[pd.Series]:
    """
    Returns feature_importances_ indexed by name, or None if the model
    doesn't expose it.

    Prefers the model's OWN feature_names_in_ over the shared
    feature_names list built from X_test. Confirmed the hard way: if
    this specific model was trained on a different column subset than
    the other models -- the same kind of per-model inconsistency already
    found in P2 (capital_contributions present for some P2 models,
    absent for others) -- feature_importances_ comes back shorter than
    the shared list, and indexing it with the wrong names either raises
    ValueError: Length of values (N) does not match length of index (M),
    or, worse, would silently mislabel importances if the lengths
    happened to match by coincidence.
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
                    f"{len(feature_names)} columns) -- it was likely "
                    f"trained on a different column subset than the "
                    f"other models. Using this model's own feature "
                    f"names for its importance values."
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


# =============================================================================
# METRICS
# =============================================================================
#
# Beyond accuracy / per-class precision-recall-F1 (results_p1.py's set),
# this adds:
#   - balanced_accuracy   average per-class recall -- unlike plain
#                         accuracy, doesn't reward a model for just
#                         predicting the majority class when the three
#                         performance tiers aren't evenly sized (which,
#                         realistically, they won't be).
#   - roc_auc_ovr_weighted / log_loss
#                         probability-based metrics (need predict_proba,
#                         computed only when available) -- accuracy and
#                         F1 only look at the final hard label; these
#                         look at how confident and well-calibrated the
#                         model's ranking of the three classes actually
#                         is, which a hard-label metric can hide entirely.
# =============================================================================

def _extract_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    proba_classes: Optional[List[str]],
) -> Dict[str, float]:

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )

    metrics: Dict[str, float] = {

        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),

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

    for cls in CLASS_ORDER:

        cls_report = report.get(cls, {})

        metrics[f"{cls}_precision"] = cls_report.get("precision", 0.0)
        metrics[f"{cls}_recall"] = cls_report.get("recall", 0.0)
        metrics[f"{cls}_f1"] = cls_report.get("f1-score", 0.0)

    metrics["roc_auc_ovr_weighted"] = None
    metrics["log_loss"] = None

    if y_proba is not None and proba_classes is not None:

        try:
            sorted_classes = sorted(CLASS_ORDER)

            column_for_class = {
                cls: proba_classes.index(cls)
                for cls in sorted_classes
                if cls in proba_classes
            }

            if len(column_for_class) == len(sorted_classes):

                ordered_proba = y_proba[
                    :, [column_for_class[cls] for cls in sorted_classes]
                ]

                metrics["roc_auc_ovr_weighted"] = float(
                    roc_auc_score(
                        y_true,
                        ordered_proba,
                        labels=sorted_classes,
                        multi_class="ovr",
                        average="weighted",
                    )
                )

                metrics["log_loss"] = float(
                    log_loss(
                        y_true,
                        ordered_proba,
                        labels=sorted_classes,
                    )
                )

        except Exception as exc:
            print(
                f"    Could not compute probability-based metrics: "
                f"{type(exc).__name__}: {exc}"
            )

    return metrics


# =============================================================================
# CLASS DISTRIBUTION
# =============================================================================

def _plot_class_distribution(
    y_test: np.ndarray,
    output_path: Path,
) -> None:

    counts = (
        pd.Series(y_test)
        .value_counts()
        .reindex(CLASS_ORDER, fill_value=0)
    )

    plt.figure(figsize=(8, 5))

    counts.plot(
        kind="bar",
        color=["#1f77b4", "#ff7f0e", "#2ca02c"],
    )

    plt.title("Test set class distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of samples")
    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# CONFUSION MATRIX
# =============================================================================

def _plot_confusion_matrix(
    y_true: np.ndarray,
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

    plt.title(f"Confusion matrix - {model_name}")
    plt.colorbar()

    tick_marks = np.arange(len(CLASS_ORDER))

    plt.xticks(tick_marks, CLASS_ORDER, rotation=45)
    plt.yticks(tick_marks, CLASS_ORDER)

    threshold = cm.max() / 2.0

    for i, j in np.ndindex(cm.shape):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > threshold else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# GLOBAL METRICS COMPARISON
# =============================================================================

def _plot_global_metrics_comparison(
    metrics_df: pd.DataFrame,
    output_path: Path,
) -> None:

    metrics_to_plot = [
        "accuracy",
        "balanced_accuracy",
        "f1_weighted",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
    ]

    plot_df = metrics_df[
        ["model"] + metrics_to_plot
    ].set_index("model")

    plot_df.plot(kind="bar", figsize=(11, 6), rot=0)

    plt.title("Global metrics comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)

    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# F1 BY CLASS
# =============================================================================

def _plot_f1_by_class(
    metrics_df: pd.DataFrame,
    output_path: Path,
) -> None:

    class_cols = [f"{c}_f1" for c in CLASS_ORDER]

    plot_df = metrics_df[
        ["model"] + class_cols
    ].set_index("model")

    plot_df.plot(kind="bar", figsize=(9, 6), rot=0)

    plt.title("F1 score by class")
    plt.ylabel("F1")
    plt.ylim(0, 1)

    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# KEY-CLASS METRICS
# =============================================================================

def _plot_key_class_metrics(
    metrics_df: pd.DataFrame,
    output_path: Path,
) -> None:

    key_cols = [
        f"{KEY_CLASS}_precision",
        f"{KEY_CLASS}_recall",
        f"{KEY_CLASS}_f1",
    ]

    plot_df = metrics_df[
        ["model"] + key_cols
    ].set_index("model")

    plot_df.plot(kind="bar", figsize=(9, 6), rot=0)

    plt.title(f"'{KEY_CLASS}' class metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)

    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# ROC-AUC COMPARISON
# =============================================================================

def _plot_roc_auc_comparison(
    metrics_df: pd.DataFrame,
    output_path: Path,
) -> None:

    plot_df = metrics_df[
        ["model", "roc_auc_ovr_weighted"]
    ].dropna().set_index("model")

    if plot_df.empty:
        return

    plot_df.plot(
        kind="bar",
        figsize=(8, 6),
        rot=0,
        legend=False,
    )

    plt.title("ROC-AUC (one-vs-rest, weighted) comparison")
    plt.ylabel("ROC-AUC")
    plt.ylim(0, 1)

    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# FEATURE IMPORTANCE COMPARISON
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

    ranking = combined.mean(axis=1).sort_values(ascending=False)
    top_features = ranking.head(top_n).index

    plot_df = combined.loc[top_features]

    plot_df.plot(kind="barh", figsize=(10, 8))
    plt.gca().invert_yaxis()

    plt.title(f"Feature importance comparison (top {len(top_features)})")
    plt.xlabel("Importance")

    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# EVALUATE ONE MODEL
# =============================================================================

def _evaluate_model(
    model_key: str,
    model: Any,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    numeric_class_order: Optional[List[str]] = None,
) -> Tuple[Dict[str, float], np.ndarray]:

    print(f"\nEvaluating {model_name}...")

    X_test_model = _prepare_model_input(
        model_key=model_key,
        model=model,
        X_test=X_test,
    )

    y_pred_raw = model.predict(X_test_model)

    if hasattr(y_pred_raw, "to_numpy"):
        y_pred_raw = y_pred_raw.to_numpy()

    y_pred = _normalize_predictions(
        y_pred_raw,
        model_name=model_name,
        numeric_class_order=numeric_class_order,
    )

    y_true = np.asarray(y_test).ravel().astype(str)

    print("  Prediction labels:", np.unique(y_pred))
    print("  True labels:", np.unique(y_true))

    y_proba = None
    proba_classes: Optional[List[str]] = None

    if hasattr(model, "predict_proba"):

        try:
            y_proba = model.predict_proba(X_test_model)

            raw_classes = getattr(model, "classes_", None)

            if raw_classes is not None:

                proba_classes = list(
                    _normalize_predictions(
                        np.asarray(raw_classes),
                        model_name=f"{model_name} classes_",
                        numeric_class_order=numeric_class_order,
                    )
                )

        except Exception as exc:
            print(
                f"    predict_proba unavailable: "
                f"{type(exc).__name__}: {exc}"
            )

    metrics = _extract_metrics(
        y_true,
        y_pred,
        y_proba,
        proba_classes,
    )

    print(f"  Accuracy           = {metrics['accuracy']:.4f}")
    print(f"  Balanced accuracy  = {metrics['balanced_accuracy']:.4f}")
    print(f"  F1 (weighted)      = {metrics['f1_weighted']:.4f}")
    print(f"  F1 (macro)         = {metrics['f1_macro']:.4f}")
    print(f"  Precision (weighted) = {metrics['precision_weighted']:.4f}")
    print(f"  Recall (weighted)    = {metrics['recall_weighted']:.4f}")

    if metrics["roc_auc_ovr_weighted"] is not None:
        print(f"  ROC-AUC (OvR, weighted) = {metrics['roc_auc_ovr_weighted']:.4f}")
        print(f"  Log loss                = {metrics['log_loss']:.4f}")

    return metrics, y_pred


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("\n" + "=" * 80)
    print("P4 FINAL RESULTS - CLAN PERFORMANCE CLASSIFICATION")
    print("=" * 80 + "\n")

    # -------------------------------------------------------------------------
    # [1/6]
    # -------------------------------------------------------------------------

    print("[1/6] Validating MLflow Run ID configuration...")

    _validate_run_configuration()

    print("\u2713 Run configuration valid\n")

    for model_name, run_id in MODEL_RUN_IDS.items():
        if run_id is None:
            print(f"  {model_name}: SKIPPED (no Run ID configured)")
        else:
            print(f"  {model_name}: {run_id}")

    print()

    # -------------------------------------------------------------------------
    # [2/6]
    # -------------------------------------------------------------------------

    print("[2/6] Configuring MLflow...")

    configure_tracking()

    print("\u2713 MLflow configured\n")

    # -------------------------------------------------------------------------
    # [3/6]
    # -------------------------------------------------------------------------

    print("[3/6] Loading dataset...")

    data = _load_dataset(DATASET_PATH)

    print(f"Dataset shape: {data.shape}")

    X, y = _prepare_features(data)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of available features: {X.shape[1]}")
    print(f"Class counts (full dataset):\n{y.value_counts().reindex(CLASS_ORDER)}")

    del data
    gc.collect()

    test_indices = _get_test_indices(
        n_samples=len(X),
        test_size=DEFAULT_TEST_SIZE,
        random_state=DEFAULT_RANDOM_STATE,
    )

    X_test = X.iloc[test_indices].copy()
    y_test = y.iloc[test_indices].copy()

    feature_names = list(X_test.columns)

    print(f"\nTest shape: {X_test.shape}")
    print(f"Test target size: {len(y_test)}\n")

    del X
    del y
    del test_indices

    gc.collect()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    _plot_class_distribution(
        np.asarray(y_test),
        RESULTS_DIR / "01_class_distribution.png",
    )

    # -------------------------------------------------------------------------
    # [4/6]
    # -------------------------------------------------------------------------

    print("[4/6] Evaluating models...\n")

    all_metrics = []

    feature_importance_records: Dict[str, pd.Series] = {}

    # Models that were fit on LabelEncoder-encoded integers, per this
    # project's actual training pipeline (confirmed -- see
    # LABEL_ENCODER_CLASS_ORDER's docstring above). random_forest is
    # deliberately absent: it predicts strings natively and never uses
    # numeric_class_order at all.
    LABEL_ENCODED_MODELS = {"xgboost", "mlp"}

    for model_key, run_id in MODEL_RUN_IDS.items():

        if run_id is None:
            print(f"Skipping {model_key}: no Run ID configured.")
            continue

        model = None
        y_pred = None

        print(f"\nLoading {model_key}...")

        try:
            model = _load_model_from_run(
                run_id=run_id,
                model_key=model_key,
            )

            print(f"\u2713 {model_key} loaded")

        except Exception as exc:

            print(f"\nWARNING: Could not load {model_key}.")
            print(f"  Model: {model_key}")
            print(f"  Run ID: {run_id}")
            print(
                f"  Load method attempted: MLflow 3 Logged Model "
                f"(models:/<model_id>), legacy runs:/, direct local "
                f"artifact inspection"
            )
            print(f"  Error: {exc}")

            continue

        importance = _extract_feature_importance(
            model=model,
            feature_names=feature_names,
        )

        if importance is not None:
            feature_importance_records[model_key] = importance

        numeric_class_order = None

        if model_key in LABEL_ENCODED_MODELS:

            numeric_class_order = _find_label_encoder_classes(run_id)

            if numeric_class_order is None:
                numeric_class_order = LABEL_ENCODER_CLASS_ORDER
                print(
                    f"    No logged label-encoder mapping found for "
                    f"{model_key} -- using the confirmed training-time "
                    f"mapping LABEL_ENCODER_CLASS_ORDER = "
                    f"{LABEL_ENCODER_CLASS_ORDER}"
                )

        try:
            metrics, y_pred = _evaluate_model(
                model_key=model_key,
                model=model,
                model_name=model_key,
                X_test=X_test,
                y_test=y_test,
                numeric_class_order=numeric_class_order,
            )

        except Exception as exc:

            print(f"\nWARNING: Could not evaluate {model_key}.")
            print(f"  Run ID: {run_id}")
            print(f"  Error: {exc}")

            del model
            model = None
            gc.collect()

            continue

        metrics["model"] = model_key
        metrics["run_id"] = run_id

        all_metrics.append(metrics)

        _plot_confusion_matrix(
            y_true=np.asarray(y_test).astype(str),
            y_pred=y_pred,
            model_name=model_key,
            output_path=RESULTS_DIR / f"02_confusion_matrix_{model_key}.png",
        )

        print(f"\u2713 {model_key} completed")

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

    print("\n[5/6] Building final comparison tables...")

    if not all_metrics:
        raise RuntimeError("No models were successfully evaluated.")

    metrics_df = pd.DataFrame(all_metrics)

    metrics_df = metrics_df.sort_values("f1_macro", ascending=False)

    csv_columns = (
        ["model", "accuracy", "balanced_accuracy", "f1_weighted", "f1_macro",
         "precision_weighted", "recall_weighted",
         "roc_auc_ovr_weighted", "log_loss"]
        + [f"{c}_{m}" for c in CLASS_ORDER for m in ("precision", "recall", "f1")]
        + ["run_id"]
    )

    final_csv_df = metrics_df[csv_columns].copy()

    final_csv_df.to_csv(
        RESULTS_DIR / "final_model_comparison.csv",
        index=False,
    )

    print("\n" + "=" * 80)
    print("FINAL MODEL COMPARISON")
    print("=" * 80 + "\n")

    print(
        final_csv_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()

    best_model_df = final_csv_df.head(1)

    best_model_df.to_csv(
        RESULTS_DIR / "best_model.csv",
        index=False,
    )

    _plot_global_metrics_comparison(
        metrics_df,
        RESULTS_DIR / "03_global_metrics_comparison.png",
    )

    _plot_f1_by_class(
        metrics_df,
        RESULTS_DIR / "04_f1_by_class.png",
    )

    _plot_key_class_metrics(
        metrics_df,
        RESULTS_DIR / f"05_{KEY_CLASS}_class_metrics.png",
    )

    _plot_roc_auc_comparison(
        metrics_df,
        RESULTS_DIR / "06_roc_auc_comparison.png",
    )

    if feature_importance_records:

        _plot_feature_importance_comparison(
            feature_importance_records,
            RESULTS_DIR / "07_feature_importance_comparison.png",
        )

        print(
            "Feature importance (random_forest / xgboost) -- top "
            f"{FEATURE_IMPORTANCE_TOP_N} saved to "
            "07_feature_importance_comparison.png"
        )

        print(
            "(mlp has no direct feature_importances_/coef_ equivalent -- "
            "skipped from this chart, not an error)\n"
        )

    summary = {

        "dataset": str(DATASET_PATH),
        "target": TARGET_COLUMN,
        "classes": CLASS_ORDER,
        "key_class": KEY_CLASS,

        "test_size": DEFAULT_TEST_SIZE,
        "random_state": DEFAULT_RANDOM_STATE,

        "models": MODEL_RUN_IDS,

        "results": metrics_df[csv_columns].to_dict(orient="records"),
    }

    with open(
        RESULTS_DIR / "results_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(summary, f, indent=2, ensure_ascii=False)

    # =========================================================================
    # [6/6]
    # =========================================================================

    print("\n[6/6] Results generated successfully!")

    print("\nOutput directory:")
    print(RESULTS_DIR)

    print("\nGenerated files:")

    for file in sorted(RESULTS_DIR.rglob("*")):
        if file.is_file():
            print(f"  - {file.relative_to(RESULTS_DIR)}")

    print("\n" + "=" * 80)
    print("P4 RESULTS COMPLETED")
    print("=" * 80 + "\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
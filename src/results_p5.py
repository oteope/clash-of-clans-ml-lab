"""
results_p5.py

Retrieves the three already-trained P5 clustering models (K-Means, DBSCAN,
Agglomerative Clustering) from MLflow, recovers their cluster assignments,
and produces the comparison tables, cluster profiles, and visualizations
for the player-clustering analysis (~836,830 players).

This script does NOT fit any clustering model. It only loads what has
already been trained and logged, and analyzes it.

Three different "sizes" appear throughout this file and are never
conflated with each other:
  - training size:   however many rows the loaded model's own
                      .labels_ covers (expected to be all ~836,830
                      players for the final configuration of all three
                      algorithms -- this script does not choose or limit
                      this, it just reports whatever the model actually
                      has).
  - PCA size:         ALL valid players (~836,830, minus only rows with
                      missing feature values). Never sampled.
  - silhouette size:  silhouette_score is O(n^2); a 50k-player sample is
                      used to recompute it for a dataset this large,
                      purely as a computational necessity for THIS
                      metric. This has no bearing on training or PCA
                      size, and is always reported explicitly as a
                      sample.
"""

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
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

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
# Fill these in manually with the real Run IDs from your MLflow experiment.
# =============================================================================

KMEANS_RUN_ID = "0ba883acb3ed4c478916a1c36e03f5d5"
DBSCAN_RUN_ID = "0ca121fe6aa64342ab4630059c5a0bf9"
AGGLOMERATIVE_RUN_ID = None

RUN_IDS = {
    "kmeans": KMEANS_RUN_ID,
    "dbscan": DBSCAN_RUN_ID,
    "agglomerative": AGGLOMERATIVE_RUN_ID,
}

# What each algorithm's loaded object is expected to be. Used only for an
# explicit compatibility check/warning after loading -- not enforced as a
# hard failure, in case a legitimate subclass is used.
EXPECTED_MODEL_TYPES = {
    "kmeans": "KMeans",
    "dbscan": "DBSCAN",
    "agglomerative": "AgglomerativeClustering",
}

DISPLAY_NAMES = {
    "kmeans": "K-Means",
    "dbscan": "DBSCAN",
    "agglomerative": "Agglomerative",
}

# DBSCAN's noise label. Never treated as a real cluster anywhere below.
NOISE_LABEL = -1

RESULTS_DIR = ROOT_DIR / "src" / "results" / "P5"

DATASET_PATH = (
    ROOT_DIR
    / "data"
    / "datasets"
    / "player_clustering.parquet"
)

# Column names that, if present and numeric, are still identifiers rather
# than clustering features (a player tag stored as an integer would
# otherwise silently become "just another feature").
IDENTIFIER_NAME_HINTS = (
    "tag",
    "player_id",
    "clan_id",
)

PCA_COMPONENTS = 2

DEFAULT_RANDOM_STATE = 42

# Silhouette is O(n^2) in time and memory -- at ~836,830 rows a full
# computation is not just slow, it's not going to finish. This is a
# computational sampling limit for THIS metric only. It never limits
# training data, PCA data, cluster labels, cluster sizes, or cluster
# profiles, all of which use every row the loaded model actually covers.
SILHOUETTE_SAMPLE_SIZE = 50_000

# Explicit, not derived from dict order.
CLUSTER_SIZE_FILE_NUMBERS = {
    "kmeans": "02",
    "dbscan": "03",
    "agglomerative": "04",
}

PCA_FILE_NUMBERS = {
    "kmeans": "05",
    "dbscan": "06",
    "agglomerative": "07",
}


# =============================================================================
# VALIDATE RUN CONFIGURATION
# =============================================================================

def validate_run_configuration() -> None:

    valid_algorithms = set(RUN_IDS.keys())

    for algorithm, run_id in RUN_IDS.items():

        if algorithm not in valid_algorithms:
            raise ValueError(f"Invalid algorithm '{algorithm}' in RUN_IDS.")

        if run_id is None:
            continue

        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError(
                f"Run ID for '{algorithm}' must be a non-empty string or None."
            )


# =============================================================================
# DATASET LOADING
# =============================================================================

def load_dataset(path: Path) -> pd.DataFrame:
    """Load the clustering dataset as-is. No target column exists, so
    nothing is split off here -- feature selection happens per-model in
    align_features_for_model(), driven by what each loaded model actually
    expects."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = pd.read_parquet(path)

    print(f"Dataset shape: {data.shape[0]:,} rows x {data.shape[1]} columns")
    print(f"Columns found: {list(data.columns)}")

    return data


# =============================================================================
# FEATURE PREPARATION
# =============================================================================

def prepare_common_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Build a numeric feature matrix from whatever columns actually exist in
    the dataset -- no column names are assumed in advance.

    - Any column whose name contains an identifier hint (tag/id-like) is
      dropped even if it's stored as a number: a numeric tag is not a
      clustering feature.
    - Everything else is coerced to numeric; columns that turn out to be
      entirely non-numeric (all-NaN after coercion) are dropped as a
      consequence of that coercion, not by name.
    - Boolean columns (including ones with missing values, which pandas
      stores as dtype "object" rather than "bool") are kept as 0/1
      features rather than being coerced away.
    - Everything is cast to float32 (not float64) specifically because
      this dataset is large (~836,830 rows) and StandardScaler/PCA will
      be run over the entire thing -- halving the matrix's memory
      footprint matters at this size.

    This is deliberately a "maximal reasonable" feature set. Which of
    these columns any *specific* model actually uses is decided later, in
    align_features_for_model(), from that model's own signature -- not
    guessed here.
    """

    X = data.copy()

    identifier_columns = [
        column
        for column in X.columns
        if any(hint in column.lower() for hint in IDENTIFIER_NAME_HINTS)
    ]

    if identifier_columns:
        print(f"Excluding identifier-like columns: {identifier_columns}")
        X = X.drop(columns=identifier_columns)

    bool_like_columns = []

    for column in X.columns:

        if X[column].dtype == bool:
            bool_like_columns.append(column)
            continue

        if X[column].dtype == object:
            non_null = X[column].dropna()
            if len(non_null) > 0 and non_null.isin([True, False]).all():
                bool_like_columns.append(column)

    X = X.apply(pd.to_numeric, errors="coerce")

    for column in bool_like_columns:
        X[column] = X[column].fillna(False).astype(np.int8)

    all_nan_columns = X.columns[X.isna().all()].tolist()

    if all_nan_columns:
        dropped_and_not_bool = [c for c in all_nan_columns if c not in bool_like_columns]
        if dropped_and_not_bool:
            print(
                f"Dropping non-numeric columns (could not be used as "
                f"features): {dropped_and_not_bool}"
            )
        X = X.drop(columns=all_nan_columns)

    remaining_nan_counts = X.isna().sum()
    columns_with_gaps = remaining_nan_counts[remaining_nan_counts > 0]

    if not columns_with_gaps.empty:
        print(
            "WARNING: these numeric columns still have missing values -- "
            "rows with any of them are dropped only where strictly "
            "necessary (per-algorithm and for PCA), never dropped from "
            "the common matrix itself:"
        )
        print(columns_with_gaps.to_string())

    for column in X.columns:
        if pd.api.types.is_float_dtype(X[column]) or pd.api.types.is_integer_dtype(X[column]):
            X[column] = X[column].astype(np.float32)

    return X


# =============================================================================
# PER-MODEL FEATURE ALIGNMENT
# =============================================================================

def align_features_for_model(
    model: Any,
    X_common: pd.DataFrame,
    algorithm_name: str,
) -> pd.DataFrame:
    """
    Select/order the exact columns a specific loaded model expects, using
    the model's own signature -- never assumed to be identical to
    X_common just because it usually will be.

    feature_names_in_ is preferred (catches column-order or column-set
    differences); n_features_in_ is used as a count-only fallback.
    Fails explicitly (raises) if the model's expected features can't be
    reconstructed from the dataset -- never silently falls back to the
    wrong columns.
    """

    feature_names = getattr(model, "feature_names_in_", None)

    if feature_names is not None:

        feature_names = list(feature_names)
        missing = [c for c in feature_names if c not in X_common.columns]

        if missing:
            raise RuntimeError(
                f"{algorithm_name}: model expects columns not present in "
                f"the prepared dataset: {missing}"
            )

        extra = [c for c in X_common.columns if c not in feature_names]

        if extra:
            print(
                f"    {algorithm_name}: dataset has {len(extra)} column(s) "
                f"this model wasn't fit on (excluded for this model): {extra}"
            )

        return X_common[feature_names]

    expected_count = getattr(model, "n_features_in_", None)

    if expected_count is not None and expected_count != X_common.shape[1]:
        raise RuntimeError(
            f"{algorithm_name}: feature count mismatch. Model expects "
            f"{expected_count}, prepared dataset has {X_common.shape[1]}. "
            f"The model doesn't expose feature_names_in_, so which "
            f"columns to drop can't be determined automatically."
        )

    return X_common


# =============================================================================
# MLflow 3 MODEL LOADING
# =============================================================================
#
# Same strategy validated in results_p2.py / results_p3.py / results_p4.py:
# mlflow.sklearn.log_model() does not write anything under the run's own
# classic artifact directory in current MLflow 3 -- runs:/<run_id>/model
# has nothing to find. run.outputs.model_outputs -> models:/<model_id> is
# the reliable path. Kept unchanged here (loading is not the concern this
# refactor is addressing).
#
# The loaded object is always the raw sklearn estimator, never a pyfunc
# wrapper -- this matters specifically because DBSCAN and
# AgglomerativeClustering have no predict() for new observations, so
# .labels_ on the raw object is the only way to recover their
# assignments at all.
# =============================================================================

def _file_uri_to_local_path(uri: str) -> Path:
    """Convert a file:// artifact URI (percent-encoded, e.g. spaces as
    %20) into a real local Path, correcting the Windows
    leading-slash-before-drive-letter case a bare urlparse().path leaves
    in place."""
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
    """Resolve a file:// artifact URI to a local path and print what's
    there, WITH FILE SIZES, before any load is attempted -- some
    exceptions (a bare MemoryError) print as an empty string otherwise."""
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
        print(f"      - {file_path.relative_to(local_root)}: {size_mb:,.1f} MB")


def load_model_from_run(run_id: str, algorithm_name: str) -> Any:
    """
    Load a clustering model logged with MLflow 3 via log_model_and_artifacts().

    Strategy, in order:
      1. run.outputs.model_outputs -> models:/<model_id>.
      2. Legacy runs:/<run_id>/model and runs:/<run_id>/modelo.
      3. Direct local artifact inspection (resolve the file:// URI, list
         what's there with sizes, load from whatever directory actually
         contains an MLmodel file).
      4. Raw pickle.load() on model.pkl next to that MLmodel file, as an
         absolute last resort.

    Every exception is printed as "ExceptionType: message", since some
    exceptions (a bare MemoryError) stringify to nothing.

    Returns the raw fitted estimator. Raises if every strategy fails.
    """

    client = MlflowClient()
    run = None

    # -------------------------------------------------------------------
    # Strategy 1: run -> LoggedModel(s) -> models:/<model_id>
    # -------------------------------------------------------------------

    try:
        run = client.get_run(run_id)
        model_outputs = run.outputs.model_outputs if run.outputs is not None else []

        print(f"    run.outputs.model_outputs: {len(model_outputs)} entry(ies)")

        for output in model_outputs:

            model_id = output.model_id
            logged_model = None

            try:
                logged_model = client.get_logged_model(model_id)
                print(
                    f"      - model_id={model_id} name={logged_model.name} "
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
                loaded_model = mlflow.sklearn.load_model(model_uri)
                print("    \u2713 Loaded via models:/<model_id>")
                return loaded_model
            except Exception as exc:
                print(f"    {model_uri} failed: {type(exc).__name__}: {exc}")

    except Exception as exc:
        print(f"    Could not inspect run outputs: {type(exc).__name__}: {exc}")

    # -------------------------------------------------------------------
    # Strategy 2: legacy runs:/ paths
    # -------------------------------------------------------------------

    for fallback_uri in (f"runs:/{run_id}/model", f"runs:/{run_id}/modelo"):
        try:
            print(f"    Trying fallback: {fallback_uri}")
            loaded_model = mlflow.sklearn.load_model(fallback_uri)
            print(f"    \u2713 Loaded using fallback: {fallback_uri}")
            return loaded_model
        except Exception as exc:
            print(f"    Fallback failed: {type(exc).__name__}: {exc}")

    # -------------------------------------------------------------------
    # Strategy 3: direct local artifact inspection
    # -------------------------------------------------------------------

    print("    Falling back to direct local artifact inspection...")

    candidate_locations: List[Tuple[str, str]] = []

    try:
        if run is None:
            run = client.get_run(run_id)

        candidate_locations.append(("run artifact root", run.info.artifact_uri))

        model_outputs = run.outputs.model_outputs if run.outputs is not None else []

        for output in model_outputs:
            try:
                logged_model = client.get_logged_model(output.model_id)
                candidate_locations.append(
                    (f"logged model {output.model_id}", logged_model.artifact_location)
                )
            except Exception:
                pass

    except Exception as exc:
        print(f"    Could not enumerate candidate locations: {type(exc).__name__}: {exc}")

    for label, uri in candidate_locations:

        try:
            local_root = _file_uri_to_local_path(uri)
        except Exception as exc:
            print(f"    Could not resolve {label} ({uri}): {type(exc).__name__}: {exc}")
            continue

        print(f"    Inspecting {label} on disk: {local_root}")

        if not local_root.exists():
            print("      (path does not exist)")
            continue

        found_files = sorted(
            str(p.relative_to(local_root)) for p in local_root.rglob("*") if p.is_file()
        )
        print(f"      Files found: {found_files if found_files else '(empty)'}")

        mlmodel_dir = _find_mlmodel_dir(local_root)

        if mlmodel_dir is None:
            continue

        print(f"    Found MLmodel at: {mlmodel_dir}")

        pkl_path = mlmodel_dir / "model.pkl"

        if pkl_path.exists():
            size_mb = pkl_path.stat().st_size / (1024 * 1024)
            print(f"    model.pkl size: {size_mb:,.1f} MB")

        try:
            loaded_model = mlflow.sklearn.load_model(str(mlmodel_dir))
            print(f"    \u2713 Loaded directly from local path: {mlmodel_dir}")
            return loaded_model
        except Exception as exc:
            print(f"    Native flavor load from local path failed: {type(exc).__name__}: {exc}")

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
                    print(f"    Raw pickle.load failed: {type(exc2).__name__}: {exc2}")
                    print("    Full traceback for this last attempt:")
                    print("    " + traceback.format_exc().replace("\n", "\n    "))

    raise RuntimeError(
        f"Could not load {algorithm_name} model from run {run_id}.\n"
        f"Tried: MLflow 3 Logged Model (models:/<model_id>), legacy "
        f"runs:/ paths, and direct local artifact inspection."
    )


def check_model_type(model: Any, algorithm: str) -> None:
    """Explicit compatibility check: warn (don't hard-fail) if the loaded
    object's class doesn't match what this algorithm slot expects -- e.g.
    catches a KMEANS_RUN_ID that actually points at a DBSCAN run."""

    actual_type = type(model).__name__
    expected_type = EXPECTED_MODEL_TYPES[algorithm]

    if actual_type != expected_type:
        print(
            f"    WARNING: expected a {expected_type} for '{algorithm}', "
            f"but the loaded object is a {actual_type}. Double-check "
            f"{algorithm.upper()}_RUN_ID points at the right run."
        )
    else:
        print(f"    Model type confirmed: {actual_type}")


# =============================================================================
# MLflow RUN METADATA
# =============================================================================

def _get_metric_ci(metrics: Dict[str, float], *candidates: str) -> Optional[float]:
    """Case-insensitive lookup of a metric by any of several plausible
    logged names -- this project's exact naming for these metrics isn't
    assumed."""

    lowered = {key.lower(): value for key, value in metrics.items()}

    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]

    return None


def get_run_metadata(run_id: str) -> Dict[str, Any]:
    """
    Pull everything logged for a run -- params, metrics, tags.

    Specifically also looks for a logged sample-size parameter alongside
    the silhouette score (e.g. "silhouette_sample_size",
    "eval_sample_size"), so that if the training/exploration run logged a
    silhouette computed on a sample, that context travels with the score
    rather than the score being reported as if it covered every player.
    """

    client = MlflowClient()
    run = client.get_run(run_id)

    metrics = dict(run.data.metrics)
    params = dict(run.data.params)
    tags = {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")}

    logged_silhouette_sample_size = _get_metric_ci(
        {**metrics, **{k: v for k, v in params.items() if _is_number(v)}},
        "silhouette_sample_size", "eval_sample_size", "silhouette_n_samples",
    )

    return {
        "run_id": run_id,
        "status": run.info.status,
        "params": params,
        "metrics": metrics,
        "tags": tags,
        "logged_silhouette_score": _get_metric_ci(
            metrics, "silhouette_score", "silhouette"
        ),
        "logged_silhouette_sample_size": logged_silhouette_sample_size,
        "logged_inertia": _get_metric_ci(metrics, "inertia"),
        "logged_n_clusters": _get_metric_ci(
            metrics, "n_clusters", "num_clusters", "number_of_clusters"
        ),
    }


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


# =============================================================================
# LABELS ARTIFACT LOOKUP
# =============================================================================

def find_labels_artifact(run_id: str, n_expected: int) -> Optional[np.ndarray]:
    """
    Look for a labels array explicitly logged as an artifact (a file whose
    name contains "label"), in case that's how cluster assignments were
    preserved. Best-effort -- tried before falling back to the model
    object's own .labels_/.predict().
    """

    client = MlflowClient()

    try:
        artifacts = client.list_artifacts(run_id)
    except Exception as exc:
        print(f"    Could not list artifacts for labels lookup: {type(exc).__name__}: {exc}")
        return None

    candidates = [
        a for a in artifacts
        if "label" in a.path.lower() and a.path.lower().endswith((".csv", ".json", ".parquet"))
    ]

    if not candidates:
        return None

    for candidate in candidates:

        print(f"    Found possible labels artifact: {candidate.path}")

        try:
            local_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=candidate.path
            )

            if local_path.endswith(".csv"):
                loaded = pd.read_csv(local_path)
            elif local_path.endswith(".parquet"):
                loaded = pd.read_parquet(local_path)
            else:
                with open(local_path) as f:
                    loaded = pd.Series(json.load(f))

            if isinstance(loaded, pd.DataFrame):
                label_col = next(
                    (c for c in loaded.columns if "label" in c.lower() or c.lower() == "cluster"),
                    loaded.columns[-1],
                )
                labels = loaded[label_col].to_numpy()
            else:
                labels = np.asarray(loaded)

            if len(labels) == n_expected:
                print(f"    \u2713 Using labels artifact: {candidate.path}")
                return labels

            print(
                f"    Labels artifact has {len(labels):,} entries, "
                f"expected {n_expected:,} -- skipping."
            )

        except Exception as exc:
            print(f"    Could not load labels artifact {candidate.path}: {type(exc).__name__}: {exc}")

    return None


# =============================================================================
# CLUSTER LABEL RECOVERY
# =============================================================================

def get_cluster_labels(
    model: Any,
    X_aligned: pd.DataFrame,
    run_id: str,
    algorithm_name: str,
) -> np.ndarray:
    """
    Recover the cluster assignment for every row in X_aligned, in order
    of decreasing reliability:

      1. A labels artifact explicitly logged with the run, if found and
         its length matches.
      2. model.labels_, if present and its length matches len(X_aligned).
         This is fit-time truth for all three algorithms (confirmed
         empirically: labels_ survives a full mlflow.sklearn.log_model /
         load_model round trip for KMeans, DBSCAN, AND
         AgglomerativeClustering).
      3. model.predict(X_aligned), only if the model implements it --
         DBSCAN and AgglomerativeClustering do not.

    If none of these work, raises with an explicit description of what's
    missing and the minimal training-side fix, rather than guessing.
    """

    n_expected = len(X_aligned)

    artifact_labels = find_labels_artifact(run_id, n_expected)
    if artifact_labels is not None:
        return artifact_labels

    labels_attr = getattr(model, "labels_", None)

    if labels_attr is not None and len(labels_attr) == n_expected:
        print(
            f"    Using model.labels_ ({algorithm_name}, fit-time "
            f"assignment) -- covers all {n_expected:,} rows"
        )
        return np.asarray(labels_attr)

    if labels_attr is not None and len(labels_attr) != n_expected:
        print(
            f"    model.labels_ exists but has {len(labels_attr):,} entries, "
            f"the prepared dataset has {n_expected:,} -- likely not the "
            f"same data the model was fit on. Not using it."
        )

    if hasattr(model, "predict"):
        try:
            print(f"    Trying model.predict() ({algorithm_name})")
            predicted = model.predict(X_aligned)
            return np.asarray(predicted)
        except Exception as exc:
            print(f"    model.predict() failed: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        f"Could not recover cluster labels for {algorithm_name} "
        f"(run {run_id}).\n"
        f"model.labels_ is {'missing' if labels_attr is None else 'the wrong length'}, "
        f"and this algorithm "
        f"{'has no predict() method (this is normal for ' + algorithm_name + ' in scikit-learn)' if not hasattr(model, 'predict') else 'predict() also failed'}.\n"
        f"No labels artifact was found for this run either.\n"
        f"Minimal fix on the training side: after fitting, log the "
        f"labels explicitly, e.g.\n"
        f"    pd.DataFrame({{'cluster': model.labels_}}).to_csv('labels.csv', index=False)\n"
        f"    mlflow.log_artifact('labels.csv')\n"
        f"inside the same run that logs the model."
    )


# =============================================================================
# SILHOUETTE SCORE (RECOMPUTED)
# =============================================================================

def compute_silhouette(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    algorithm_name: str,
    sample_size: int = SILHOUETTE_SAMPLE_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Dict[str, Any]:
    """
    Recompute silhouette score defensively:
      - DBSCAN noise points (label == NOISE_LABEL) are excluded entirely,
        not treated as their own cluster.
      - Requires at least 2 distinct labels among the remaining points,
        and fewer labels than remaining points.
      - If more than `sample_size` valid points remain, sklearn's own
        sample_size= parameter is used to compute silhouette on a random
        subset (silhouette_score is O(n^2); a full computation over
        ~836,830 rows is not a "slow" version of this metric, it does
        not complete). This is a computation-only limit and never
        affects labels, PCA, cluster sizes, or cluster profiles.

    Returns a dict: {"score": float or None, "n_samples_used": int,
    "was_sampled": bool, "reason_skipped": str or None} -- never just a
    bare float, so the sample size actually used is always visible to
    the caller rather than implied.
    """

    labels = np.asarray(labels)
    mask = labels != NOISE_LABEL

    n_excluded = int((~mask).sum())

    if n_excluded > 0:
        print(
            f"    {algorithm_name}: excluding {n_excluded:,} noise "
            f"point(s) (label == {NOISE_LABEL}) from silhouette"
        )

    X_valid = X_scaled[mask]
    labels_valid = labels[mask]

    unique_labels = np.unique(labels_valid)

    if len(unique_labels) < 2:
        reason = (
            f"only {len(unique_labels)} valid cluster(s) after excluding "
            f"noise -- silhouette needs at least 2"
        )
        print(f"    {algorithm_name}: {reason}, skipping.")
        return {"score": None, "n_samples_used": 0, "was_sampled": False, "reason_skipped": reason}

    if len(unique_labels) >= len(labels_valid):
        reason = "number of clusters >= number of valid samples"
        print(f"    {algorithm_name}: {reason}, skipping.")
        return {"score": None, "n_samples_used": 0, "was_sampled": False, "reason_skipped": reason}

    n_valid = len(labels_valid)
    use_sample = n_valid > sample_size

    try:
        score = float(
            silhouette_score(
                X_valid,
                labels_valid,
                sample_size=sample_size if use_sample else None,
                random_state=random_state,
            )
        )
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        print(f"    {algorithm_name}: silhouette computation failed: {reason}")
        return {"score": None, "n_samples_used": 0, "was_sampled": False, "reason_skipped": reason}

    n_used = min(sample_size, n_valid) if use_sample else n_valid

    print(
        f"    {algorithm_name}: silhouette (recomputed) = {score:.4f} "
        f"({'sampled ' + format(n_used, ',') + ' of ' + format(n_valid, ',') if use_sample else format(n_used, ',') + ' (all valid)'} points)"
    )

    return {
        "score": score,
        "n_samples_used": n_used,
        "was_sampled": use_sample,
        "reason_skipped": None,
    }


# =============================================================================
# CLUSTER SIZES
# =============================================================================

def calculate_cluster_sizes(labels: np.ndarray) -> pd.Series:
    """Size of every cluster, DBSCAN noise included as its own explicit
    row (never merged into a real cluster's count)."""
    return pd.Series(labels).value_counts().sort_index()


# =============================================================================
# CLUSTER PROFILES
# =============================================================================

def calculate_cluster_profiles(
    X_aligned: pd.DataFrame,
    labels: np.ndarray,
    exclude_noise: bool = True,
) -> pd.DataFrame:
    """Mean of every feature, per cluster, over ALL rows in X_aligned
    (no sampling). Noise (-1) is excluded by default."""

    working = X_aligned.copy()
    working["cluster"] = labels

    if exclude_noise:
        working = working[working["cluster"] != NOISE_LABEL]

    return working.groupby("cluster").mean().sort_index()


# =============================================================================
# FEATURE DIFFERENCES FROM GLOBAL MEAN
# =============================================================================

def calculate_feature_differences(
    X_aligned: pd.DataFrame,
    labels: np.ndarray,
    exclude_noise: bool = True,
) -> pd.DataFrame:
    """
    For every (cluster, feature) pair: the cluster's mean, the global
    mean, and the difference -- over ALL rows, purely descriptive
    statistics (not a supervised feature_importances_ equivalent, since
    these are unsupervised models).
    """

    working = X_aligned.copy()
    working["cluster"] = labels

    if exclude_noise:
        working = working[working["cluster"] != NOISE_LABEL]

    feature_columns = [c for c in working.columns if c != "cluster"]

    global_means = working[feature_columns].mean()

    rows = []

    for cluster_id, group in working.groupby("cluster"):

        cluster_means = group[feature_columns].mean()
        difference = cluster_means - global_means

        for feature in feature_columns:
            rows.append({
                "cluster": cluster_id,
                "feature": feature,
                "cluster_mean": cluster_means[feature],
                "global_mean": global_means[feature],
                "difference_from_global": difference[feature],
            })

    result = pd.DataFrame(rows)
    result["abs_difference"] = result["difference_from_global"].abs()
    result = result.sort_values(
        ["cluster", "abs_difference"], ascending=[True, False]
    ).drop(columns="abs_difference")

    return result.reset_index(drop=True)


# =============================================================================
# PLOTS
# =============================================================================

def plot_silhouette_comparison(comparison_df: pd.DataFrame, output_path: Path) -> None:

    plot_df = comparison_df[["algorithm", "silhouette_score"]].dropna().set_index("algorithm")

    if plot_df.empty:
        print("    No silhouette scores available -- skipping 01_silhouette_comparison.png")
        return

    plt.figure(figsize=(8, 6))
    plot_df["silhouette_score"].plot(kind="bar", color="#4c72b0")
    plt.title("Silhouette score comparison")
    plt.ylabel("Silhouette score")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_cluster_sizes(sizes: pd.Series, algorithm_name: str, output_path: Path) -> None:

    labels_display = ["Noise" if idx == NOISE_LABEL else str(idx) for idx in sizes.index]
    colors = ["#999999" if idx == NOISE_LABEL else "#4c72b0" for idx in sizes.index]

    plt.figure(figsize=(8, 5))
    plt.bar(labels_display, sizes.values, color=colors)
    plt.title(f"Cluster sizes - {DISPLAY_NAMES.get(algorithm_name, algorithm_name)}")
    plt.xlabel("Cluster")
    plt.ylabel("Number of players")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_pca(
    pca_coords: np.ndarray,
    labels: np.ndarray,
    algorithm_name: str,
    n_players: int,
    output_path: Path,
) -> None:
    """PCA scatter over every valid player. Markers are small and
    semi-transparent so ~836,830 overlapping points remain readable as a
    density rather than a solid blob."""

    plt.figure(figsize=(10, 9))

    unique_labels = sorted(set(labels))

    for label in unique_labels:

        mask = labels == label
        is_noise = label == NOISE_LABEL
        display_label = "Noise" if is_noise else f"Cluster {label}"

        plt.scatter(
            pca_coords[mask, 0],
            pca_coords[mask, 1],
            label=display_label,
            alpha=0.15 if not is_noise else 0.3,
            s=4,
            color="#cccccc" if is_noise else None,
            marker="x" if is_noise else "o",
        )

    plt.title(
        f"PCA projection (2D) - {DISPLAY_NAMES.get(algorithm_name, algorithm_name)}\n"
        f"{n_players:,} players"
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    legend = plt.legend(loc="best", fontsize=8, markerscale=3)

    # Cosmetic only (fully-opaque legend markers even though the actual
    # points are semi-transparent) -- the attribute holding these was
    # renamed between matplotlib versions (legendHandles -> 3.7+
    # legend_handles), so this is looked up defensively rather than
    # assumed, since this plot must not fail to save over a naming
    # difference.
    legend_handles = getattr(legend, "legend_handles", None)
    if legend_handles is None:
        legend_handles = getattr(legend, "legendHandles", None)
    if legend_handles is not None:
        for handle in legend_handles:
            try:
                handle.set_alpha(1.0)
            except Exception:
                pass

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("\n" + "=" * 80)
    print("P5 FINAL RESULTS - PLAYER CLUSTERING")
    print("=" * 80 + "\n")

    # -------------------------------------------------------------------------
    # [1/6]
    # -------------------------------------------------------------------------

    print("[1/6] Validating MLflow Run ID configuration...")
    validate_run_configuration()
    print("\u2713 Run configuration valid\n")

    for algorithm, run_id in RUN_IDS.items():
        if run_id is None:
            print(f"  {algorithm}: SKIPPED (no Run ID configured)")
        else:
            print(f"  {algorithm}: {run_id}")

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

    data = load_dataset(DATASET_PATH)
    X_common = prepare_common_features(data)

    print(f"\nPrepared feature matrix: {X_common.shape[0]:,} rows x {X_common.shape[1]} features")
    print(f"Features: {list(X_common.columns)}\n")

    del data
    gc.collect()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Shared PCA: ALL valid players, never sampled. Rows are dropped here
    # ONLY where a feature value is actually missing -- not for size.
    # -------------------------------------------------------------------------

    X_for_pca = X_common.dropna()

    n_dropped_for_pca = len(X_common) - len(X_for_pca)
    if n_dropped_for_pca > 0:
        print(
            f"PCA: dropping {n_dropped_for_pca:,} row(s) with at least one "
            f"missing feature value (strictly necessary for PCA/scaling; "
            f"not a sampling choice)."
        )

    scaler = StandardScaler()
    X_scaled_for_pca = scaler.fit_transform(X_for_pca).astype(np.float32)

    pca = PCA(n_components=PCA_COMPONENTS, random_state=DEFAULT_RANDOM_STATE)
    pca_coords_shared = pca.fit_transform(X_scaled_for_pca)

    explained_variance = pca.explained_variance_ratio_.tolist()

    # Captured as plain values / lightweight objects now, deliberately:
    # everything the per-algorithm loop and the final summary still need
    # from X_for_pca is its row index, its column list, and these two
    # numbers -- never the actual (rows x features) data again after this
    # point. Keeping the full DataFrame (and the full scaled copy) alive
    # for the rest of the run would mean an extra ~100MB+ (at the real
    # ~836,830-row scale) held for no reason for the whole per-algorithm
    # loop below.
    n_players_for_pca = len(X_for_pca)
    n_features_for_pca = X_for_pca.shape[1]
    X_for_pca_index = X_for_pca.index
    X_for_pca_columns = list(X_for_pca.columns)

    del X_for_pca, X_scaled_for_pca
    gc.collect()

    print(
        f"\nShared PCA fit on ALL {n_players_for_pca:,} valid players "
        f"({n_features_for_pca} features)"
    )
    print(
        f"Explained variance ratio: "
        f"PC1={explained_variance[0]:.4f}, PC2={explained_variance[1]:.4f}\n"
    )

    if not len(pca_coords_shared) == n_players_for_pca:
        raise RuntimeError(
            "Internal error: PCA coordinate count does not match the "
            "number of valid players used to fit it."
        )

    # -------------------------------------------------------------------------
    # [4/6]
    # -------------------------------------------------------------------------

    print("[4/6] Evaluating clustering models...\n")

    comparison_rows = []
    run_metadata_by_algorithm: Dict[str, Dict[str, Any]] = {}
    final_summary_by_algorithm: Dict[str, Dict[str, Any]] = {}

    for algorithm, run_id in RUN_IDS.items():

        if run_id is None:
            print(f"Skipping {algorithm}: no Run ID configured.")
            continue

        model = None

        print(f"\nLoading {algorithm}...")

        try:
            model = load_model_from_run(run_id, algorithm)
            print(f"\u2713 {algorithm} loaded ({type(model).__name__})")
            check_model_type(model, algorithm)
        except Exception as exc:
            print(f"\nWARNING: Could not load {algorithm}.")
            print(f"  Run ID: {run_id}")
            print(f"  Error: {exc}")
            continue

        try:
            X_aligned = align_features_for_model(model, X_common, algorithm)
            X_aligned_complete = X_aligned.dropna()

            if len(X_aligned_complete) != len(X_aligned):
                print(
                    f"    Dropping {len(X_aligned) - len(X_aligned_complete):,} "
                    f"row(s) with missing values in this model's features "
                    f"before recovering labels."
                )

            labels = get_cluster_labels(
                model=model,
                X_aligned=X_aligned_complete,
                run_id=run_id,
                algorithm_name=algorithm,
            )

            if not len(labels) == len(X_aligned_complete):
                raise RuntimeError(
                    f"Internal error: recovered {len(labels)} labels for "
                    f"{len(X_aligned_complete)} rows."
                )

        except Exception as exc:
            print(f"\nWARNING: Could not recover cluster labels for {algorithm}.")
            print(f"  Run ID: {run_id}")
            print(f"  Error: {exc}")

            del model
            gc.collect()
            continue

        run_metadata = get_run_metadata(run_id)
        run_metadata_by_algorithm[algorithm] = run_metadata

        real_clusters = sorted(c for c in set(labels) if c != NOISE_LABEL)
        n_noise = int(np.sum(np.asarray(labels) == NOISE_LABEL))
        n_samples = len(labels)
        noise_pct = (n_noise / n_samples * 100.0) if n_samples else 0.0

        print(f"\n  {algorithm}: cluster assignments cover {n_samples:,} players")
        if n_samples > 500_000:
            print(
                f"    (this matches the full-dataset training size, not "
                f"a sample -- confirmed by the length of model.labels_ "
                f"itself, not assumed)"
            )

        # ---------------------------------------------------------------
        # PCA coordinates for this algorithm: reuse the shared embedding
        # directly when this algorithm used the same feature set as the
        # shared PCA fit (the expected case); otherwise fit a separate
        # scaler+PCA scoped to this algorithm's own features, and say so.
        # ---------------------------------------------------------------

        same_features_as_shared = list(X_aligned_complete.columns) == X_for_pca_columns
        rows_match_shared = same_features_as_shared and X_aligned_complete.index.isin(X_for_pca_index).all()

        if rows_match_shared:
            algorithm_pca_coords = pca_coords_shared[
                X_for_pca_index.get_indexer(X_aligned_complete.index)
            ]
            X_scaled_aligned = scaler.transform(X_aligned_complete).astype(np.float32)
        else:
            print(
                f"    {algorithm}: feature set differs from the shared "
                f"PCA fit -- fitting a separate scaler+PCA for this "
                f"algorithm instead of reusing the shared one."
            )
            algorithm_scaler = StandardScaler()
            X_scaled_aligned = algorithm_scaler.fit_transform(X_aligned_complete).astype(np.float32)
            algorithm_pca_coords = PCA(
                n_components=PCA_COMPONENTS, random_state=DEFAULT_RANDOM_STATE
            ).fit_transform(X_scaled_aligned)

        if not len(algorithm_pca_coords) == len(labels):
            raise RuntimeError(
                f"Internal error: {algorithm} has "
                f"{len(algorithm_pca_coords)} PCA coordinates but "
                f"{len(labels)} labels."
            )

        # ---------------------------------------------------------------
        # Silhouette: MLflow-logged value preferred for the comparison
        # table; always ALSO recomputed (on a sample, at this scale) so
        # there's an independent check, with both clearly labeled.
        # ---------------------------------------------------------------

        silhouette_result = compute_silhouette(
            X_scaled_aligned, np.asarray(labels), algorithm
        )

        logged_silhouette = run_metadata["logged_silhouette_score"]
        logged_sample_size = run_metadata["logged_silhouette_sample_size"]

        if logged_silhouette is not None:
            silhouette_for_comparison = logged_silhouette
            silhouette_source = "mlflow"
            if logged_sample_size is not None:
                print(
                    f"    Silhouette (logged in MLflow): {logged_silhouette:.4f} "
                    f"(training run reported this as computed on a "
                    f"{int(logged_sample_size):,}-sample -- NOT the full "
                    f"{n_samples:,}-player training set)"
                )
            else:
                print(f"    Silhouette (logged in MLflow): {logged_silhouette:.4f}")
        else:
            silhouette_for_comparison = silhouette_result["score"]
            silhouette_source = "recomputed" if silhouette_result["score"] is not None else None

        if run_metadata["logged_inertia"] is not None:
            print(f"    Inertia (logged in MLflow): {run_metadata['logged_inertia']:.4f}")

        if n_noise > 0:
            print(f"    Noise: {n_noise:,} ({noise_pct:.2f}%)")

        comparison_rows.append({
            "algorithm": algorithm,
            "n_clusters": len(real_clusters),
            "silhouette_score": silhouette_for_comparison,
            "silhouette_score_source": silhouette_source,
            "silhouette_recomputed_n_samples": silhouette_result["n_samples_used"] or None,
            "silhouette_recomputed_was_sampled": silhouette_result["was_sampled"],
            "inertia": run_metadata["logged_inertia"] if algorithm == "kmeans" else None,
            "noise_count": n_noise if n_noise > 0 else None,
            "noise_pct": round(noise_pct, 2) if n_noise > 0 else None,
            "n_samples": n_samples,
            "run_id": run_id,
        })

        final_summary_by_algorithm[algorithm] = {
            "n_clusters": len(real_clusters),
            "silhouette": silhouette_for_comparison,
            "n_noise": n_noise,
            "noise_pct": noise_pct,
            "n_samples": n_samples,
        }

        # ---------------------------------------------------------------
        # Outputs for this algorithm
        # ---------------------------------------------------------------

        cluster_sizes = calculate_cluster_sizes(labels)

        plot_cluster_sizes(
            cluster_sizes,
            algorithm,
            RESULTS_DIR / f"{CLUSTER_SIZE_FILE_NUMBERS[algorithm]}_cluster_sizes_{algorithm}.png",
        )

        profiles = calculate_cluster_profiles(X_aligned_complete, labels)
        profiles.to_csv(RESULTS_DIR / f"cluster_profiles_{algorithm}.csv")

        differences = calculate_feature_differences(X_aligned_complete, labels)
        differences.to_csv(
            RESULTS_DIR / f"cluster_feature_differences_{algorithm}.csv",
            index=False,
        )

        plot_pca(
            algorithm_pca_coords,
            np.asarray(labels),
            algorithm,
            n_samples,
            RESULTS_DIR / f"{PCA_FILE_NUMBERS[algorithm]}_pca_{algorithm}.png",
        )

        print(f"\u2713 {algorithm} completed")

        del model, X_aligned, X_aligned_complete, X_scaled_aligned, algorithm_pca_coords, labels
        gc.collect()

    del X_common, pca_coords_shared
    gc.collect()

    # =========================================================================
    # [5/6]
    # =========================================================================

    print("\n[5/6] Building final comparison tables...")

    if not comparison_rows:
        raise RuntimeError("No models were successfully evaluated.")

    comparison_df = pd.DataFrame(comparison_rows)

    comparison_df.to_csv(RESULTS_DIR / "final_model_comparison.csv", index=False)

    print("\n" + "=" * 80)
    print("FINAL MODEL COMPARISON")
    print("=" * 80 + "\n")

    print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    plot_silhouette_comparison(comparison_df, RESULTS_DIR / "01_silhouette_comparison.png")

    summary = {
        "dataset": str(DATASET_PATH),
        "algorithms": RUN_IDS,
        "noise_label": NOISE_LABEL,
        "pca": {
            "n_players_visualized": n_players_for_pca,
            "n_features": n_features_for_pca,
            "n_components": PCA_COMPONENTS,
            "explained_variance_ratio": explained_variance,
        },
        "silhouette_sample_size_limit": SILHOUETTE_SAMPLE_SIZE,
        "comparison": comparison_df.to_dict(orient="records"),
        "run_metadata": run_metadata_by_algorithm,
    }

    with open(RESULTS_DIR / "results_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    # =========================================================================
    # [6/6] -- final human-readable summary, in the requested format
    # =========================================================================

    print("\n[6/6] Results generated successfully!\n")

    print("=" * 80)
    print("P5 RESULTS COMPLETED")
    print("=" * 80)

    print("\nDataset:")
    print(f"  Players: {n_players_for_pca:,}")
    print(f"  Features: {n_features_for_pca}")

    print("\nPCA:")
    print(f"  Players visualized: {n_players_for_pca:,}")
    print(f"  Components: {PCA_COMPONENTS}")
    print(
        f"  Explained variance: "
        f"PC1={explained_variance[0]:.4f}, PC2={explained_variance[1]:.4f}"
    )

    for algorithm in RUN_IDS:

        if algorithm not in final_summary_by_algorithm:
            continue

        s = final_summary_by_algorithm[algorithm]

        print(f"\n{DISPLAY_NAMES.get(algorithm, algorithm)}:")
        print(f"  Clusters: {s['n_clusters']}")

        if s["n_noise"] > 0:
            print(f"  Noise: {s['n_noise']:,} ({s['noise_pct']:.2f}%)")

        silhouette_display = (
            f"{s['silhouette']:.4f}" if s["silhouette"] is not None else "N/A"
        )
        print(f"  Silhouette: {silhouette_display}")
        print(f"  Samples: {s['n_samples']:,}")

    print("\nOutput directory:")
    print(f"  {RESULTS_DIR}")

    print("\nGenerated files:")
    for file in sorted(RESULTS_DIR.rglob("*")):
        if file.is_file():
            print(f"  - {file.relative_to(RESULTS_DIR)}")

    print()


if __name__ == "__main__":
    main()
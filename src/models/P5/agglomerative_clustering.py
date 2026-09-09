from pathlib import Path

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import mlflow

from mlflow_tracking.tracking_utils import (
    configure_tracking,
    mlflow_run,
    log_dataset_context,
    log_model_params,
    log_metrics,
    log_model_and_artifacts,
)

from mlflow_tracking.experiments import get_experiment_name


# ============================================================
# Config
# ============================================================

DATASET_PATH = "data/datasets/player_clustering.parquet"

RANDOM_STATE = 42

K_VALUES = range(2, 7)

SILHOUETTE_SAMPLE_SIZE = 50_000

# Agglomerative configurations to explore.
#
# Ward requires Euclidean distance.
# Complete, average and single can be combined
# with different distance metrics.

LINKAGE_METRICS = [
    ("ward", "euclidean"),
    ("complete", "euclidean"),
    ("average", "euclidean"),
    ("single", "euclidean"),
]

# Results directory
RESULTS_DIR = Path(
    "src/models/P5/results_agglomerative_clustering"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Read dataset
# ============================================================

data = pd.read_parquet(DATASET_PATH)

print(f"Dataset shape: {data.shape}")
print(f"Number of players: {len(data)}")


# ============================================================
# Feature selection
# ============================================================

X = data.select_dtypes(
    include=["number", "bool"]
).copy()

print(
    f"Number of clustering features: {X.shape[1]}"
)

print("\nFeatures used for clustering:")
print(X.columns.tolist())


# ============================================================
# Data validation
# ============================================================

missing_values = X.isna().sum().sum()

print("\nMissing values:")
print(missing_values)

print(
    f"\nDuplicated rows: {X.duplicated().sum()}"
)

if missing_values > 0:
    raise ValueError(
        "The dataset contains missing values "
        "in clustering features."
    )


# ============================================================
# Scaling
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nFeatures scaled successfully.")


# ============================================================
# MLflow configuration
# ============================================================

experiment_name = get_experiment_name("p5")

configure_tracking()


# ============================================================
# Sample for Silhouette Score
# ============================================================

rng = np.random.RandomState(
    RANDOM_STATE
)

sample_indices = rng.choice(
    len(X_scaled),
    size=min(
        SILHOUETTE_SAMPLE_SIZE,
        len(X_scaled),
    ),
    replace=False,
)

X_silhouette = X_scaled[
    sample_indices
]


# ============================================================
# Evaluate Agglomerative configurations
# ============================================================

print(
    "\nEvaluating Agglomerative "
    "Clustering configurations..."
)

results = []

for linkage, metric in LINKAGE_METRICS:

    for k in K_VALUES:

        print(
            f"Testing K={k}, "
            f"linkage={linkage}, "
            f"metric={metric}..."
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        model = AgglomerativeClustering(
            n_clusters=k,
            linkage=linkage,
            metric=metric,
        )

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        model.fit(X_scaled)

        labels = model.labels_

        # ----------------------------------------------------
        # Silhouette Score
        # ----------------------------------------------------

        sample_labels = labels[
            sample_indices
        ]

        silhouette = silhouette_score(
            X_silhouette,
            sample_labels,
        )

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        results.append(
            {
                "k": k,
                "linkage": linkage,
                "metric": metric,
                "silhouette_score": silhouette,
            }
        )


# ============================================================
# Convert results to DataFrame
# ============================================================

results_df = pd.DataFrame(results)

print(
    "\nAgglomerative Clustering evaluation:"
)

print(results_df)


# ============================================================
# Select best configuration
# ============================================================

best_result = results_df.loc[
    results_df[
        "silhouette_score"
    ].idxmax()
]

best_k = int(
    best_result["k"]
)

best_linkage = (
    best_result["linkage"]
)

best_metric = (
    best_result["metric"]
)

best_silhouette = float(
    best_result["silhouette_score"]
)


print(
    "\nBest Agglomerative configuration:"
)

print(
    f"K: {best_k}"
)

print(
    f"Linkage: {best_linkage}"
)

print(
    f"Metric: {best_metric}"
)

print(
    f"Silhouette Score: "
    f"{best_silhouette:.4f}"
)


# ============================================================
# Save evaluation results
# ============================================================

evaluation_path = (
    RESULTS_DIR
    / "agglomerative_evaluation.csv"
)

results_df.to_csv(
    evaluation_path,
    index=False,
)


# ============================================================
# Silhouette comparison plot
# ============================================================

plt.figure(figsize=(9, 6))

for linkage, metric in LINKAGE_METRICS:

    subset = results_df[
        (results_df["linkage"] == linkage)
        & (results_df["metric"] == metric)
    ]

    plt.plot(
        subset["k"],
        subset["silhouette_score"],
        marker="o",
        label=f"{linkage} + {metric}",
    )

plt.xlabel("Number of clusters (K)")
plt.ylabel("Silhouette Score")

plt.title(
    "Agglomerative Clustering "
    "Silhouette Score"
)

plt.xticks(list(K_VALUES))

plt.legend()

plt.grid(True)

plt.tight_layout()

silhouette_path = (
    RESULTS_DIR
    / "silhouette_scores.png"
)

plt.savefig(
    silhouette_path,
    dpi=150,
)

plt.show()

plt.close()


# ============================================================
# Best configuration comparison
# ============================================================

best_config_df = pd.DataFrame(
    [
        {
            "k": best_k,
            "linkage": best_linkage,
            "metric": best_metric,
            "silhouette_score": best_silhouette,
        }
    ]
)

best_config_path = (
    RESULTS_DIR
    / "best_configuration.csv"
)

best_config_df.to_csv(
    best_config_path,
    index=False,
)


# ============================================================
# Final Agglomerative model
# ============================================================

final_model = AgglomerativeClustering(
    n_clusters=best_k,
    linkage=best_linkage,
    metric=best_metric,
)

final_model.fit(X_scaled)


# ============================================================
# Cluster assignments
# ============================================================

cluster_labels = final_model.labels_

data_with_clusters = data.copy()

data_with_clusters[
    "cluster"
] = cluster_labels


# ============================================================
# Cluster sizes
# ============================================================

cluster_sizes = (
    data_with_clusters[
        "cluster"
    ]
    .value_counts()
    .sort_index()
)

print("\nCluster sizes:")
print(cluster_sizes)


# ============================================================
# Cluster profiles
# ============================================================

cluster_profile = (
    data_with_clusters
    .groupby("cluster")[
        X.columns.tolist()
    ]
    .mean()
)

print("\nCluster profiles:")
print(cluster_profile)


# ============================================================
# Save cluster analysis results
# ============================================================

cluster_profile_path = (
    RESULTS_DIR
    / "cluster_profile.csv"
)

cluster_sizes_path = (
    RESULTS_DIR
    / "cluster_sizes.csv"
)

cluster_profile.to_csv(
    cluster_profile_path
)

cluster_sizes.to_csv(
    cluster_sizes_path,
    header=["player_count"],
)


# ============================================================
# Save clustering dataset
# ============================================================

output_path = (
    RESULTS_DIR
    / "player_clustering_with_clusters.parquet"
)

data_with_clusters.to_parquet(
    output_path,
    index=False,
)

print(
    f"\nClustering dataset saved to: "
    f"{output_path}"
)


# ============================================================
# MLflow experiment
# ============================================================

with mlflow_run(
    experiment_name,
    run_name="Agglomerative Clustering Final",
):

    # --------------------------------------------------------
    # Dataset context
    # --------------------------------------------------------

    log_dataset_context(
        DATASET_PATH,
        row_count=len(data),
        feature_count=X.shape[1],
    )

    # --------------------------------------------------------
    # Model parameters
    # --------------------------------------------------------

    log_model_params(
        {
            "algorithm": (
                "AgglomerativeClustering"
            ),
            "n_clusters": best_k,
            "linkage": best_linkage,
            "metric": best_metric,
            "scaler": "StandardScaler",
            "silhouette_sample_size": (
                SILHOUETTE_SAMPLE_SIZE
            ),
        }
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    log_metrics(
        {
            "n_clusters": float(best_k),
            "silhouette_score": (
                best_silhouette
            ),
        }
    )

    # --------------------------------------------------------
    # Artifacts
    # --------------------------------------------------------

    log_model_and_artifacts(
        final_model,
        extra_artifacts={
            str(silhouette_path): "plots",
            str(cluster_profile_path): "data",
            str(cluster_sizes_path): "data",
            str(evaluation_path): "data",
            str(best_config_path): "data",
            str(output_path): "data",
        },
    )


# ============================================================
# Final message
# ============================================================

print(
    "\nP5 Agglomerative Clustering "
    "completed successfully."
)

print(
    f"All local results saved to: "
    f"{RESULTS_DIR}"
)
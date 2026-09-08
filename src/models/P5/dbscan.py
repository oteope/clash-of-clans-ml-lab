from pathlib import Path

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
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

SILHOUETTE_SAMPLE_SIZE = 50_000

# Values to explore for DBSCAN
EPS_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
MIN_SAMPLES_VALUES = [5, 10, 20]

# Number of neighbours used for the k-distance graph.
# This should normally match the min_samples value
# you want to investigate.
K_DISTANCE_MIN_SAMPLES = 10

# Results directory
RESULTS_DIR = Path("src/models/P5/results_dbscan")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Read dataset
# ============================================================

data = pd.read_parquet(DATASET_PATH)

print(f"Dataset shape: {data.shape}")
print(f"Number of players: {len(data)}")


# ============================================================
# Feature selection
# ============================================================

# K-Means and DBSCAN require numerical features.
# Boolean features are also accepted as numerical features.

X = data.select_dtypes(
    include=["number", "bool"]
).copy()

print(f"Number of clustering features: {X.shape[1]}")

print("\nFeatures used for clustering:")
print(X.columns.tolist())


# ============================================================
# Data validation
# ============================================================

missing_values = X.isna().sum().sum()

print("\nMissing values:")
print(missing_values)

print(f"\nDuplicated rows: {X.duplicated().sum()}")

if missing_values > 0:
    raise ValueError(
        "The dataset contains missing values in clustering features."
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

rng = np.random.RandomState(42)

sample_indices = rng.choice(
    len(X_scaled),
    size=min(
        SILHOUETTE_SAMPLE_SIZE,
        len(X_scaled),
    ),
    replace=False,
)

X_silhouette = X_scaled[sample_indices]


# ============================================================
# K-distance graph
# ============================================================

print("\nCalculating k-distance graph...")

neighbors = NearestNeighbors(
    n_neighbors=K_DISTANCE_MIN_SAMPLES
)

neighbors.fit(X_scaled)

distances, _ = neighbors.kneighbors(X_scaled)

# Distance to the kth nearest neighbour
k_distances = distances[:, -1]

k_distances = np.sort(k_distances)

plt.figure(figsize=(8, 6))

plt.plot(k_distances)

plt.xlabel("Points sorted by distance")
plt.ylabel(
    f"Distance to {K_DISTANCE_MIN_SAMPLES}th nearest neighbour"
)
plt.title("DBSCAN k-Distance Graph")
plt.grid(True)

plt.tight_layout()

k_distance_path = (
    RESULTS_DIR / "k_distance_graph.png"
)

plt.savefig(
    k_distance_path,
    dpi=150,
)

plt.show()
plt.close()

print(
    f"k-distance graph saved to: {k_distance_path}"
)


# ============================================================
# Evaluate DBSCAN configurations
# ============================================================

print("\nEvaluating DBSCAN configurations...")

results = []

for eps in EPS_VALUES:

    for min_samples in MIN_SAMPLES_VALUES:

        print(
            f"Testing eps={eps}, "
            f"min_samples={min_samples}..."
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            n_jobs=-1
        )

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        model.fit(X_scaled)

        labels = model.labels_

        # ----------------------------------------------------
        # Cluster information
        # ----------------------------------------------------

        unique_labels = set(labels)

        cluster_labels = unique_labels - {-1}

        n_clusters = len(cluster_labels)

        noise_count = int(
            np.sum(labels == -1)
        )

        noise_percentage = (
            noise_count / len(labels)
        ) * 100

        # ----------------------------------------------------
        # Silhouette Score
        # ----------------------------------------------------

        sample_labels = labels[sample_indices]

        sample_non_noise = sample_labels != -1

        silhouette = np.nan

        if (
            n_clusters >= 2
            and sample_non_noise.sum() > 1
        ):

            silhouette = silhouette_score(
                X_silhouette[sample_non_noise],
                sample_labels[sample_non_noise],
            )

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        results.append(
            {
                "eps": eps,
                "min_samples": min_samples,
                "n_clusters": n_clusters,
                "noise_count": noise_count,
                "noise_percentage": noise_percentage,
                "silhouette_score": silhouette,
            }
        )


# ============================================================
# Convert results to DataFrame
# ============================================================

results_df = pd.DataFrame(results)

print("\nDBSCAN evaluation:")
print(results_df)


# ============================================================
# Select best configuration
# ============================================================

valid_results = results_df.dropna(
    subset=["silhouette_score"]
)

if valid_results.empty:

    raise ValueError(
        "No DBSCAN configuration produced at least "
        "two valid clusters for Silhouette Score."
    )


best_result = valid_results.loc[
    valid_results["silhouette_score"].idxmax()
]

best_eps = float(best_result["eps"])

best_min_samples = int(
    best_result["min_samples"]
)

best_n_clusters = int(
    best_result["n_clusters"]
)

best_noise_count = int(
    best_result["noise_count"]
)

best_noise_percentage = float(
    best_result["noise_percentage"]
)

best_silhouette = float(
    best_result["silhouette_score"]
)


print(
    f"\nBest DBSCAN configuration:"
)

print(
    f"eps: {best_eps}"
)

print(
    f"min_samples: {best_min_samples}"
)

print(
    f"Number of clusters: {best_n_clusters}"
)

print(
    f"Noise count: {best_noise_count}"
)

print(
    f"Noise percentage: {best_noise_percentage:.2f}%"
)

print(
    f"Silhouette Score: {best_silhouette:.4f}"
)


# ============================================================
# Save evaluation results
# ============================================================

evaluation_path = (
    RESULTS_DIR / "dbscan_evaluation.csv"
)

results_df.to_csv(
    evaluation_path,
    index=False,
)


# ============================================================
# Visualize Silhouette Score
# ============================================================

plt.figure(figsize=(8, 6))

for min_samples in MIN_SAMPLES_VALUES:

    subset = results_df[
        results_df["min_samples"]
        == min_samples
    ]

    plt.plot(
        subset["eps"],
        subset["silhouette_score"],
        marker="o",
        label=f"min_samples={min_samples}",
    )

plt.xlabel("eps")
plt.ylabel("Silhouette Score")
plt.title(
    "DBSCAN Silhouette Score by Configuration"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

silhouette_path = (
    RESULTS_DIR / "dbscan_silhouette_scores.png"
)

plt.savefig(
    silhouette_path,
    dpi=150,
)

plt.show()
plt.close()


# ============================================================
# Final DBSCAN model
# ============================================================

final_model = DBSCAN(
    eps=best_eps,
    min_samples=best_min_samples,
)

final_model.fit(X_scaled)


# ============================================================
# Cluster assignments
# ============================================================

cluster_labels = final_model.labels_

data_with_clusters = data.copy()

data_with_clusters["cluster"] = cluster_labels


# ============================================================
# Cluster sizes
# ============================================================

cluster_sizes = (
    data_with_clusters["cluster"]
    .value_counts()
    .sort_index()
)

print("\nCluster sizes:")
print(cluster_sizes)


# ============================================================
# Cluster profiles
# ============================================================

# Exclude noise (-1) from cluster profiles.

clustered_data = data_with_clusters[
    data_with_clusters["cluster"] != -1
]

cluster_profile = (
    clustered_data
    .groupby("cluster")[X.columns.tolist()]
    .mean()
)

print("\nCluster profiles:")
print(cluster_profile)


# ============================================================
# Save cluster analysis results
# ============================================================

cluster_profile_path = (
    RESULTS_DIR / "cluster_profile.csv"
)

cluster_sizes_path = (
    RESULTS_DIR / "cluster_sizes.csv"
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
    f"\nClustering dataset saved to: {output_path}"
)


# ============================================================
# MLflow experiment
# ============================================================

with mlflow_run(
    experiment_name,
    run_name="DBSCAN Final",
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
            "algorithm": "DBSCAN",
            "eps": best_eps,
            "min_samples": best_min_samples,
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
            "n_clusters": float(best_n_clusters),
            "noise_count": float(best_noise_count),
            "noise_percentage": (
                best_noise_percentage
            ),
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
            str(k_distance_path): "plots",
            str(silhouette_path): "plots",
            str(cluster_profile_path): "data",
            str(cluster_sizes_path): "data",
            str(evaluation_path): "data",
            str(output_path): "data",
        },
    )


# ============================================================
# Final message
# ============================================================

print("\nP5 DBSCAN completed successfully.")

print(
    f"All local results saved to: {RESULTS_DIR}"
)
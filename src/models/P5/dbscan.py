from pathlib import Path

from sklearn.cluster import DBSCAN
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
SILHOUETTE_SAMPLE_SIZE = 50_000

# Results directory
RESULTS_DIR = Path("src/models/P5/results_kmeans")
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

X = data.select_dtypes( inlcude = ["number","bool"]).copy()

print(f"Number of clustering features: {X.shape[1]}")

print("\nFeatures used for clustering:")
print(X.columns.tolist())

# ============================================================
# Scaling
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ============================================================
# Sample for Silhouette Score
# ============================================================

rng = np.random.RandomState(RANDOM_STATE)

sample_indices = rng.choice(
    len(X_scaled),
    size = min(SILHOUETTE_SAMPLE_SIZE, len(X_scaled)),
    replace = False,
    )

X_silhouette = X_scaled[sample_indices]

# --------------------------------------------------------
# Model
# --------------------------------------------------------
results = []

model = DBSCAN(
    random_state = RANDOM_STATE,
    
)

# --------------------------------------------------------
# Training
# --------------------------------------------------------

model.fit(X_scaled)

labels = model.labels_

# --------------------------------------------------------
# Inertia
# --------------------------------------------------------

inertia = model.inertia_

# --------------------------------------------------------
# Silhouette Score
# --------------------------------------------------------

sample_labels = labels[sample_indices]

silhouette = silhouette_score(
    X_silhouette,
    sample_labels,
)

# --------------------------------------------------------
 # Store results
# --------------------------------------------------------

results.append(
    {
        "inertia": inertia,
        "silhouette_score": silhouette,
        }
    )

# ============================================================
# Convert results to DataFrame
# ============================================================

results_df = pd.DataFrame(results)
print("\nClustering evaluation:")
print(results_df)

# ============================================================
# Final K-Means model
# ============================================================

final_model = DBSCAN(
    random_state = RANDOM_STATE,
    
)

final_model.fit(X_scaled)

# ============================================================
# Cluster assignments
# ============================================================

cluster_labels = final_model.labels_
data_with_clusters = data.copy()
data_with_clusters ["clusters"] = cluster_labels

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

cluster_profile = (
    data_with_clusters
    .groupby("cluster")[X.columns.tolist()]
    .mean()
)

print("\nCluster profiles:")
print(cluster_profile)

# ============================================================
# Save cluster analysis results
# ============================================================

cluster_profile_path = RESULTS_DIR / "cluster_profile.csv"
cluster_sizes_path = RESULTS_DIR / "cluster_sizes.csv"

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
    RESULTS_DIR / "player_clustering_with_clusters.parquet"
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
experiment_name = get_experiment_name("p5")

configure_tracking()

with mlflow_run(
    experiment_name,
    run_name = "DBSCAN baseline",
):
    #Log dataset context
    log_dataset_context(
        DATASET_PATH,
        row_count = len(data),
        feature_count = X.shape[1],
        
    )
    
    #Log model parameters
    log_model_params(
        {
            "algorithm": "DBSCAN",
            "random_state": RANDOM_STATE,
            "scaler": "StandardScaler",
        }
    )
    
    #Log metrcis
    log_metrics(
        {
            "inertia": float(final_model.inertia_),
            "silhouette_score": float(silhouette),
        }
    )
    
    # --------------------------------------------------------
    # Artifacts
    # --------------------------------------------------------

    log_model_and_artifacts(
        final_model,
        extra_artifacts={
            str(cluster_profile_path): "data",
            str(cluster_sizes_path): "data",
            str(output_path): "data",
        },
    )

# ============================================================
# Final message
# ============================================================

print("\nP5 K-Means completed successfully.")
print(f"All local results saved to: {RESULTS_DIR}")
    
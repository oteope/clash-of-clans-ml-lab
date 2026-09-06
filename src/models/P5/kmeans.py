from sklearn.cluster import KMeans
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

# Candidate number of clusters
K_VALUES = range(2, 11)


# ============================================================
# Read dataset
# ============================================================

data = pd.read_parquet(DATASET_PATH)

print(f"Dataset shape: {data.shape}")
print(f"Number of players: {len(data)}")


# ============================================================
# Feature selection
# ============================================================

# Exclude player identifiers and non-numeric columns.
# K-Means requires numerical features.
X = data.select_dtypes(include=["number", "bool"]).copy()

print(f"Number of clustering features: {X.shape[1]}")
print("\nFeatures used for clustering:")
print(X.columns.tolist())


# ============================================================
# Data validation
# ============================================================

print("\nMissing values:")
print(X.isna().sum().sum())

print(f"\nDuplicated rows: {X.duplicated().sum()}")


# K-Means cannot work with missing values.
if X.isna().sum().sum() > 0:
    raise ValueError("The dataset contains missing values in clustering features.")


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
# Evaluate different K values
# ============================================================

results = []

# ============================================================
# Sample for Silhouette Score
# ============================================================

rng = np.random.RandomState(RANDOM_STATE)

sample_indices = rng.choice(
    len(X_scaled),
    size=min(SILHOUETTE_SAMPLE_SIZE, len(X_scaled)),
    replace=False,
)

X_silhouette = X_scaled[sample_indices]

# ============================================================
# Evaluate candidate cluster counts
# ============================================================

print("\nEvaluating candidate cluster counts...")

for k in K_VALUES:

    print(f"Testing K={k}...")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init="auto",
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
            "k": k,
            "inertia": inertia,
            "silhouette_score": silhouette,
        }
    )


# Convert results to DataFrame
results_df = pd.DataFrame(results)

print("\nClustering evaluation:")
print(results_df)


# ============================================================
# Select K using Silhouette Score
# ============================================================

best_k = results_df.loc[
    results_df["silhouette_score"].idxmax(),
    "k",
]

best_silhouette = results_df.loc[
    results_df["silhouette_score"].idxmax(),
    "silhouette_score",
]

print(f"\nBest K according to Silhouette Score: {best_k}")
print(f"Best Silhouette Score: {best_silhouette:.4f}")


# ============================================================
# Elbow plot
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    results_df["k"],
    results_df["inertia"],
    marker="o",
)

plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.title("K-Means Elbow Method")
plt.xticks(list(K_VALUES))
plt.grid(True)

plt.tight_layout()

elbow_path = "elbow_plot.png"
plt.savefig(elbow_path, dpi=150)
plt.show()

plt.close()


# ============================================================
# Silhouette plot
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    results_df["k"],
    results_df["silhouette_score"],
    marker="o",
)

plt.xlabel("Number of clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score by Number of Clusters")
plt.xticks(list(K_VALUES))
plt.grid(True)

plt.tight_layout()

silhouette_path = "silhouette_scores.png"
plt.savefig(silhouette_path, dpi=150)
plt.show()

plt.close()


# ============================================================
# Final K-Means model
# ============================================================

final_model = KMeans(
    n_clusters=int(best_k),
    random_state=RANDOM_STATE,
    n_init="auto",
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

cluster_profile = (
    data_with_clusters
    .groupby("cluster")[X.columns.tolist()]
    .mean()
)

print("\nCluster profiles:")
print(cluster_profile)


# ============================================================
# MLflow experiment
# ============================================================

with mlflow_run(
    experiment_name,
    run_name="K-Means Final",
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
            "algorithm": "KMeans",
            "n_clusters": int(best_k),
            "random_state": RANDOM_STATE,
            "n_init": "auto",
            "scaler": "StandardScaler",
        }
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    log_metrics(
        {
            "inertia": float(final_model.inertia_),
            "silhouette_score": float(best_silhouette),
        }
    )

    # --------------------------------------------------------
    # Artifacts
    # --------------------------------------------------------

    cluster_profile_path = "cluster_profile.csv"
    cluster_sizes_path = "cluster_sizes.csv"
    evaluation_path = "k_evaluation.csv"

    cluster_profile.to_csv(
        cluster_profile_path
    )

    cluster_sizes.to_csv(
        cluster_sizes_path,
        header=["player_count"],
    )

    results_df.to_csv(
        evaluation_path,
        index=False,
    )

    log_model_and_artifacts(
        final_model,
        extra_artifacts={
            "elbow_plot": elbow_path,
            "silhouette_plot": silhouette_path,
            "cluster_profile": cluster_profile_path,
            "cluster_sizes": cluster_sizes_path,
            "k_evaluation": evaluation_path,
        },
    )


# ============================================================
# Save clustering results
# ============================================================

output_path = "data/datasets/player_clustering_with_clusters.parquet"

data_with_clusters.to_parquet(
    output_path,
    index=False,
)

print(
    f"\nClustering dataset saved to: {output_path}"
)

print("\nP5 K-Means completed successfully.")
# ⚔️ Clash of Clans ML Lab

> An end-to-end Machine Learning and data engineering project built from real Clash of Clans API data.

**Clash of Clans ML Lab** started as a way to learn Machine Learning through a real-world dataset instead of toy datasets.

The project transforms raw Clash of Clans API data into reusable tables, engineered features and ML datasets for several supervised and unsupervised learning problems.

The main focus is not only model performance, but also **reproducibility, data quality, feature engineering, automated testing, leakage prevention and experiment tracking**.

🚧 **Status: Active development**

---

## 🎯 Project Overview

The goal is to build a complete ML pipeline around Clash of Clans data and investigate different Machine Learning formulations using the same underlying data.

```text
⚔️ Clash of Clans API
        ↓
📦 Raw JSON
        ↓
🔧 Data Processing & Normalization
        ↓
🗃️ Parquet Tables
        ↓
🧠 Feature Engineering
        ↓
📊 ML Datasets
        ↓
🤖 Model Experiments
        ↓
📈 Evaluation & Tracking
        ↓
🐳 Reproducible Deployment
        ↓
🖥️ Decentralized Compute
```

The data infrastructure and ML dataset engineering layers are already implemented. The project is currently moving into the **model experimentation and reproducibility stage**.

---

# 🧠 Machine Learning Problems

The project uses the same underlying Clash of Clans data to investigate several different ML problems.

## 1️⃣ Player Role Classification

**Task:** Classification  
**Unit:** Player-clan relationship  
**Target:** `role`

The goal is to investigate whether player and clan characteristics can be used to classify a player's role within a clan.

📁 Dataset:

```text
data/.../role_classification.parquet
```

---

## 2️⃣ Clan Rank Regression

**Task:** Regression / Ordinal prediction  
**Unit:** Player-clan relationship  
**Target:** `clan_rank`

The goal is to investigate:

> **To what extent can a player's relative position within their clan be estimated from their characteristics?**

Two dataset variants are used:

```text
📊 clan_rank_regression_with_trophies.parquet
📊 clan_rank_regression_without_trophies.parquet
```

The second variant removes trophy-related information to investigate how much predictive information is provided by that feature family.

The problem is initially formulated as regression, while the ordinal nature of ranking may be explored later as an alternative formulation.

Particular attention is given to **target leakage and proxy variables**, especially features that may directly reproduce the logic used to calculate the rank.

---

## 3️⃣ Clan War Performance Regression

**Task:** Regression  
**Unit:** Clan  
**Target:** `war_success_rate`

```text
war_wins
────────────────────────────────────────
war_wins + war_losses + war_ties
```

A minimum amount of historical war data is required so that clans with very limited history do not disproportionately influence the dataset.

Direct variables representing the target itself are excluded from the feature set to prevent artificially easy predictions.

📁 Dataset:

```text
clan_war_performance_regression.parquet
```

---

## 4️⃣ Clan Performance Classification

**Task:** Multiclass classification  
**Unit:** Clan  
**Target:** `performance_class`

Classes:

```text
🟥 low
🟨 medium
🟩 high
```

The class boundaries are derived from the observed distribution of `war_success_rate` using **terciles**, rather than arbitrary manually selected thresholds.

📁 Dataset:

```text
clan_performance_classification.parquet
```

---

## 5️⃣ Player Clustering

**Task:** Unsupervised learning  
**Unit:** Player  
**Target:** None

The goal is to discover natural player profiles without defining the categories beforehand.

Planned algorithms include:

- 🔵 K-Means
- 🔵 DBSCAN
- 🔵 Hierarchical Clustering

Potential feature groups include:

- 🏰 Town Hall / Builder Hall
- ⭐ Experience
- 🏆 Trophies
- ⚔️ Combat activity
- 🎁 Donations
- 🪖 Troop progression
- 👑 Hero progression
- ✨ Spell progression
- 🛡️ Equipment progression
- 🏅 Achievement progression

🚧 **Status:** Dataset construction and validation are in progress.

---

# 🏗️ Data Pipeline Architecture

```text
                    ⚔️ Clash of Clans API
                            │
                            ▼
                       📦 Raw JSON
                            │
                            ▼
                  🔧 Normalization
                            │
                            ▼
                   🗃️ Parquet Tables
                            │
             ┌──────────────┴──────────────┐
             ▼                             ▼
      📊 Small tables              📊 Large tables
      players / clans              troops / heroes
      clan members                 spells / equipment
                                   achievements
             │                             │
             └──────────────┬──────────────┘
                            ▼
                  🧠 player_features
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
            P1             P2             P3
             │              │              │
             └──────────────┼──────────────┘
                            ▼
                    Problem-specific
                         datasets
                            │
                            ▼
                   🤖 ML Experiments
```

Large Parquet tables are processed in **batches** to avoid loading the entire dataset into memory simultaneously.

---

# 🧩 Reusable Feature Layer

One of the central components of the project is:

```text
player_features.parquet
```

**1 row = 1 player**

This reusable feature layer combines base player information with derived and aggregated features such as:

- 🏰 Town Hall
- 🏗️ Builder Hall
- ⭐ Experience
- 🏆 Trophies
- 🥇 Best trophies
- ⚔️ War stars
- 🗡️ Attack wins
- 🛡️ Defense wins
- 🎁 Donations
- 📥 Donations received
- 🏗️ Capital contributions
- 🪖 Troop progression
- 👑 Hero progression
- ✨ Spell progression
- 🛡️ Equipment progression
- 🏅 Achievement progression

The feature layer allows different ML problems to reuse the same engineered player-level information without repeatedly processing the largest raw tables.

---

# 🧪 Testing

Testing is an important part of the project.

The current test suite covers:

- ✅ API extraction
- ✅ Raw data handling
- ✅ Data processing and normalization
- ✅ Problem 1
- ✅ Problem 2
- ✅ Problem 3
- ✅ Problem 4

Before starting Problem 5 development, the complete suite reached:

```text
✅ 124 passed
```

🚧 Problem 5-specific tests are still to be added.

---

# 📈 Experiment Tracking with MLflow

The project uses **MLflow for local experiment tracking and reproducibility**.

MLflow records information such as:

- Dataset context
- Dataset size and feature count
- Target variable
- Train/test split configuration
- Preprocessing configuration
- Model parameters
- Evaluation metrics
- Trained models
- Evaluation artifacts

## Requirements

Install the project dependencies:

```powershell
pip install -r requirements.txt
```

## Local MLflow Server

Start the local tracking server:

```powershell
mlflow server `
  --backend-store-uri sqlite:///mlflow/mlflow.db `
  --default-artifact-root ./mlflow/mlruns `
  --host 127.0.0.1 `
  --port 5000
```

Set the tracking URI:

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
```

## Experiment Naming

Each ML problem has a dedicated MLflow experiment:

| Code | Experiment |
|---|---|
| `p1` | `p1_role_classification` |
| `p2` | `p2_clan_rank` |
| `p3` | `p3_war_performance` |
| `p4` | `p4_clan_performance_classification` |
| `p5` | `p5_player_clustering` |

## Example

```python
from mlflow_tracking.tracking_utils import (
    configure_tracking,
    mlflow_run,
    log_dataset_context,
    log_split_config,
    log_model_params,
    log_metrics,
    log_model_and_artifacts,
)
from mlflow_tracking.experiments import get_experiment_name

configure_tracking()

experiment_name = get_experiment_name("p1")

with mlflow_run(
    experiment_name,
    run_name="example_experiment",
):
    log_dataset_context(
        "data/processed/p1_dataset.parquet",
        row_count=1000,
        feature_count=20,
        target="role",
    )

    log_split_config(
        split_strategy="GroupKFold",
        grouping_col="clan_tag",
        random_seed=42,
        preprocessing_config={
            "imputer": "median",
        },
    )

    log_model_params({
        "n_estimators": 100,
        "max_depth": 3,
    })

    # Train and evaluate the model...

    log_metrics({
        "accuracy": 0.95,
    })

    log_model_and_artifacts(
        model,
        confusion_matrix=cm,
        class_names=["0", "1"],
    )
```

## MLflow Smoke Test

```powershell
python -m unittest tests.test_mlflow_smoke
```

---

# 🤖 Model Experimentation

The project is currently entering the model experimentation stage.

The initial approach is deliberately based on **simple baseline models before hyperparameter optimization**.

For supervised problems, baseline experiments are being tracked with MLflow to establish reproducible reference points before more advanced experimentation.

For example, Problem 2 currently uses:

- Ridge Regression
- Random Forest Regressor
- XGBoost Regressor

Both the **with-trophies** and **without-trophies** datasets are evaluated using equivalent baseline configurations.

Evaluation metrics depend on the problem formulation. For regression problems, the initial metrics include:

- MAE
- RMSE
- R²

The purpose of these baselines is not to maximize performance immediately, but to establish a reliable reference for subsequent experiments.

---

# 📊 Current Status

| Component | Status |
|---|---|
| ⚔️ API extraction | ✅ |
| 📦 Raw storage | ✅ |
| 🔧 Data processing | ✅ |
| 🗃️ Parquet pipeline | ✅ |
| 🧠 Player feature layer | ✅ |
| 1️⃣ Problem 1 dataset | ✅ |
| 2️⃣ Problem 2 dataset | ✅ |
| 3️⃣ Problem 3 dataset | ✅ |
| 4️⃣ Problem 4 dataset | ✅ |
| 5️⃣ Problem 5 dataset | 🟡 |
| 🧪 Automated testing | ✅ |
| 📊 EDA | 🟡 |
| 🤖 Baseline experiments | 🟡 |
| 🔬 Model comparison | ⬜ |
| ⚙️ Hyperparameter experiments | ⬜ |
| 📈 MLflow tracking | ✅ |
| 🐳 Docker | ⬜ |
| 🖥️ Nosana | ⬜ |
| 🌐 Model API / inference | ⬜ |
| 💾 Arweave | ⬜ |
| 🖥️ Frontend | ⬜ |

---

# 🌐 Decentralize AI Hackathon

🚀 **Clash of Clans ML Lab is being developed as a submission for the Decentralize AI Hackathon.**

The existing project already provides a real ML/data pipeline. The next stage is to investigate whether selected ML workloads can be made **portable and reproducible on decentralized GPU infrastructure**.

### Planned architecture

```text
⚔️ Clash of Clans data
        ↓
🧠 Feature pipeline
        ↓
🤖 ML experiment
        ↓
📈 MLflow
        ↓
🐳 Docker
        ↓
🖥️ Nosana
        ↓
⚡ Training / inference
```

### Technologies

**📈 MLflow**  
Experiment tracking, metrics, artifacts and model versions.

**🐳 Docker**  
Portable and reproducible ML environments.

**🖥️ Nosana**  
Decentralized GPU infrastructure for selected training and/or inference workloads.

**💾 Arweave**  
Potential future use for permanent storage of selected model artifacts or provenance information.

> **Note:** MLflow is already implemented for experiment tracking. Docker, Nosana and Arweave are planned extensions.

---

# 🎯 Hackathon Goal

The project is not intended to claim that decentralized compute will replace AWS, GCP or Azure.

Instead, it aims to answer a practical engineering question:

> **Can a real ML workload developed locally be packaged and executed reproducibly on decentralized GPU infrastructure?**

Clash of Clans provides a concrete real-world workload for testing that idea.

Rather than building a theoretical architecture from scratch, the project already contains:

- ✅ Real API data
- ✅ Data processing
- ✅ Feature engineering
- ✅ Multiple ML datasets
- ✅ Automated tests
- ✅ Experiment tracking

The next step is to make selected workloads portable.

---

# 🗺️ Roadmap

## ✅ Phase 1 — Data Infrastructure

- [x] Clash of Clans API extraction
- [x] Raw data storage
- [x] Data normalization
- [x] Parquet processing
- [x] Reusable player features

## ✅ Phase 2 — ML Dataset Engineering

- [x] Player role classification
- [x] Clan rank regression
- [x] Clan war performance regression
- [x] Clan performance classification
- [ ] Player clustering

## 🔄 Phase 3 — ML Experiments

- [ ] Exploratory Data Analysis
- [x] Initial baseline models
- [ ] Model comparison
- [ ] Hyperparameter experiments
- [ ] Final evaluation

## 🔄 Phase 4 — MLOps

- [x] MLflow
- [x] Experiment tracking
- [ ] Model versioning
- [ ] Docker
- [ ] Reproducible training

## 🔜 Phase 5 — Decentralized Compute

- [ ] Nosana integration
- [ ] GPU training workload
- [ ] Inference workload
- [ ] Local vs decentralized execution comparison

## 🔜 Phase 6 — Future Product

- [ ] Model serving
- [ ] API
- [ ] Potential frontend
- [ ] Possible Arweave integration

---

# 🔬 Project Philosophy

The project is not focused exclusively on achieving the highest possible model score.

It is also an exercise in understanding how to build a **reproducible ML system from raw data to deployment**.

The project therefore emphasizes:

- 🧪 Experiment design
- 🔍 Data leakage and proxy detection
- 🧠 Feature engineering
- 🧱 Reusable data pipelines
- ✅ Automated testing
- 📦 Reproducibility
- 📈 Experiment tracking
- 🚀 Deployment

---

# 🛠️ Tech Stack

| Area | Technology |
|---|---|
| 🐍 Language | Python |
| 📊 Data | pandas, NumPy |
| 🗃️ Storage | Parquet / PyArrow |
| 🤖 Machine Learning | scikit-learn, XGBoost |
| 📈 Experiment Tracking | MLflow |
| 🧪 Testing | pytest |
| 🌿 Version Control | Git / GitHub |
| 🐳 Planned Packaging | Docker |
| 🖥️ Planned Compute | Nosana |
| 💾 Potential Storage | Arweave |

---

# ⚠️ Disclaimer

This is an independent Machine Learning project using data obtained through the Clash of Clans API.

**Clash of Clans**, Supercell and their related trademarks and intellectual property belong to their respective owners.

This project is **not affiliated with or endorsed by Supercell**.

---

# 📄 License

This project is licensed under the **MIT License**.

See [LICENSE](LICENSE) for details.

---

# 🚀 Follow the Project

The project is being developed publicly as part of the **Decentralize AI Hackathon**.

⭐ Star the repository if you want to follow the development.

🔧 **Current milestone:** finish the ML experimentation layer and begin making selected workloads portable and reproducible on decentralized infrastructure.

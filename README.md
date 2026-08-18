# ⚔️ Clash of Clans ML Lab

> 🧠 An end-to-end Machine Learning and data pipeline built from Clash of Clans API data.

Clash of Clans ML Lab started as a way to learn Machine Learning through a **real-world dataset** instead of toy datasets.

The project transforms raw Clash of Clans API data into reusable features and datasets for multiple ML problems, with a strong focus on **reproducibility, testing, feature engineering and avoiding data leakage**.

🚧 **Status: Active development**

---

## 🎯 What is this project?

The goal is to build a complete ML pipeline around Clash of Clans data and investigate different types of Machine Learning problems using the same underlying data.

```text
⚔️ Clash of Clans API
        ↓
📦 Raw JSON
        ↓
🔧 Data Processing
        ↓
🗃️ Parquet Tables
        ↓
🧠 Feature Engineering
        ↓
📊 ML Datasets
        ↓
🤖 Model Experiments
        ↓
📈 Evaluation
        ↓
🚀 MLOps / Deployment
```

The project is currently focused on the **data and feature-engineering layers**. Model experimentation and MLOps are the next stages.

---

## 🧠 Machine Learning Problems

The same underlying data is used to explore several different ML formulations.

### 1️⃣ Player Role Classification

**Task:** Classification
**Unit:** Player-clan relationship
**Target:** `role`

The goal is to investigate whether player and clan characteristics can be used to classify a player's role within a clan.

📁 Dataset:

```text
data/.../role_classification.parquet
```

---

### 2️⃣ Clan Rank Regression

**Task:** Regression
**Unit:** Player-clan relationship
**Target:** `clan_rank`

Two versions are created:

```text
📊 clan_rank_regression_with_trophies.parquet
📊 clan_rank_regression_without_trophies.parquet
```

The second version removes trophy-related information to investigate how strongly clan rank depends on that feature family.

🔍 The feature-engineering process also includes checks for potential proxies and target leakage.

---

### 3️⃣ Clan War Performance Regression

**Task:** Regression
**Unit:** Clan
**Target:** `war_success_rate`

```text
war_wins
─────────────────────────────────
war_wins + war_losses + war_ties
```

A minimum amount of historical war data is required so that clans with very limited history do not dominate the dataset.

Direct war-performance variables are excluded from the feature set to avoid making the prediction task artificially easy.

📁 Dataset:

```text
clan_war_performance_regression.parquet
```

---

### 4️⃣ Clan Performance Classification

**Task:** Classification
**Unit:** Clan
**Target:** `performance_class`

Classes:

```text
🟥 low
🟨 medium
🟩 high
```

The class thresholds are derived from the observed distribution of `war_success_rate` using terciles instead of arbitrary values.

📁 Dataset:

```text
clan_performance_classification.parquet
```

---

### 5️⃣ Player Clustering

**Task:** Unsupervised Learning
**Unit:** Player
**Target:** None

The goal is to discover **natural player profiles** without defining the categories beforehand.

Planned algorithms:

* 🔵 K-Means
* 🔵 DBSCAN
* 🔵 Hierarchical Clustering

Potential features include:

* 🏰 Town Hall / Builder Hall
* ⭐ Experience
* 🏆 Trophies
* ⚔️ Combat activity
* 🎁 Donations
* 🪖 Troop progression
* 👑 Hero progression
* ✨ Spell progression
* 🛡️ Equipment progression
* 🏅 Achievement progression

🚧 **Status:** Dataset construction and validation are still in progress.

---

# 🏗️ Pipeline Architecture

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
             ▼              ▼              ▼
          P4 / P5 → problem-specific datasets
                            │
                            ▼
                   🤖 ML Experiments
```

Large Parquet tables are processed in **batches** to avoid loading everything into memory simultaneously.

---

# 🧩 Reusable Feature Layer

One of the central parts of the project is:

```text
player_features.parquet
```

📌 **1 row = 1 player**

It combines base player information with derived and aggregated features such as:

* 🏰 Town Hall
* 🏗️ Builder Hall
* ⭐ Experience
* 🏆 Trophies
* 🥇 Best trophies
* ⚔️ War stars
* 🗡️ Attack wins
* 🛡️ Defense wins
* 🎁 Donations
* 📥 Donations received
* 🏗️ Capital contributions
* 🪖 Troop progression
* 👑 Hero progression
* ✨ Spell progression
* 🛡️ Equipment progression
* 🏅 Achievement progression

This reusable feature layer prevents the project from repeatedly processing the largest raw tables for every ML problem.

---

# 🧪 Testing

Testing is an important part of the project.

The current test suite covers:

✅ API / extraction
✅ Raw data handling
✅ Processing / normalization
✅ Problem 1
✅ Problem 2
✅ Problem 3
✅ Problem 4

Before Problem 5 development started, the complete suite reached:

```text
✅ 124 passed
```

🚧 Problem 5-specific tests are still to be added.

---

# 📊 Current Status

| Component            | Status |
| -------------------- | ------ |
| ⚔️ API extraction    | ✅      |
| 📦 Raw storage       | ✅      |
| 🔧 Data processing   | ✅      |
| 🗃️ Parquet pipeline | ✅      |
| 🧠 Player features   | ✅      |
| 1️⃣ Problem 1        | ✅      |
| 2️⃣ Problem 2        | ✅      |
| 3️⃣ Problem 3        | ✅      |
| 4️⃣ Problem 4        | ✅      |
| 5️⃣ Problem 5        | 🟡     |
| 📊 Full EDA          | 🟡     |
| 🤖 Model experiments | ⬜      |
| 📈 MLflow            | ⬜      |
| 🐳 Docker            | ⬜      |
| 🖥️ Nosana           | ⬜      |
| 🌐 API / inference   | ⬜      |
| 💾 Arweave           | ⬜      |
| 🖥️ Frontend         | ⬜      |

---

# 🌐 Decentralize AI Hackathon

🚀 **Clash of Clans ML Lab is being developed as a submission for the Decentralize AI Hackathon.**

The existing project is a real ML/data pipeline.

The next stage is to investigate whether selected ML workloads can be made **portable and reproducible on decentralized GPU infrastructure**.

### 🛠️ Planned direction

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

### 🔬 Technologies being explored

**📈 MLflow**
Experiment tracking, metrics, artifacts and model versions.

**🐳 Docker**
Portable and reproducible ML environments.

**🖥️ Nosana**
Decentralized GPU infrastructure for selected training and/or inference workloads.

**💾 Arweave**
Potential future use for permanent storage of selected model artifacts or provenance information.

> ⚠️ **Important:** MLflow, Docker, Nosana and Arweave are planned extensions. They are not part of the currently implemented pipeline yet.

---

# 🎯 Hackathon Goal

The goal isn't to claim that decentralized compute will replace AWS, GCP or Azure.

Instead, I want to answer a practical question:

> **Can a real ML workload developed locally be packaged and executed reproducibly on decentralized GPU infrastructure?**

COC provides a concrete workload to test that idea.

Rather than building a theoretical architecture from scratch, the project already has:

✅ Real API data
✅ Data processing
✅ Feature engineering
✅ Multiple ML datasets
✅ Automated tests

The next step is infrastructure.

---

# 🗺️ Roadmap

### ✅ Phase 1 — Data Infrastructure

* [x] Clash of Clans API extraction
* [x] Raw data storage
* [x] Data normalization
* [x] Parquet processing
* [x] Reusable player features

### ✅ Phase 2 — ML Dataset Engineering

* [x] Player role classification
* [x] Clan rank regression
* [x] Clan war performance regression
* [x] Clan performance classification
* [ ] Player clustering

### 🔜 Phase 3 — ML Experiments

* [ ] Exploratory Data Analysis
* [ ] Baseline models
* [ ] Model comparison
* [ ] Hyperparameter experiments
* [ ] Final evaluation

### 🔜 Phase 4 — MLOps

* [ ] MLflow
* [ ] Experiment tracking
* [ ] Model versioning
* [ ] Docker
* [ ] Reproducible training

### 🔜 Phase 5 — Decentralized Compute

* [ ] Nosana integration
* [ ] GPU training workload
* [ ] Inference workload
* [ ] Compare local vs decentralized execution

### 🔜 Phase 6 — Future Product

* [ ] Model serving
* [ ] API
* [ ] Potential frontend
* [ ] Possible Arweave integration

---

# 🔬 Philosophy

The project is not focused only on getting the highest possible model score.

It is also about understanding how to build a **reproducible ML system**.

That means paying attention to:

🧪 Experiment design
🔍 Leakage and proxies
🧠 Feature engineering
🧱 Reusable data pipelines
✅ Automated testing
📦 Reproducibility
📈 Experiment tracking
🚀 Deployment

---

# 🛠️ Tech Stack

| Area                 | Technology        |
| -------------------- | ----------------- |
| 🐍 Language          | Python            |
| 📊 Data              | pandas, NumPy     |
| 🗃️ Storage          | Parquet / PyArrow |
| 🤖 ML                | scikit-learn      |
| 🧪 Testing           | pytest            |
| 🌿 Version Control   | Git / GitHub      |
| 📈 Planned MLOps     | MLflow            |
| 🐳 Planned Packaging | Docker            |
| 🖥️ Planned Compute  | Nosana            |
| 💾 Possible Storage  | Arweave           |

---

# ⚠️ Disclaimer

This is an independent Machine Learning project using data obtained through the Clash of Clans API.

**Clash of Clans**, Supercell and their related trademarks and intellectual property belong to their respective owners.

This project is **not affiliated with or endorsed by Supercell**.

---

# 📄 License

This project is licensed under the **MIT License**.

See [`LICENSE`](LICENSE) for details.

---

## 🚀 Follow the Project

The project is being developed publicly as part of the **Decentralize AI Hackathon**.

⭐ Star the repository if you want to follow the development.

🔧 The next milestone: **finish the ML experimentation layer and start making the workload portable.**

# Clash of Clans ML Lab

A machine learning laboratory and end-to-end ML pipeline built around real data from the Clash of Clans Developer API.

The project explores multiple supervised and unsupervised machine learning problems using large-scale Clash of Clans data, while applying the complete workflow of a real ML project: data extraction, exploratory data analysis, preprocessing, feature engineering, model training, hyperparameter optimization, evaluation, experiment tracking and result analysis.

---

## 📌 Overview

### What is this project?

**Clash of Clans ML Lab** is a machine learning project built using data obtained from the official Clash of Clans Developer API.

Rather than focusing on a single prediction problem, the project is structured as an **ML laboratory**, where different machine learning approaches are applied to several problems derived from the same domain.

The project includes:

- **4 supervised learning problems**
  - Classification
  - Regression
- **1 unsupervised learning problem**
  - Player clustering
- Exploratory Data Analysis (EDA)
- Data preprocessing and feature engineering
- Multiple machine learning algorithms
- Hyperparameter optimization
- Model comparison and evaluation
- Experiment tracking with MLflow
- Analysis and visualization of results

The goal is not simply to train models, but to understand and implement the complete process required to turn raw data into reproducible machine learning experiments.

### Motivation

Before starting this project, I had already spent time studying the mathematical foundations behind many machine learning algorithms.

I could understand, at least at a theoretical level, what was happening inside algorithms such as regression, classification and clustering. However, there was an important gap between understanding the mathematics and actually implementing a machine learning model from scratch in code without relying on AI to write it for me.

This project was created to close that gap.

Instead of continuing with small datasets and isolated exercises, I wanted to build a more serious project using larger real-world datasets, while forcing myself to understand the implementation side of machine learning:

- How data is transformed into usable features
- How machine learning datasets are constructed
- How models are implemented and trained
- How different algorithms behave on the same problem
- How hyperparameters affect performance
- How to evaluate and interpret models
- How to structure a reproducible ML project

At the same time, I wanted to go beyond the machine learning models themselves and start introducing **MLOps practices** into the project.

For that reason, **MLflow** was integrated to track experiments, parameters, metrics, models and artifacts.

### Project approach

The project combines two perspectives:

**ML Laboratory**

Each problem is treated as an independent experiment. Different algorithms, preprocessing strategies, hyperparameters and evaluation methods are tested and compared.

**End-to-End ML Pipeline**

At the same time, the project follows a complete machine learning workflow:

```text
Clash of Clans API
        ↓
Raw JSON Data
        ↓
Data Extraction & Normalization
        ↓
Exploratory Data Analysis
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
Problem-Specific Datasets
        ↓
Model Training
        ↓
Hyperparameter Optimization
        ↓
Evaluation & Analysis
        ↓
MLflow Experiment Tracking
        ↓
Results & Artifacts

## 🏗️ Project Architecture

The project is organized around a common source of raw data and five independent machine learning pipelines.

All problems start from the same raw Clash of Clans data obtained through the Developer API. From these JSON files, each problem has its own dataset construction process, including the feature engineering required for that specific task.

This means that the project does not rely on a single shared feature engineering pipeline. Instead, each problem defines its own set of features and dataset according to the information required by its machine learning objective.

### Overall Pipeline

The general architecture can be summarized as:

```text
Clash of Clans Developer API
            ↓
        Raw JSON Data
            ↓
   Problem-Specific Dataset
            ↓
     Feature Engineering
            ↓
      Model Training
            ↓
 Hyperparameter Optimization
            ↓
    Evaluation & Analysis
            ↓
      MLflow Tracking
```

The same raw data is used as the foundation for all five problems, but each pipeline transforms it differently depending on the problem being solved.

### Problem-Specific Pipelines

Each problem is implemented independently under its own section of the `models` directory.

A typical problem follows this structure:

```text
Raw JSON Data
      ↓
Build Dataset
      ↓
┌───────────────────────────┐
│ Problem-Specific Features │
│                           │
│ • Clan Features           │
│ • Player Features         │
└───────────────────────────┘
      ↓
Problem Dataset
      ↓
Model Training & Evaluation
      ↓
MLflow
```

The feature engineering is divided into **clan-level and player-level features** when both types of information are relevant to the problem. Problems that only require clan-level information use the corresponding clan features without introducing unnecessary player-level features.

This approach allows each problem to have a dataset specifically designed around its target and available information.

### Independent ML Pipelines

The five problems are intentionally separated rather than being implemented as variations of the same pipeline.

Each problem has its own:

- Dataset construction
- Feature engineering
- Dataset
- Model implementations
- Training scripts
- Evaluation process
- Results and artifacts

This makes it possible to experiment with different approaches independently while keeping the problems reproducible and isolated from one another.

### Experiment Tracking

**MLflow** is integrated into the project as the experiment tracking layer.

During model development, MLflow is used to record:

- Experiment runs
- Model information
- Artifacts
- Parameters and metrics associated with experiments

As the project evolved and more advanced hyperparameter optimization techniques were introduced, such as randomized search and Optuna, MLflow continued to be used to track the relevant experiments and resulting models.

This provides a consistent tracking system across the different machine learning problems.

## 📊 Dataset

The dataset used in this project was built from data collected through the **Clash of Clans Developer API**.

The collection process was designed with a focus on obtaining a sufficiently large and diverse population of clans and players rather than repeatedly querying the same high-ranking clans.

### Data Collection

The crawler first discovers **clans** through the Clash of Clans API and then collects the associated clan members and player profiles.

Instead of relying exclusively on the top clans returned by the API, the crawler uses a **diversified search strategy** based on different combinations of clan filters.

The current search space is defined across three dimensions:

- **Members:** 5 ranges
  - 2–10
  - 11–20
  - 21–30
  - 31–40
  - 41–50
- **Clan level:** 4 ranges
  - 2–5
  - 6–10
  - 11–15
  - 16–20
- **Clan points:** 6 ranges
  - 1–1,000
  - 1,001–3,000
  - 3,001–5,000
  - 5,001–10,000
  - 10,001–40,000
  - 40,001–999,999

The crawler currently generates:

- **15** single-dimension configurations
- **74** two-dimension configurations
- **89 configurations in total**

Three-dimensional combinations are not currently used.

The crawler also maintains a history of previously used configurations. This allows different executions to progressively explore different areas of the search space instead of repeatedly querying the same configurations.

The resulting process is therefore an **incremental exploration strategy**:

```text
Generate Search Configurations
            ↓
Check Search History
            ↓
Select Unused Configurations
            ↓
Query Clash of Clans API
            ↓
Store Retrieved Data
            ↓
Register Configuration as Used
```

The purpose of this approach is not primarily to make the crawler faster. The main bottleneck is the API itself, so the objective is to make each execution more useful by increasing the probability of discovering **new clans and players**.

### Raw Data

The crawler stores the collected information as raw JSON files.

The project uses three types of raw entities:

```text
data/raw/
├── clans/
├── clan_members/
└── players/
```

The crawler is responsible for **data acquisition and storage**, not machine learning preprocessing or feature engineering.

Players are stored as unique entities. Once a player's profile has been downloaded and stored, it is not repeatedly queried to create historical snapshots.

Therefore, the resulting dataset represents the player's state at the time of their first extraction. Dataset diversity is obtained primarily by discovering additional clans and players rather than repeatedly collecting the same player's profile over time.

### Dataset Construction

The raw JSON data is transformed into **problem-specific Parquet datasets** using Python scripts.

Each machine learning problem has its own dataset construction process. The construction step performs the feature engineering required for that particular problem and produces the final dataset consumed by its corresponding ML pipeline.

Conceptually:

```text
Raw JSON Data
      ↓
Problem-Specific Dataset Builder
      ↓
Feature Engineering
      ↓
Problem-Specific Parquet Dataset
      ↓
ML Pipeline
```

Each problem therefore has its own:

- Dataset builder
- Feature engineering
- Selected features
- Target
- Final Parquet dataset

Depending on the problem, the feature engineering can use information from:

- **Clans**
- **Players**
- Or both

This allows each dataset to be specifically designed around the machine learning objective rather than forcing all problems to use the same feature representation.

### Final Datasets

The resulting Parquet datasets are the datasets directly consumed by the five machine learning pipelines:

```text
data/datasets/
├── P1 dataset
├── P2 dataset
├── P3 dataset
├── P4 dataset
└── P5 dataset
```

All five datasets originate from the same underlying raw data, but differ in their feature engineering, selected variables and target depending on the problem they are designed to solve.

The largest dataset is the player clustering dataset, containing approximately **836,000 players**.

The detailed characteristics, distributions and feature engineering of each dataset are analyzed later in the **Exploratory Data Analysis** section.

## 🔎 Exploratory Data Analysis

Exploratory Data Analysis was performed **independently for each machine learning problem** after constructing the corresponding problem-specific datasets.

The purpose of the EDA was not only to visualize the data, but also to understand the distributions and relationships between variables, validate the quality of the processed datasets, and support the decisions made during feature engineering.

The analysis included:

- Feature distributions
- Target distributions
- Missing values
- Potential outliers and extreme values
- Pearson correlation
- Spearman correlation
- Relationships between features and targets
- Problem-specific feature analysis

Because the data preprocessing and dataset construction were performed before the EDA, the final datasets were already in a clean and usable state. As a result, the EDA did not require major corrective preprocessing, but instead served primarily to **understand the resulting data and validate the decisions made during dataset construction**.

### Correlation Analysis

Both **Pearson** and **Spearman** correlations were used to analyze relationships between variables.

This distinction was particularly useful because a high correlation does not automatically mean that a feature should be removed.

In the context of Clash of Clans, many variables are naturally correlated because the game is based heavily on player progression.

For example, a player's **Town Hall level** is naturally related to characteristics such as troop levels, spell levels and other progression-related variables.

Therefore, removing a feature simply because it has a high correlation with another variable could remove meaningful information from the dataset.

The correlation analysis was consequently used as a tool for **understanding the data and identifying potentially problematic relationships**, rather than as an automatic feature-removal mechanism.

### Target and Feature Distributions

The distributions of the features and targets were analyzed separately for each problem.

This allowed the datasets to be inspected in the context of their specific machine learning objective and helped identify:

- Highly concentrated variables
- Skewed distributions
- Extreme values
- Class distributions
- Strong relationships between variables
- Potentially redundant or proxy features

The resulting visualizations are used throughout the individual problem analyses to explain the reasoning behind the final feature sets and modelling decisions.

---

### P2 — Feature and Target Analysis

The Exploratory Data Analysis was particularly important for **Problem 2**, where the objective is to predict `clan_rank`.

One of the most relevant findings was the strong predictive relationship between **`trophies`** and `clan_rank`.

Rather than automatically removing `trophies`, the feature was investigated further to determine whether it represented data leakage or legitimate predictive information.

#### Proxy and Leakage Audit

Before finalizing the P2 dataset, a dedicated audit was performed to identify variables that could potentially act as **proxies for `clan_rank`**.

The analysis considered variables including:

- `trophies`
- `town_hall_level`
- `exp_level`
- `war_stars`
- `attack_wins`
- `defense_wins`
- `donations`
- `donations_received`
- `capital_contributions`
- `builder_base_trophies`

The relationship between each variable and `clan_rank` was analyzed within individual clans using **Spearman correlation**.

The analysis summarized:

- Median correlation
- 90th percentile of absolute correlation
- Percentage of clans with an absolute correlation greater than `0.9`

The purpose was not to automatically remove highly correlated variables. Instead, the audit was used to identify variables that could potentially act as **target proxies** and require further investigation.

For `trophies`, the audit produced:

- Median Spearman correlation: **−0.3963**
- P90 of `|corr|`: **1.0**
- Clans with `|corr| > 0.9`: **17.28%**

These results showed that `trophies` had a strong association with `clan_rank` for a relevant proportion of clans, making it a potential proxy worth investigating.

However, a strong correlation alone was not considered sufficient evidence of data leakage.

#### Trophies: With vs. Without

To measure the actual predictive contribution of `trophies`, two versions of the P2 dataset were created:

- **With `trophies`** — retaining the player trophy-related features.
- **Without `trophies`** — removing the trophy-related features.

This creates a controlled experiment in which the same modelling approach can be evaluated with and without the potentially problematic feature group.

The motivation is particularly relevant to the way the Clash of Clans league system works.

Historically, trophies had a much more direct relationship with clan ranking. However, changes to the league system and the introduction of **league floors** altered this relationship.

Under the current system, trophies still contain predictive information, but they do not necessarily determine clan rank on their own. They can provide additional information for differentiating players within the same league.

At the same time, **Town Hall level** becomes more important because league floors constrain the possible league positions associated with player progression.

The experiment therefore investigates three different aspects:

1. The historical relationship between trophies and clan rank.
2. The current relationship under the league-floor system.
3. The remaining predictive value of trophies as additional information.

The final decision to retain both dataset variants allows the contribution of `trophies` to be evaluated **empirically rather than assumed beforehand**.

Detailed model results and the final interpretation of this experiment are discussed in the **P2 — Clan Rank Regression** section.

---

### EDA Findings

Overall, the EDA confirmed that the processed datasets were suitable for the subsequent modelling stages and provided additional insight into the structure of the data.

Most importantly, it showed that feature selection cannot always be reduced to removing highly correlated variables.

In a progression-based game such as Clash of Clans, many correlations are expected and can represent genuine relationships between player characteristics rather than redundant information.

For this reason, the EDA was used primarily as a **decision-support stage** for the modelling pipelines, with problem-specific decisions documented alongside each individual machine learning problem.

### P1 — Clan Member Role Classification

#### Objective

The objective of **P1** is to predict the role of a player within their clan based primarily on **player-level characteristics**, rather than relying heavily on information that directly describes the clan.

The target variable is:

```text
role
```

The dataset contains four classes:

```text
member
admin
coLeader
leader
```

One peculiarity of the original dataset is the presence of the `admin` category. Since this is not a standard role name used in the current Clash of Clans role system, its behavior within the dataset was analyzed and interpreted as corresponding to the **Elder** role.

The problem was therefore treated as a **four-class classification task**.

#### Features

The feature set combines contextual clan information with a much larger set of player-level characteristics.

Some of the relevant features include:

**Clan context**

- Clan level
- War League
- Clan Capital points
- Other clan-level characteristics

**Player characteristics**

- Town Hall level
- Troop levels and progression
- Hero levels
- Spell levels
- Other player progression and activity features

The objective was to determine whether the role assigned to a player could be inferred from their characteristics and progression within the game.

#### Target Distribution

The target distribution was deliberately controlled during dataset construction so that the classes were as balanced as possible.

The main exception was the `leader` class, since a clan normally has only **one leader**, making it naturally difficult to balance in the same way as the other roles.

![P1 target distribution](path/to/p1_target_distribution.png)

This imbalance became particularly important when evaluating the models.

#### Models

Three classification algorithms were evaluated:

- **Logistic Regression**
- **Random Forest**
- **XGBoost**

The experiments were initially performed using a relatively simple, manual hyperparameter tuning process. Parameters were changed iteratively and the models were retrained to evaluate their effect on performance.

This represented an intentionally straightforward first approach to model optimization, before introducing more systematic hyperparameter search techniques later in the project.

#### Handling the `leader` Class

The main modelling challenge was the `leader` class.

Because there is only one leader per clan, obtaining a perfectly balanced dataset for this class was considerably more difficult than for the other roles.

Simply evaluating the models using overall accuracy would therefore have been misleading. A model could obtain a high accuracy while performing poorly on the minority class.

To address this, **class weighting** was introduced.

The `leader` class was assigned approximately **3–4× more weight** during training, encouraging the models to pay more attention to correctly identifying leaders.

This was particularly interesting when comparing the different algorithms.

#### Evaluation

The models were evaluated using classification metrics calculated **per class**, including:

- Precision
- Recall
- F1-score

The **F1-score** was the main metric used for comparison.

Accuracy alone was not considered sufficient because it could hide poor performance on the `leader` class. Looking at the F1-score for each role provided a much clearer picture of how the models actually behaved across the four classes.

![P1 F1-score by class](path/to/p1_f1_by_class.png)

#### Results

The **Random Forest** achieved the best overall performance, reaching approximately **90% performance across the main classification metrics**.

More importantly, Random Forest was also particularly effective at taking advantage of the additional weight assigned to the `leader` class.

The F1-score comparison shows that the Random Forest was able to maintain strong performance across the classes while handling the more difficult `leader` class better than the other approaches.

![P1 model comparison](path/to/p1_model_comparison.png)

The detailed per-class results are shown in the plots above.

#### Analysis

One of the most interesting observations from P1 was the effect of **class weighting**.

The `leader` class is fundamentally different from the other roles because of its very low natural frequency. Increasing its training weight changed how the models treated this class, but the effect was not identical across algorithms.

Random Forest appeared to make particularly effective use of this weighting strategy, producing a much more competitive F1-score for `leader` while maintaining strong performance on the remaining classes.

This illustrates an important point about classification problems with imbalanced targets: **overall accuracy does not necessarily describe how well a model solves the actual problem**.

For this reason, the per-class F1-score was more informative than accuracy alone.

#### Computational Cost

The experiments also highlighted a practical consideration of model selection.

Some of the Random Forest training runs took approximately **40 minutes** to complete, making iterative experimentation considerably more expensive.

Although this was not a limitation of the final model itself, it became a useful practical lesson about the trade-off between model performance, hyperparameter search and computational cost.

#### Key Takeaways

P1 provided the first complete classification problem in the project and highlighted several important aspects of practical machine learning:

- Class balance needs to be considered when defining evaluation metrics.
- Accuracy can hide poor performance on minority classes.
- Per-class F1-scores provide a more informative evaluation for this problem.
- Class weighting can substantially affect minority-class performance.
- Different algorithms can respond differently to the same weighting strategy.
- Random Forest achieved the strongest overall results, reaching approximately **90% performance**.
- Manual hyperparameter tuning was useful as a first approach, but later problems motivated more systematic optimization methods.

### P2 — Clan Rank Regression

#### Objective

The objective of **P2** is to predict the position a player occupies within their clan based on their progression, characteristics and available clan context.

The target variable is:

```text id="y9tmar"
clan_rank
```

`clan_rank` represents the player's position within the clan, from **1 to 50**.

The central question of this problem was:

> **Can we predict a player's position within their clan from characteristics such as Town Hall level, troops, progression and other player and clan features?**

Two different versions of the dataset were created to investigate how much predictive information was provided by trophy-related variables.

#### Dataset & Features

The same general feature set was used for both experiments, with one important difference: the second version removed all trophy-related variables.

The **with-trophies** dataset contains the complete feature set used for the problem.

The **without-trophies** dataset removes the following variables:

```text id="4g02ez"
trophies
trophies_diff_from_clan_mean
trophies_ratio_to_clan_mean
trophies_clan_pct
best_trophies
progression_ratio_trophies
clan_mean_trophies
required_trophies
```

The remaining features describe different aspects of player progression and clan context, including variables such as:

- Town Hall level
- Player progression
- Troop levels
- Hero progression
- Spell progression
- Clan characteristics
- Other player and clan attributes

This allowed the model to attempt to infer clan position without directly accessing the information most closely related to trophies.

#### Why Two Versions?

The decision to create two datasets was motivated by the relationship between **trophies** and **clan rank**.

Before the main EDA, a dedicated proxy analysis was performed to investigate whether trophy-related variables were simply highly correlated with the target or represented a form of data leakage.

The analysis showed that trophies provided significant predictive information, but the relationship was not treated as automatic evidence of leakage.

Instead of simply removing the variables, the project used an empirical approach:

```text id="u6pra9"
Potentially Strong Proxy
        ↓
Correlation / Proxy Analysis
        ↓
Not Treated as Automatic Leakage
        ↓
Create Two Dataset Variants
        ↓
With Trophies vs. Without Trophies
        ↓
Compare Model Performance
```

This makes it possible to measure how much predictive performance is actually lost when trophy-related information is removed.

#### Models

Three regression approaches were evaluated:

- **Ridge Regression**
- **Random Forest**
- **XGBoost**

The initial experiments provided a baseline comparison between different model families.

As the experiments progressed, **Optuna** was introduced for systematic hyperparameter optimization of XGBoost.

Random Forest was considerably more computationally expensive to train, making extensive hyperparameter optimization less practical. XGBoost therefore became the main candidate for automated optimization.

#### Hyperparameter Optimization

Optuna was used to search for better XGBoost configurations rather than relying exclusively on manually selected parameters.

The optimization process was used to identify the strongest XGBoost configuration for the regression task.

This became an important point in the development of the project because P2 was where Optuna became particularly useful for finding strong configurations without manually testing parameters one by one.

The resulting **Optuna-optimized XGBoost model** was the best-performing approach in both dataset variants.

#### Evaluation

The regression models were evaluated using:

- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **R²**

MAE provides an interpretable measure of the average prediction error in ranking positions, while RMSE gives greater weight to larger errors.

R² was used to measure how much of the variance in `clan_rank` was explained by the model.

#### Results — With Trophies

The best-performing model was **XGBoost optimized with Optuna**.

| Metric | XGBoost + Optuna |
|---|---:|
| MAE | **2.0537** |
| RMSE | **3.3092** |
| R² | **0.9134** |

For comparison, the Random Forest v2 model achieved:

| Metric | Random Forest v2 |
|---|---:|
| MAE | 2.0882 |
| RMSE | 3.4162 |
| R² | 0.9077 |

The difference between the two models was relatively small, but XGBoost achieved the strongest overall results.

#### Results — Without Trophies

Removing the trophy-related variables reduced performance, but the models retained substantial predictive capability.

The best-performing model was again **XGBoost + Optuna**:

| Metric | XGBoost + Optuna |
|---|---:|
| MAE | **2.5287** |
| RMSE | **3.8863** |
| R² | **0.8805** |

Random Forest v2 achieved:

| Metric | Random Forest v2 |
|---|---:|
| MAE | 2.5836 |
| RMSE | 3.9731 |
| R² | 0.8751 |

Ridge Regression performed substantially worse:

| Metric | Ridge Regression |
|---|---:|
| MAE | 4.1836 |
| RMSE | 5.5754 |
| R² | 0.7541 |

#### With vs. Without Trophies

The comparison between the two dataset variants shows a clear but relatively moderate performance difference.

| Dataset | Best Model | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| **With trophies** | XGBoost + Optuna | **2.0537** | **3.3092** | **0.9134** |
| **Without trophies** | XGBoost + Optuna | **2.5287** | **3.8863** | **0.8805** |

Removing trophy-related features therefore resulted in:

- Higher MAE
- Higher RMSE
- Lower R²

However, the model still achieved an **R² of 0.8805** without any of the explicitly trophy-related variables.

This suggests that trophies provide meaningful predictive information, but they are **not the only source of information available for predicting clan rank**.

#### Why Are Trophies Predictive?

The importance of trophies was not particularly surprising given their role in the game's league system.

Players compete within a **league system**, and their league is related to their progression and, in particular, their Town Hall level. The current system also introduces **trophy floors depending on Town Hall level**.

Players then compete within their respective leagues, with their trophy count providing additional information about their relative position.

The trophy count is also part of a **weekly cycle**: after the league period, trophies are reset and players begin competing again in the next cycle.

Therefore, Town Hall level provides important contextual information about the player's league, while trophies provide additional information that can help distinguish players within that context.

Conceptually:

```text id="rutr5y"
Town Hall / Player Progression
            ↓
      League Context
            ↓
         Trophies
            ↓
    Relative Position
            ↓
        Clan Rank
```

This helps explain why both **Town Hall-related features** and **trophy-related features** can be highly predictive of clan rank without trophies necessarily representing data leakage.

#### Feature Importance

A feature-importance analysis is planned as part of the refactoring of the P2 results analysis.

The objective will be to analyze which variables contributed most to the predictions, particularly for the optimized XGBoost models.

![P2 feature importance](path/to/p2_feature_importance.png)

#### Analysis

The main finding from P2 is that **removing a highly predictive group of variables does not necessarily make a model useless**.

Trophy-related variables clearly improve performance, but the remaining player and progression features still contain enough information for XGBoost to predict clan rank with an R² of approximately **0.88**.

The comparison also reinforced the importance of distinguishing between **correlation and data leakage**.

A variable being strongly correlated with a target does not automatically mean that it should be removed. Instead, the relationship should be understood in the context of how the target and feature are generated.

In this case, rather than assuming that trophies represented leakage, the project tested both possibilities empirically.

The results showed that trophies provide useful information, but the model is still capable of learning substantial information about clan rank from other characteristics.

#### Model Selection

XGBoost was the strongest model across both dataset variants.

This experiment also reinforced an observation from the project: **XGBoost performed particularly well on the regression problems**, while Random Forest showed stronger characteristics in the classification experiments.

The use of Optuna made XGBoost especially attractive because it allowed systematic hyperparameter optimization without the same level of manual experimentation required during the earlier stages of the project.

#### MLflow & Model Reproducibility

P2 also exposed an important ML engineering issue when loading trained models from MLflow.

A model cannot be reliably reused simply because the model file itself has been saved. The feature representation used during inference must also remain consistent with the representation used during training.

During the development of P2, a **feature mismatch** was encountered when loading a trained model.

The problem was addressed by preserving and validating the model's expected feature information, using the information available through the trained model and MLflow, including:

- Model signatures
- Feature names
- Number of expected features
- Stored model information

This ensured that the model could be loaded and evaluated using the same feature representation expected during training.

This became an important practical lesson about **model reproducibility and feature alignment**.

#### Key Takeaways

P2 highlighted several important lessons from both the machine learning and engineering perspectives:

- Strong correlation does not automatically mean data leakage.
- Potential proxy variables should be investigated rather than removed blindly.
- Comparing models with and without a potentially dominant feature group can quantify its actual contribution.
- Trophy-related variables significantly improve clan-rank prediction, but they are not essential for achieving strong predictive performance.
- XGBoost + Optuna achieved the best results in both dataset variants.
- The model achieved an R² of **0.9134 with trophies** and **0.8805 without trophies**.
- Optuna provided a more systematic approach to XGBoost optimization.
- Feature alignment is essential when saving and loading machine learning models.
- A model artifact alone is not enough for reliable inference; the expected feature representation must also be preserved.

### P3 — Clan War Performance Regression

#### Objective

The objective of **P3** is to investigate whether a clan's war performance can be explained by its characteristics and overall profile.

Rather than attempting to predict a clan's **future** war performance, the problem uses the clan's existing war history to construct a performance metric and then investigates how well that metric can be explained by the available features.

The target variable is:

```text id="m3q7vx"
war_success_rate
```

It is calculated as:

```python
war_success_rate = war_wins / war_total
```

Therefore, the target represents the clan's **historical war success rate at the time represented by the dataset**.

The central question was:

> **Can a clan's war performance be explained by its characteristics and available clan-level information?**

This makes P3 a **regression problem**, where the model attempts to estimate a continuous performance measure rather than a discrete class.

#### Dataset Construction

The dataset was intentionally restricted to clans with a sufficiently large war history.

Clans with too few recorded wars were filtered out before calculating the target:

```python
valid_clans = clans[clans["war_total"] >= min_war_total].copy()
```

The target was then calculated as:

```python
valid_clans["war_success_rate"] = (
    valid_clans["war_wins"].astype(float)
    / valid_clans["war_total"].replace(0, np.nan)
)
```

Rows for which a valid success rate could not be calculated were subsequently removed.

This filtering resulted in a smaller dataset than some of the other problems in the project, but provided a more reliable basis for measuring historical war performance.

The final dataset contains:

```text id="8v5cnd"
31,289 rows
51 columns
```

After removing identifiers, metadata and the target variable, **44 features** were used for model training.

#### Features

P3 focuses primarily on **general clan-level characteristics**.

The features describe different aspects of the clan and its composition, allowing the models to investigate whether a clan's overall profile contains enough information to explain differences in historical war success.

The dataset intentionally focuses on information available about the clan rather than introducing future war results as predictive variables.

#### Models

Three regression approaches were evaluated:

- **Linear Regression**
- **Random Forest**
- **Gradient Boosting**

The models were introduced progressively to determine whether increasingly flexible approaches could capture relationships that a simple linear model could not.

#### Linear Regression Baseline

Linear Regression was used as the initial baseline.

The purpose was not necessarily to obtain the best possible model, but to establish how much of the target could be explained using a simple linear relationship between the features and `war_success_rate`.

The baseline achieved:

| Metric | Linear Regression |
|---|---:|
| MAE | 0.1071 |
| RMSE | 0.1366 |
| R² | 0.0764 |

The relatively low R² indicated that the problem was considerably more difficult than some of the other regression tasks in the project.

This provided a useful reference point for evaluating whether more complex models could extract additional information from the same features.

#### Random Forest

Random Forest was then introduced as a non-linear alternative.

The motivation was to investigate whether the distribution of the data across decision-tree leaves could capture relationships that Linear Regression could not model.

Randomized hyperparameter search was used to improve the model configuration.

The resulting Random Forest achieved:

| Metric | Random Forest |
|---|---:|
| MAE | 0.0986 |
| RMSE | 0.1262 |
| R² | 0.2113 |

The improvement over Linear Regression confirmed that non-linear relationships were present in the dataset.

However, the overall predictive power remained relatively limited.

#### Gradient Boosting + Optuna

After observing that tree-based models performed better than the linear baseline, **Gradient Boosting** was selected as the next approach.

Gradient Boosting was chosen because boosting methods are well suited to regression problems and can progressively improve predictions by focusing on the errors made by previous trees.

It also provided an opportunity to explore a different boosting approach after having relied heavily on XGBoost in other parts of the project.

The Gradient Boosting model was optimized using **Optuna**, with **30 trials** used to search for a strong hyperparameter configuration.

The resulting model became the best-performing model in P3.

| Metric | Gradient Boosting + Optuna |
|---|---:|
| MAE | **0.0980** |
| RMSE | **0.1255** |
| R² | **0.2206** |

#### Model Comparison

The progression of the models can be summarized as:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 0.1071 | 0.1366 | 0.0764 |
| Random Forest | 0.0986 | 0.1262 | 0.2113 |
| **Gradient Boosting + Optuna** | **0.0980** | **0.1255** | **0.2206** |

The results show a clear improvement when moving from a linear model to tree-based approaches.

However, the difference between Random Forest and Gradient Boosting was relatively small. The final R² of **0.2206** indicates that the model could capture some of the structure in the target, but a large amount of the variability in war success remained unexplained.

#### Feature Importance

Feature importance analysis provided additional insight into what the models were learning.

One of the most prominent features was **clan rank**.

![P3 feature importance](path/to/p3_feature_importance.png)

This result is intuitive within the context of the game.

A clan with a higher overall rank is generally associated with a more established progression history, which can also be related to the amount of experience accumulated through wars and the number of wars won over time.

The feature importance analysis therefore provided a useful connection between the model's behavior and the underlying domain.

#### Analysis

P3 produced a substantially different result from P2.

While P2 achieved a very high R², P3 showed that predicting a clan's historical war success rate from its available characteristics was considerably more difficult.

The progression of the experiments demonstrates this clearly:

```text
Linear Regression
R² = 0.0764
        ↓
Random Forest
R² = 0.2113
        ↓
Gradient Boosting + Optuna
R² = 0.2206
```

The non-linear models were clearly better than the linear baseline, but increasing model complexity did not produce a dramatic improvement.

This suggests that the available clan-level characteristics contain **some information about war performance, but not enough to accurately explain all of its variability**.

The result was somewhat surprising during development, as the expectation was that war performance would be easier to model from the available clan characteristics.

One possible source of additional variability in Clash of Clans is the way clans actually approach wars. For example, some communities use practices such as **Friendly War Arrangements (FWAs)**, where war outcomes can be deliberately coordinated for purposes such as resource farming. However, there is no evidence from this experiment that such behavior was the primary explanation for the model's limited predictive performance.

The main conclusion was therefore not that the dataset contained a specific hidden factor, but that **war performance is simply a difficult quantity to explain from the available features**.

#### What P3 Demonstrated

P3 provided an important lesson about the limits of machine learning.

A more complex model does not automatically turn a difficult prediction problem into an accurate one.

Even after moving from Linear Regression to Random Forest and then to an optimized Gradient Boosting model, the final R² remained relatively modest.

The experiment therefore demonstrated that:

- Some problems contain substantially more predictable structure than others.
- Non-linear models can improve performance over simple baselines.
- Hyperparameter optimization can produce measurable improvements.
- However, model complexity cannot compensate indefinitely for limited predictive information in the features.
- Not every problem is equally suitable for highly accurate machine learning predictions.

This was one of the main reasons for moving the project's attention toward **P4**, which reframed the problem as a classification task.

#### Key Takeaways

P3 highlighted several important lessons from the project:

- The objective was to analyze **historical clan war performance**, not predict future war outcomes.
- The `war_success_rate` target was constructed as `war_wins / war_total`.
- Filtering clans by minimum war history helped avoid unreliable targets based on very few wars.
- Tree-based models significantly outperformed the Linear Regression baseline.
- Gradient Boosting + Optuna achieved the best results.
- The final model achieved an **R² of 0.2206**, showing that the problem remained difficult despite using non-linear models and hyperparameter optimization.
- Feature importance identified **clan rank** as one of the strongest predictors.
- The experiment demonstrated that **not every real-world problem contains enough predictable information for machine learning to produce highly accurate results**.
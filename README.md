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

(falta ridge con trophies avisar a alvaro al leer esto)

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

### P4 — Clan Performance Classification

#### Objective

The objective of **P4** is exactly the same as in P3: to investigate whether a clan's war performance can be explained by its characteristics and available clan-level information.

The difference is how the problem is formulated.

While **P3 treats `war_success_rate` as a continuous target and approaches the problem as regression**, P4 transforms the same underlying performance measure into three discrete classes:

```text
low
medium
high
```

The central question therefore remains:

> **Can a clan's war performance be classified based on its characteristics and available clan-level information?**

This reformulation was motivated by the difficulty observed in P3. Predicting an exact continuous value for `war_success_rate` proved to be a relatively difficult task, so P4 investigates whether the problem becomes more tractable when performance is divided into meaningful categories.

#### Dataset Construction

P4 uses the **same underlying dataset and feature set as P3**.

However, P4 has its own dataset because the target variable is transformed from the continuous `war_success_rate` used in P3 into a categorical variable called:

```text
performance_class
```

The transformation is performed using two configurable thresholds:

```text
war_success_rate < low_threshold
        ↓
       low

low_threshold ≤ war_success_rate < high_threshold
        ↓
     medium

war_success_rate ≥ high_threshold
        ↓
      high
```

The classification dataset is therefore constructed from the P3 regression dataset while preserving the same predictive information.

The only fundamental difference is the target representation:

```text
P3
war_success_rate
      ↓
Continuous value
      ↓
Regression
```

```text
P4
war_success_rate
      ↓
low / medium / high
      ↓
Classification
```

As in P3, variables that directly represent war outcomes are excluded from the predictive features. This includes variables such as:

```text
war_wins
war_losses
war_ties
war_win_streak
war_points
war_total
war_success_rate
win_rate
loss_rate
tie_rate
```

This prevents the models from directly using the information from which the target performance was constructed.

#### Target Distribution

Unlike P1, the class distribution in P4 was deliberately balanced.

This was possible because the class boundaries could be controlled through the thresholds used to transform `war_success_rate`.

The resulting distribution was approximately:

| Class | Proportion |
|---|---:|
| Medium | 33.42% |
| Low | 33.32% |
| High | 33.25% |

This was an intentional design decision.

In P1, the `leader` class was naturally much rarer because a clan can only have one leader. In P4, the target definition itself could be adjusted, so the classes were balanced to avoid introducing an unnecessary class-imbalance problem.

With three approximately equally represented classes, a random classifier would achieve an accuracy of roughly **33.3%**, providing a useful baseline for interpreting the results.

#### Models

Three classification approaches were evaluated:

- Random Forest
- XGBoost
- Multi-Layer Perceptron (MLP)

Random Forest and XGBoost were included as tree-based approaches already used elsewhere in the project.

The **MLP** was introduced specifically in P4 to explore a different modelling approach.

Rather than choosing it because it was expected to perform better beforehand, it was added after investigating whether a neural-network-based classifier could be integrated into the existing workflow using **scikit-learn and Optuna**, without requiring a separate PyTorch implementation.

This provided another perspective on the same problem and expanded the range of models evaluated in the ML Lab.

#### Hyperparameter Optimization

Hyperparameter optimization was performed using **Optuna**.

Both XGBoost and MLP were given **50 optimization trials**.

The purpose of using 50 trials was not only to search for better hyperparameters, but also to investigate whether increasing the number of trials could provide a meaningful improvement compared with the smaller searches used in other problems.

This was part of evaluating the practical trade-off between **optimization effort and model performance**.

Random Forest was evaluated separately without the same 50-trial Optuna search.

#### Evaluation

Because the dataset contains three balanced classes, several classification metrics were used:

- Accuracy
- Balanced Accuracy
- Macro F1
- Weighted F1

**Macro F1** is particularly useful for this problem because it gives equal importance to all three classes instead of allowing the overall score to be dominated by the most frequent class.

#### Results

| Model | Accuracy | Balanced Accuracy | F1 Macro | F1 Weighted |
|---|---:|---:|---:|---:|
| Random Forest | 0.4811 | 0.4811 | 0.4773 | 0.4775 |
| XGBoost + Optuna | 0.4986 | 0.4985 | 0.4925 | 0.4926 |
| **MLP + Optuna** | **0.5037** | **0.5038** | **0.4962** | **0.4962** |

The **MLP + Optuna** configuration achieved the best overall results, although the difference between the three models was relatively small.

With approximately balanced classes, the random baseline is around 33.3%. The best model achieved approximately **50.4% accuracy**, indicating that the models were able to capture meaningful information about clan performance, while also showing that the problem remained difficult.

#### Analysis

P4 produced substantially better results than the regression formulation explored in P3.

This suggests that, for this particular problem, predicting an exact value of `war_success_rate` is considerably more difficult than determining whether a clan belongs to a lower, intermediate or higher performance category.

The introduction of the **medium** class also makes the problem more nuanced.

The confusion matrices show that the models can identify the **low** and **high** classes relatively well, while the main difficulty is distinguishing the **medium** class from the two extremes.

This behaviour is intuitive: a clan near the boundary between performance categories can share characteristics with both neighbouring classes, making it harder for the model to assign a clear label.

Therefore, the classification formulation does not simply make the problem "easy". Instead, it changes the type of uncertainty the model has to deal with.

#### From Regression to Classification

The comparison between P3 and P4 demonstrated an important modelling consideration:

> **When an exact continuous target is difficult to predict, reformulating the problem into meaningful categories can make the task more tractable.**

Instead of asking the model to predict an exact value such as:

```text
0.63
```

the problem can ask whether the clan belongs to:

```text
low
medium
high
```

This also allows additional intermediate classes to be introduced when a simple binary distinction would be too coarse.

In P4, the `medium` category provides information about clans whose performance is neither clearly low nor clearly high, although this also makes classification more difficult.

#### What P4 Demonstrated

P4 demonstrated that **problem formulation can be as important as model selection**.

Using the same underlying predictive information as P3 but changing the target representation produced a considerably more tractable machine learning problem.

The results also showed that adding more sophisticated models does not necessarily produce dramatically different performance. MLP achieved the best result, but only by a relatively small margin over XGBoost.

The problem therefore remains limited by the predictive information available in the dataset rather than simply by the choice of algorithm.

#### Key Takeaways

- P4 uses the **same underlying data and predictive features as P3**, but creates a separate classification dataset by replacing `war_success_rate` with `performance_class`.
- The target is divided into **low, medium and high** performance classes using configurable thresholds.
- The three classes were deliberately balanced to avoid unnecessary class-imbalance issues.
- Random Forest, XGBoost and MLP were evaluated.
- XGBoost and MLP were optimized with **50 Optuna trials**.
- **MLP + Optuna** achieved the best result with **50.37% accuracy and 49.62% macro F1**.
- The approximately 50% accuracy is meaningfully above the ~33.3% random baseline for three balanced classes.
- Low and high performance were easier to classify than the medium class.
- Reformulating a difficult regression problem as a classification problem can make the target more tractable.
- The experiment reinforced the importance of **choosing an appropriate problem formulation**, rather than assuming that a more complex model will always solve a difficult prediction problem.

### P5 — Player Clustering

#### Objective

The objective of **P5** is to investigate whether Clash of Clans players can be divided into different groups based on their characteristics.

Unlike P1–P4, there is no predefined target variable. Instead, the goal is to discover whether the dataset contains naturally occurring patterns that can be used to identify different types of players.

The central question was:

> **What types of players can be identified based on their characteristics and progression?**

This makes P5 an **unsupervised learning problem**, where the models are not given predefined labels and must instead discover structure within the data.

Some of the most informative characteristics include variables such as:

- Town Hall level
- Player progression
- League and trophy-related statistics
- Donations and donation ratios
- War activity
- Clan Capital contributions
- Troop, hero, spell and equipment progression

These features capture different aspects of a player's level of progression and activity within the game.

#### Dataset

P5 uses the largest dataset in the project, containing approximately **836,830 players**.

The clustering dataset contains **31 columns**, of which **30 features** are used for clustering.

The features describe different aspects of player progression, activity and gameplay.

The dataset does not contain a target variable because the objective is to discover the groups directly from the feature space.

An additional analysis showed **370 duplicated feature rows**. These were retained because they can represent different players who happen to have exactly the same values across all clustering features. Removing them would therefore remove valid player observations rather than simply removing accidental duplicate records.

#### Preprocessing

Before applying the clustering algorithms, the features were standardized using `StandardScaler`.

This step was particularly important because the dataset contains variables with very different numerical scales.

For example, features such as:

```text
town_hall_level
donation_ratio
clan_capital_contributions
best_trophies
```

operate on completely different scales.

Distance-based clustering algorithms are sensitive to these differences. Without scaling, variables with larger numerical ranges could dominate the distance calculations and disproportionately influence the resulting clusters.

Therefore, the features were standardized before applying the clustering algorithms.

#### K-Means

K-Means was used as the main clustering approach.

Different values of **K from 2 to 10** were evaluated to determine how the structure of the dataset changed depending on the number of clusters.

The **Silhouette Score** was used to evaluate the resulting cluster structures.

Because calculating the Silhouette Score over the complete dataset would be computationally expensive and would provide little additional practical value for this experiment, a sample of **50,000 players** was used for the metric calculation.

This made the evaluation substantially more manageable while still providing a representative estimate of cluster quality.

The best configuration was:

| Parameter | Value |
|---|---:|
| Number of clusters | **2** |
| Silhouette Score | **0.3926** |

The resulting clusters contained:

| Cluster | Players |
|---|---:|
| Cluster 0 | 404,225 |
| Cluster 1 | 432,605 |

The relatively balanced cluster sizes indicate that K-Means did not simply isolate a small group of unusual players while assigning almost everyone else to a single cluster.

#### Cluster Analysis

The two clusters can be interpreted as two broad player profiles.

**Cluster 0 — Higher progression and activity**

The first cluster contains players with substantially higher values across many progression and activity-related features.

For example, compared with the global mean, this cluster has considerably higher:

- Town Hall level
- Experience level
- Best trophies
- War stars
- Donations
- Clan Capital contributions
- Troop levels and counts
- Hero levels
- Spell and equipment progression
- Combat activity
- Donation ratio

For example:

| Feature | Cluster 0 | Cluster 1 |
|---|---:|---:|
| Town Hall level | 16.16 | 10.46 |
| Experience level | 202.77 | 84.61 |
| Best trophies | 4361.11 | 1578.34 |
| War stars | 1420.74 | 205.21 |
| Donations | 215.89 | 14.07 |
| Clan Capital contributions | 1,812,094 | 144,682 |
| Donation ratio | 0.853 | 0.083 |
| Hero mean level | 49.97 | 19.38 |
| Troop count | 72.54 | 42.33 |

This cluster can therefore be broadly interpreted as containing **more progressed and/or more active players**.

**Cluster 1 — Lower progression and activity**

The second cluster shows substantially lower values across most progression and activity-related features.

For example, it has lower:

- Town Hall level
- Experience level
- War activity
- Donations
- Clan Capital contributions
- Troop and hero progression
- Equipment progression
- Combat activity
- Donation ratio

It can therefore be broadly interpreted as a group of **less progressed and/or less active players**.

An interesting exception is the current trophy count, where Cluster 1 has a slightly higher mean than Cluster 0. This highlights that player profiles cannot necessarily be reduced to a single progression variable such as trophies.

The clustering instead captures a broader combination of progression and activity characteristics.

#### DBSCAN

DBSCAN was introduced to investigate whether the dataset contained more irregular or differently shaped groups that K-Means might not capture effectively.

Unlike K-Means, DBSCAN does not require the number of clusters to be specified in advance and can identify observations considered to be noise.

A total of **21 configurations** were evaluated.

The best configuration obtained:

| Parameter | Value |
|---|---:|
| `eps` | 0.3 |
| `min_samples` | 20 |
| Number of clusters | 160 |
| Noise points | 769,743 |
| Noise proportion | 91.98% |
| Silhouette Score | 0.1373 |

Although DBSCAN identified 160 different groups, the result was not considered successful because approximately **92% of the players were classified as noise**.

The purpose of using DBSCAN was to investigate whether it could discover more diverse player groups than K-Means. Instead, the algorithm produced a highly fragmented structure while leaving the vast majority of the dataset outside the clusters.

This suggests that the dataset did not fit the assumptions of DBSCAN particularly well at this scale and parameterization.

The PCA visualization also reinforced this conclusion. Rather than identifying dense regions as clusters and isolated observations as noise, DBSCAN classified a very large proportion of the dataset as noise.

Therefore, DBSCAN was considered a **negative result** for this particular dataset.

#### Agglomerative Clustering

Agglomerative Clustering was also investigated as another alternative clustering approach.

However, applying it to the complete dataset revealed a major computational limitation.

The dataset contains more than **836,000 players**, making the number of pairwise relationships extremely large. Running the algorithm on the full dataset would require approximately **2.55 TiB of memory**, making the experiment impractical on the available hardware.

The issue was therefore not an implementation error, but a consequence of the computational requirements of applying the algorithm to a dataset of this scale.

Although code for Agglomerative Clustering was implemented, the full experiment was not executed.

Reducing the dataset to a smaller subset was considered, but selecting a representative subset would introduce another problem: the dataset contains many dimensions and player profiles with substantial variation, making the process of manually selecting a smaller group of players representative of the complete population unnecessary for the objective of this experiment.

For this reason, Agglomerative Clustering was left as a documented scalability limitation rather than forcing an experiment on an arbitrarily reduced dataset.

#### PCA Visualization

PCA was used to visualize the clustering structure and provide a lower-dimensional representation of the feature space.

The PCA visualization of the K-Means results supported the interpretation obtained from the cluster statistics, showing the separation between the two broad player profiles.

The DBSCAN visualization was particularly informative because it showed the limitations of the resulting clustering.

A more useful DBSCAN result would have been expected to identify dense regions of players as clusters while treating relatively isolated observations as noise. Instead, the configuration classified most of the dataset as noise.

This provided a visual confirmation that the DBSCAN result was not useful for describing the player population.

#### Model Comparison

The three approaches provided very different outcomes:

| Method | Result |
|---|---|
| **K-Means** | 2 meaningful and relatively balanced clusters |
| **DBSCAN** | 160 clusters but ~91.98% of players classified as noise |
| **Agglomerative** | Not feasible on the complete dataset due to memory requirements |

K-Means was therefore the most useful approach for this dataset.

The results also demonstrate that clustering algorithms can behave very differently when applied to the same feature space. A method producing more clusters is not necessarily producing a better representation of the underlying data.

#### Analysis

The main result of P5 is that the player population can be divided into two broad groups based on a combination of progression and activity-related characteristics.

The clustering did not simply separate players according to one variable. Instead, the differences appear across many dimensions simultaneously.

Features such as **Town Hall level** provide an indication of game progression, while variables such as **donation ratio**, donations, war activity and Clan Capital contributions provide additional information about player activity.

The resulting clusters therefore represent broader player profiles rather than a single-dimensional ranking.

P5 also demonstrated the importance of evaluating unsupervised learning results beyond simply looking at the number of clusters produced.

DBSCAN generated **160 clusters**, which might initially appear more informative than K-Means producing only two. However, the fact that almost 92% of observations were classified as noise made the result substantially less useful.

Similarly, Agglomerative Clustering showed that an algorithm can be theoretically applicable to a problem while still being impractical at the scale of the available dataset.

#### What P5 Demonstrated

P5 demonstrated a different side of machine learning compared with the previous problems.

In supervised learning, the target is already defined and the model attempts to learn a relationship between the features and that target.

In unsupervised learning, there is no predefined concept of what is "good" or "bad". The objective is instead to discover patterns and structure within the data.

This makes the process more difficult to evaluate, but also more exploratory.

The clustering results revealed patterns in the player population that were not explicitly defined beforehand. The distinction between more progressed and active players and less progressed or active players emerged from the combination of multiple features rather than from a manually assigned label.

The experiment also reinforced that **algorithm choice must consider both the structure and scale of the dataset**.

K-Means produced a useful result, DBSCAN did not fit the dataset particularly well, and Agglomerative Clustering was computationally impractical on the complete population.

#### Key Takeaways

- P5 is the project's **unsupervised learning problem**.
- The objective is to discover different player profiles based on their characteristics.
- The dataset contains approximately **836,830 players and 30 clustering features**.
- Features were standardized using `StandardScaler` because the algorithms rely on distances and the original variables operate on very different scales.
- K-Means was evaluated for **K = 2–10** using a 50,000-player sample for Silhouette Score calculation.
- **K = 2** achieved the best Silhouette Score of approximately **0.3926**.
- The two clusters were relatively balanced and can be broadly interpreted as more progressed/active and less progressed/active player profiles.
- DBSCAN produced **160 clusters**, but approximately **91.98% of players were classified as noise**, making it an unsuccessful approach for this dataset.
- Agglomerative Clustering was not feasible on the complete dataset because of its extreme memory requirements.
- PCA provided a useful visual representation of the clustering structure and highlighted the differences between K-Means and DBSCAN.
- Unsupervised learning is more exploratory because there are no predefined labels telling the model what constitutes a good or bad group.
- The experiment showed that discovering patterns can be more ambiguous than supervised prediction, but also more interesting because meaningful structures can emerge directly from the data.

## 📈 Results & Findings

> **Note:** This section will be completed once all experiments and results have been fully reviewed and the reported metrics are consistent across the project.
>
> The final version will include a consolidated comparison of the five machine learning problems, their best-performing models, key findings and the main lessons learned from the experiments.

## 🔬 Machine Learning Engineering

Although the project is primarily focused on machine learning experimentation, it also introduces several machine learning engineering practices to make the experiments easier to reproduce, compare and analyze.

The engineering layer is intentionally lightweight. It is not designed as a production MLOps system, but as a practical and reproducible experimentation workflow.

### Experiment Tracking with MLflow

**MLflow** has been part of the project from the beginning and became one of the most useful components of the entire workflow.

Instead of treating each training run as an isolated experiment, MLflow provides a centralized way to record what was done and compare the results afterwards.

The experiments track:

- Parameters
- Metrics
- Trained models
- Artifacts
- Dataset context
- Target information
- Dataset split configuration

This makes it possible to inspect an experiment without having to remember exactly which configuration was used during training.

The local MLflow setup uses **SQLite as the backend store** and local artifact storage:

```text
MLflow Server
      │
      ├── SQLite backend
      │       └── mlflow.db
      │
      └── Artifact storage
              └── mlflow/mlruns/
```

MLflow runs locally through:

```text
127.0.0.1:5000
```

### Dataset Context

One particularly useful part of the tracking system was recording the **dataset context** for every experiment.

Although all five problems originate from the same raw Clash of Clans data, they do not use the same final dataset.

Each problem has its own:

- Feature engineering
- Selected features
- Target
- Dataset construction process

Recording this information in MLflow makes it possible to understand exactly which dataset and target were used for each experiment.

This becomes particularly important when comparing experiments across P1–P5, where datasets may originate from the same raw data but represent fundamentally different machine learning problems.

The dataset split configuration is also recorded. This provides additional context when reviewing results and makes it possible to verify how the data was divided during training.

### Reproducibility

The experiments consistently use:

```python
random_state = 42
```

where applicable.

The same general split configuration is therefore maintained across the experiments, while the split information itself is also logged in MLflow.

This provides a consistent experimental setup and makes it easier to reproduce and compare results.

The project is therefore designed so that the machine learning experiments can be executed again on another machine using the same code, datasets and configuration.

### Reusable MLflow Tracking Layer

The project uses a reusable layer around MLflow to avoid having to implement the same tracking logic independently for every problem.

The tracking functionality covers operations such as:

```text
configure_tracking
mlflow_run
log_dataset_context
log_split_config
log_model_params
log_metrics
log_model_and_artifacts
```

This provides a common tracking workflow across the different supervised and unsupervised experiments.

As new models were introduced, the tracking layer was also extended when necessary. For example, support for the **MLP** experiment was integrated into the existing MLflow workflow.

Some issues with experiment metadata were also identified and corrected during development, such as ensuring that split configuration and dataset context were correctly stored in the early experiments.

### Model Logging

During the initial stages of the project, models were generally logged during each experiment run.

As the project introduced **Optuna** for hyperparameter optimization, the workflow evolved.

Instead of logging every intermediate model generated during the optimization process, the final selected model was logged after the optimization process had identified the best configuration.

This reduced unnecessary model artifacts while keeping the final model associated with the corresponding experiment.

### Model Loading and Feature Consistency

One practical issue encountered during the project involved loading a trained XGBoost model for later analysis.

One model had been trained using a different number of features than the feature set expected when the results were later generated. This resulted in a feature mismatch when attempting to load and use the model.

The model was subsequently retrained with the correct feature set, resolving the issue.

This highlighted an important practical aspect of machine learning engineering:

> **A trained model is not independent from the feature schema used to train it.**

The model, its expected features and the dataset used during inference need to remain aligned.

MLflow model information and signatures were also used to help validate this consistency.

### Hyperparameter Optimization with Optuna

As the project evolved, **Optuna** was introduced to perform more systematic hyperparameter optimization.

Before using Optuna, some experiments relied more heavily on manually changing parameters and observing how the models behaved.

This was useful from a learning perspective because changing parameters manually made it easier to understand how different hyperparameters affected the models rather than treating the algorithm as a complete black box.

Optuna introduced a more systematic search process.

Instead of manually selecting every configuration, Optuna evaluates different configurations and progressively searches for better-performing regions of the hyperparameter space.

For example, P4 used **50 trials** for both XGBoost and MLP to investigate whether increasing the optimization budget could produce meaningful improvements.

Optuna therefore served two purposes in the project:

1. **Practical optimization** — efficiently searching a large hyperparameter space.
2. **Understanding optimization behaviour** — observing how different configurations affected model performance and trying to understand why certain configurations performed better.

This distinction was important during the project. Hyperparameter optimization is most useful when the practitioner understands what the hyperparameters control and can interpret the behaviour observed during the search, rather than simply treating the optimization process as a black box.

### Engineering Philosophy

The engineering layer of the project is deliberately simple.

There is currently no production deployment, cloud infrastructure, Docker/Kubernetes stack, CI/CD pipeline or model monitoring system.

Instead, the focus is on establishing the foundations required for **reproducible machine learning experimentation**:

```text
Raw Data
   ↓
Problem-Specific Dataset
   ↓
Feature Engineering
   ↓
Training
   ↓
Hyperparameter Optimization
   ↓
Evaluation
   ↓
MLflow
   ├── Parameters
   ├── Metrics
   ├── Models
   ├── Artifacts
   ├── Dataset Context
   └── Split Configuration
```

For the scope of this project, this lightweight approach was sufficient to make the experiments traceable, comparable and reproducible.

### What I Learned

One of the main lessons from the project was how useful experiment tracking becomes once the number of experiments starts increasing.

MLflow made it possible to quickly inspect metrics, visualize experiments, compare configurations and identify inconsistencies without manually keeping track of every training run.

This became particularly valuable as the project grew from simple model experiments into a collection of five different machine learning problems.

The project also showed that **machine learning engineering is not only about deploying models**. Even before deployment, keeping datasets, features, parameters, metrics and models organized has a significant impact on the quality and reproducibility of the experimentation process.

Similarly, Optuna demonstrated the value of systematic hyperparameter optimization while reinforcing that automated optimization works best when combined with an understanding of the models and hyperparameters being optimized.

## ⚙️ Installation

### Requirements

The project requires **Python 3.10+** and uses a virtual environment to isolate the project dependencies.

Clone the repository and create a virtual environment:

```powershell
git clone <https://github.com/oteope/clash-of-clans-ml-lab.git
cd clash-of-clans-ml-lab>
cd clash-of-clans-ml-lab

python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The main dependencies include:

- **MLflow** — experiment tracking and model management
- **pandas / NumPy** — data processing
- **scikit-learn** — machine learning algorithms and evaluation
- **XGBoost** — gradient boosting models
- **Optuna** — hyperparameter optimization
- **PyArrow** — Parquet dataset handling
- **Matplotlib / Seaborn** — visualization and EDA
- **aiohttp** — asynchronous API requests
- **python-dotenv** — environment variable management

### Clash of Clans API

The data collection pipeline requires access to the **Clash of Clans Developer API**.

The crawler uses an API key and the expected IP address configured as environment variables.

Create a `.env` file in the project root:

```env
CLASH_API_KEY=your_api_key
CLASH_API_IP=your_expected_ip
```

These credentials are required only for running the data collection pipeline. They should **never be committed to the repository**.

### Datasets

The final `.parquet` datasets are not included in the repository.

They are excluded through `.gitignore` because of their size. Therefore, a fresh clone does not contain the processed datasets required by the ML pipelines.

To reproduce the datasets from scratch, the data pipeline must be executed using a valid Clash of Clans API configuration.

The general process is:

```text
Clash of Clans API
        ↓
Raw JSON Data
        ↓
Dataset Construction
        ↓
Feature Engineering
        ↓
Parquet Datasets
```

The raw data and generated datasets are stored locally under the project's `data/` directory.

### MLflow

The project uses a **local MLflow Tracking Server** with SQLite as the backend store.

After installing the dependencies, start the MLflow server:

```powershell
mlflow server `
  --backend-store-uri sqlite:///mlflow/mlflow.db `
  --default-artifact-root ./mlflow/mlruns `
  --host 127.0.0.1 `
  --port 5000
```

The tracking server will be available at:

```text
http://127.0.0.1:5000
```

Configure the tracking URI in the terminal running the ML pipeline:

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
```

Alternatively, the project can use its local SQLite configuration through the tracking utilities when no external tracking URI is provided.

For a more detailed explanation of the MLflow setup, see the **Machine Learning Engineering** section.

## 🚀 Usage

The project is executed primarily through the command line.

The general workflow is:

```text
Data Extraction
      ↓
Feature Engineering & Dataset Construction
      ↓
Model Training / Clustering
      ↓
Hyperparameter Optimization
      ↓
Evaluation
      ↓
MLflow Results
```

### 1. Extract Raw Data

After configuring the Clash of Clans API credentials, run the main extraction pipeline:

```powershell
python src/extraction/main_extraction.py
```

The extracted raw data is stored locally under the `data/raw/` directory.

### 2. Build the Dataset

Each machine learning problem has its own feature engineering and dataset construction pipeline.

The corresponding scripts are located under:

```text
src/features/
├── problem1/
├── problem2/
├── problem3/
├── problem4/
└── problem5/
```

After extracting the raw data, run the dataset builder corresponding to the problem you want to work with.

For example:

```powershell
python src/features/problem1/<build_problem1_name>_dataset.py
```

The same process applies to the other problems by using their corresponding `problem2`, `problem3`, `problem4` or `problem5` feature pipeline.

Each pipeline performs the feature engineering required for its specific machine learning problem and generates the corresponding Parquet dataset.

### 3. Start MLflow

Before running the machine learning experiments, start the local MLflow tracking server:

```powershell
mlflow server `
  --backend-store-uri sqlite:///mlflow/mlflow.db `
  --default-artifact-root ./mlflow/mlruns `
  --host 127.0.0.1 `
  --port 5000
```

Keep the MLflow server running while executing experiments.

The MLflow UI will be available at:

```text
http://127.0.0.1:5000
```

### 4. Run Machine Learning Experiments

The machine learning experiments are organized by problem under:

```text
src/models/
├── P1/
├── P2/
├── P3/
├── P4/
└── P5/
```

Each problem contains its own training, optimization and evaluation scripts.

Depending on the experiment, model parameters can either be configured manually before execution or optimized automatically using **Optuna**.

Run the corresponding experiment from the terminal:

```powershell
python src/models/P1/<experiment_script>.py
```

The same structure applies to `P2`, `P3`, `P4` and `P5`.

Experiments that use Optuna perform the configured hyperparameter search automatically before logging the selected final model and results to MLflow.

### 5. Analyze Results

After running the experiments, the corresponding results scripts can be used to analyze the trained models and generate metrics, visualizations and comparisons.

Each problem has its own results workflow. **P5 is an exception:** executing the P5 pipeline also produces problem-specific results, but the project additionally includes a **global results analysis** that consolidates the main results
# Problem 2: Clan Rank Regression - EDA

## Diagnostic and comparison of full and trophy-free dataset variants

This notebook analyzes the dataset for **Problem 2: Clan Rank Regression**. It evaluates two variants: the full dataset including direct trophy metrics, and a trophy-free variant built solely from clan composition, activity, and progression signals.

## 1. Setup and Dynamic Data Loading

Libraries are imported and plotting styles are configured. The project root is located dynamically, the dataset is loaded, and two DataFrame views are created: `df_full` and `df_no_trophies`.


```python
# === 1. Setup and Dynamic Data Loading ===
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from pathlib import Path
from collections import defaultdict

sns.set_theme(style='whitegrid', palette='viridis')
%matplotlib inline

def find_project_root() -> Path:
    """Locate the project root by searching for the Problem 2 datasets."""
    candidates = [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]
    for cand in candidates:
        if (cand / 'data' / 'datasets' / 'clan_rank_regression_with_trophies.parquet').exists():
            return cand
    raise FileNotFoundError(
        'Could not find data/datasets/clan_rank_regression_with_trophies.parquet. '
        'Adjust the path or run the notebook from the project root.'
    )

root = find_project_root()
PATH_WITH_TROPHIES = root / 'data' / 'datasets' / 'clan_rank_regression_with_trophies.parquet'
PATH_WITHOUT_TROPHIES = root / 'data' / 'datasets' / 'clan_rank_regression_without_trophies.parquet'

print(f'Project root directory: {root}')
print(f'Full dataset path: {PATH_WITH_TROPHIES}')
print(f'Trophy-free dataset path: {PATH_WITHOUT_TROPHIES}')

# Load both dataset variants directly
df_full = pd.read_parquet(PATH_WITH_TROPHIES)
df_no_trophies = pd.read_parquet(PATH_WITHOUT_TROPHIES)

TARGET = 'clan_rank'

# Identify trophy columns that exist in df_full but not in df_no_trophies for logging
trophy_cols = list(set(df_full.columns) - set(df_no_trophies.columns))
print('')
print(f'Excluded trophy-related columns in trophy-free variant: {trophy_cols}')

print('')
print(f'Full variant shape: {df_full.shape}')
print(f'Trophy-free variant shape: {df_no_trophies.shape}')
df_full.head()
```

    Project root directory: c:\Users\Usuario\Desktop\Clash of Clans ML Lab
    Full dataset path: c:\Users\Usuario\Desktop\Clash of Clans ML Lab\data\datasets\clan_rank_regression_with_trophies.parquet
    Trophy-free dataset path: c:\Users\Usuario\Desktop\Clash of Clans ML Lab\data\datasets\clan_rank_regression_without_trophies.parquet
    
    Excluded trophy-related columns in trophy-free variant: ['required_trophies', 'best_trophies', 'trophies_diff_from_clan_mean', 'trophies', 'clan_mean_trophies', 'progression_ratio_trophies', 'trophies_clan_pct', 'trophies_ratio_to_clan_mean']
    
    Full variant shape: (837411, 87)
    Trophy-free variant shape: (837411, 79)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>player_tag</th>
      <th>clan_tag</th>
      <th>name</th>
      <th>town_hall_level</th>
      <th>exp_level</th>
      <th>league_id</th>
      <th>league_name</th>
      <th>league_tier_id</th>
      <th>league_tier_name</th>
      <th>trophies</th>
      <th>...</th>
      <th>war_stars_diff_from_clan_mean</th>
      <th>war_stars_ratio_to_clan_mean</th>
      <th>war_stars_clan_pct</th>
      <th>attack_wins_diff_from_clan_mean</th>
      <th>attack_wins_ratio_to_clan_mean</th>
      <th>attack_wins_clan_pct</th>
      <th>defense_wins_diff_from_clan_mean</th>
      <th>defense_wins_ratio_to_clan_mean</th>
      <th>defense_wins_clan_pct</th>
      <th>clan_rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>#Y8GGPRUPR</td>
      <td>#200022JVL</td>
      <td>F A N</td>
      <td>18</td>
      <td>270</td>
      <td>29000022</td>
      <td>Legend League</td>
      <td>105000035</td>
      <td>Legend II</td>
      <td>1218</td>
      <td>...</td>
      <td>1235.020408</td>
      <td>1.550125</td>
      <td>0.857143</td>
      <td>26.795918</td>
      <td>9.363057</td>
      <td>1.000000</td>
      <td>0.55102</td>
      <td>2.227273</td>
      <td>0.857143</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>#9PQV8JPCP</td>
      <td>#200022JVL</td>
      <td>Dark to</td>
      <td>18</td>
      <td>249</td>
      <td>29000000</td>
      <td>Unranked</td>
      <td>105000035</td>
      <td>Legend II</td>
      <td>1045</td>
      <td>...</td>
      <td>1274.020408</td>
      <td>1.567498</td>
      <td>0.897959</td>
      <td>-3.204082</td>
      <td>0.000000</td>
      <td>0.387755</td>
      <td>-0.44898</td>
      <td>0.000000</td>
      <td>0.397959</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>#P8RLQVJL</td>
      <td>#200022JVL</td>
      <td>Aerox</td>
      <td>18</td>
      <td>291</td>
      <td>29000022</td>
      <td>Legend League</td>
      <td>105000035</td>
      <td>Legend II</td>
      <td>859</td>
      <td>...</td>
      <td>4128.020408</td>
      <td>2.838779</td>
      <td>1.000000</td>
      <td>16.795918</td>
      <td>6.242038</td>
      <td>0.959184</td>
      <td>-0.44898</td>
      <td>0.000000</td>
      <td>0.397959</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>#2VU80892Y</td>
      <td>#200022JVL</td>
      <td>Prof,D,Bijoxz</td>
      <td>18</td>
      <td>266</td>
      <td>29000022</td>
      <td>Legend League</td>
      <td>105000035</td>
      <td>Legend II</td>
      <td>674</td>
      <td>...</td>
      <td>1210.020408</td>
      <td>1.538989</td>
      <td>0.836735</td>
      <td>12.795918</td>
      <td>4.993631</td>
      <td>0.918367</td>
      <td>-0.44898</td>
      <td>0.000000</td>
      <td>0.397959</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>#98CPPQYUC</td>
      <td>#200022JVL</td>
      <td>Majestic.</td>
      <td>18</td>
      <td>250</td>
      <td>29000000</td>
      <td>Unranked</td>
      <td>105000035</td>
      <td>Legend II</td>
      <td>112</td>
      <td>...</td>
      <td>372.020408</td>
      <td>1.165712</td>
      <td>0.632653</td>
      <td>-3.204082</td>
      <td>0.000000</td>
      <td>0.387755</td>
      <td>-0.44898</td>
      <td>0.000000</td>
      <td>0.397959</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 87 columns</p>
</div>



## 2. Data Integrity and Quality Audit

Dimensions, data types, missing values, duplicate rows, and quasi-constant features are audited for both variants.


```python
# === 2. Data Integrity and Quality Audit ===

def audit_dataframe(data: pd.DataFrame, name: str) -> None:
    """Prints basic integrity metrics for a DataFrame."""
    print('')
    print(f'=== {name} ===')
    print(f'Dimensions: {data.shape[0]} rows x {data.shape[1]} columns')
    print('')
    print('Data types:')
    display(data.dtypes.to_frame(name='dtype'))
    missing = data.isna().sum()
    missing_pct = (missing / len(data)) * 100
    missing_df = pd.DataFrame({'missing': missing, 'percentage': missing_pct})
    print('')
    print('Missing values (columns with missing > 0):')
    display(missing_df[missing_df['missing'] > 0].sort_values('percentage', ascending=False))
    dup_rows = data.duplicated().sum()
    print('')
    print(f'Exact duplicate rows: {dup_rows}')
    threshold = 0.95
    constant_features = []
    for col in data.columns:
        value_counts = data[col].value_counts(dropna=False)
        top_freq = value_counts.iloc[0] / len(data) if len(value_counts) > 0 else 1.0
        if top_freq >= threshold:
            constant_features.append((col, 'nearly_constant', top_freq))
        if pd.api.types.is_numeric_dtype(data[col]) and data[col].nunique(dropna=True) <= 1:
            constant_features.append((col, 'zero_variance', top_freq))
    if constant_features:
        print('')
        print(f'Quasi-constant or zero-variance features (threshold={threshold}):')
        display(pd.DataFrame(constant_features, columns=['column', 'type', 'mode_frequency']))
    else:
        print('')
        print(f'No quasi-constant features with mode frequency >= {threshold}.')

audit_dataframe(df_full, 'Full Dataset')
audit_dataframe(df_no_trophies, 'Trophy-Free Dataset')
```

    
    === Full Dataset ===
    Dimensions: 837411 rows x 87 columns
    
    Data types:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dtype</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>player_tag</th>
      <td>str</td>
    </tr>
    <tr>
      <th>clan_tag</th>
      <td>str</td>
    </tr>
    <tr>
      <th>name</th>
      <td>str</td>
    </tr>
    <tr>
      <th>town_hall_level</th>
      <td>int64</td>
    </tr>
    <tr>
      <th>exp_level</th>
      <td>int64</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>attack_wins_clan_pct</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>defense_wins_diff_from_clan_mean</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>defense_wins_ratio_to_clan_mean</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>defense_wins_clan_pct</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>clan_rank</th>
      <td>int64</td>
    </tr>
  </tbody>
</table>
<p>87 rows × 1 columns</p>
</div>


    
    Missing values (columns with missing > 0):
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>missing</th>
      <th>percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>capital_contributions</th>
      <td>837411</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>clan_mean_capital_contributions</th>
      <td>837411</td>
      <td>100.0</td>
    </tr>
  </tbody>
</table>
</div>


    
    Exact duplicate rows: 0
    
    Quasi-constant or zero-variance features (threshold=0.95):
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>column</th>
      <th>type</th>
      <th>mode_frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>capital_contributions</td>
      <td>nearly_constant</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>capital_contributions</td>
      <td>zero_variance</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>attack_wins</td>
      <td>nearly_constant</td>
      <td>0.977350</td>
    </tr>
    <tr>
      <th>3</th>
      <td>defense_wins</td>
      <td>nearly_constant</td>
      <td>0.961056</td>
    </tr>
    <tr>
      <th>4</th>
      <td>combat_activity_total</td>
      <td>nearly_constant</td>
      <td>0.956284</td>
    </tr>
    <tr>
      <th>5</th>
      <td>achievement_count</td>
      <td>nearly_constant</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>achievement_count</td>
      <td>zero_variance</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>clan_mean_capital_contributions</td>
      <td>nearly_constant</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>clan_mean_capital_contributions</td>
      <td>zero_variance</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>capital_contributions_diff_from_clan_mean</td>
      <td>nearly_constant</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>capital_contributions_diff_from_clan_mean</td>
      <td>zero_variance</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>capital_contributions_ratio_to_clan_mean</td>
      <td>nearly_constant</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>capital_contributions_ratio_to_clan_mean</td>
      <td>zero_variance</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>capital_contributions_clan_pct</td>
      <td>nearly_constant</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>capital_contributions_clan_pct</td>
      <td>zero_variance</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>attack_wins_ratio_to_clan_mean</td>
      <td>nearly_constant</td>
      <td>0.977350</td>
    </tr>
    <tr>
      <th>16</th>
      <td>defense_wins_ratio_to_clan_mean</td>
      <td>nearly_constant</td>
      <td>0.961056</td>
    </tr>
  </tbody>
</table>
</div>


    
    === Trophy-Free Dataset ===
    Dimensions: 837411 rows x 79 columns
    
    Data types:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dtype</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>player_tag</th>
      <td>str</td>
    </tr>
    <tr>
      <th>clan_tag</th>
      <td>str</td>
    </tr>
    <tr>
      <th>name</th>
      <td>str</td>
    </tr>
    <tr>
      <th>town_hall_level</th>
      <td>int64</td>
    </tr>
    <tr>
      <th>exp_level</th>
      <td>int64</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>attack_wins_clan_pct</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>defense_wins_diff_from_clan_mean</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>defense_wins_ratio_to_clan_mean</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>defense_wins_clan_pct</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>clan_rank</th>
      <td>int64</td>
    </tr>
  </tbody>
</table>
<p>79 rows × 1 columns</p>
</div>


    
    Missing values (columns with missing > 0):
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>missing</th>
      <th>percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>capital_contributions</th>
      <td>837411</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>clan_mean_capital_contributions</th>
      <td>837411</td>
      <td>100.0</td>
    </tr>
  </tbody>
</table>
</div>


    
    Exact duplicate rows: 0
    
    Quasi-constant or zero-variance features (threshold=0.95):
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>column</th>
      <th>type</th>
      <th>mode_frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>capital_contributions</td>
      <td>nearly_constant</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>capital_contributions</td>
      <td>zero_variance</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>attack_wins</td>
      <td>nearly_constant</td>
      <td>0.977350</td>
    </tr>
    <tr>
      <th>3</th>
      <td>defense_wins</td>
      <td>nearly_constant</td>
      <td>0.961056</td>
    </tr>
    <tr>
      <th>4</th>
      <td>combat_activity_total</td>
      <td>nearly_constant</td>
      <td>0.956284</td>
    </tr>
    <tr>
      <th>5</th>
      <td>achievement_count</td>
      <td>nearly_constant</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>achievement_count</td>
      <td>zero_variance</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>clan_mean_capital_contributions</td>
      <td>nearly_constant</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>clan_mean_capital_contributions</td>
      <td>zero_variance</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>capital_contributions_diff_from_clan_mean</td>
      <td>nearly_constant</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>capital_contributions_diff_from_clan_mean</td>
      <td>zero_variance</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>capital_contributions_ratio_to_clan_mean</td>
      <td>nearly_constant</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>capital_contributions_ratio_to_clan_mean</td>
      <td>zero_variance</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>capital_contributions_clan_pct</td>
      <td>nearly_constant</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>capital_contributions_clan_pct</td>
      <td>zero_variance</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>attack_wins_ratio_to_clan_mean</td>
      <td>nearly_constant</td>
      <td>0.977350</td>
    </tr>
    <tr>
      <th>16</th>
      <td>defense_wins_ratio_to_clan_mean</td>
      <td>nearly_constant</td>
      <td>0.961056</td>
    </tr>
  </tbody>
</table>
</div>


## 3. Target Analysis (`clan_rank`)

Descriptive statistics, distribution plots in raw and log-transformed scale, and skewness diagnostics are performed for the target variable.


```python
# === 3. Target Analysis (clan_rank) ===

target = df_full[TARGET].copy()
print('=== Descriptive statistics for clan_rank ===')
print(target.describe().to_string())
print('')
skewness = stats.skew(target.dropna())
kurtosis = stats.kurtosis(target.dropna())
print(f'Skewness: {skewness:.4f}')
print(f'Kurtosis: {kurtosis:.4f}')

# Raw scale distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(target, kde=True, ax=axes[0])
axes[0].set_title('clan_rank distribution (raw scale)')
axes[0].set_xlabel('clan_rank')
axes[0].set_ylabel('Frequency')

# Log-transformed scale
log_target = np.log1p(target)
sns.histplot(log_target, kde=True, ax=axes[1])
axes[1].set_title('clan_rank distribution (log1p scale)')
axes[1].set_xlabel('log1p(clan_rank)')
axes[1].set_ylabel('Frequency')
plt.tight_layout()
plt.show()

# Log-transform recommendation
if abs(skewness) > 1.0:
    print('The clan_rank distribution is highly skewed. A log1p transformation is recommended before modeling.')
elif abs(skewness) > 0.5:
    print('The clan_rank distribution is moderately skewed. A log1p transformation may improve model performance.')
else:
    print('The clan_rank distribution is approximately symmetric. No transformation is strictly required.')
```

    === Descriptive statistics for clan_rank ===
    count    837411.000000
    mean         13.780303
    std          11.250930
    min           1.000000
    25%           5.000000
    50%          10.000000
    75%          20.000000
    max          50.000000
    
    Skewness: 1.0297
    Kurtosis: 0.3160
    


    
![png](eda_p2_files/eda_p2_6_1.png)
    


    The clan_rank distribution is highly skewed. A log1p transformation is recommended before modeling.
    

## 4. Feature Group Profiling

Variables are classified into conceptual groups: **Progression**, **Activity**, **Economy**, **Trophies/Competition**, and **Clan Composition**. Skewness, zero-value ratio, and missing percentage are reported for each numerical feature.


```python
# === 4. Feature Group Profiling ===

def categorize_feature(col: str) -> str:
    """Assigns a feature to a conceptual group based on its name."""
    col_l = col.lower()
    if any(k in col_l for k in ['troop', 'hero', 'spell', 'equipment', 'builder_hall', 'town_hall', 'exp_level', 'achievement']):
        return 'Progression'
    if any(k in col_l for k in ['donat', 'attack', 'defense', 'war_stars', 'versus_battle']):
        return 'Activity'
    if any(k in col_l for k in ['loot', 'gold', 'elixir', 'dark_elixir', 'clan_games']):
        return 'Economy'
    if any(k in col_l for k in ['troph', 'points', 'league', 'legend']):
        return 'Trophies/Competition'
    if any(k in col_l for k in ['members', 'clan_', 'count', 'composition', 'size']):
        return 'Clan Composition'
    return 'Other'

def profile_numeric_features(data: pd.DataFrame, name: str) -> pd.DataFrame:
    """Computes skewness, zero ratio and missing ratio for numeric columns."""
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET in numeric_cols:
        numeric_cols.remove(TARGET)
    records = []
    for col in numeric_cols:
        series = data[col]
        records.append({
            'feature': col,
            'group': categorize_feature(col),
            'skewness': series.skew(),
            'zero_pct': (series == 0).mean() * 100,
            'missing_pct': series.isna().mean() * 100
        })
    profile = pd.DataFrame(records).sort_values('skewness', key=lambda s: s.abs(), ascending=False)
    print('')
    print(f'=== Numerical feature profile for {name} ===')
    display(profile)
    return profile

profile_full = profile_numeric_features(df_full, 'Full Dataset')
profile_no_trophies = profile_numeric_features(df_no_trophies, 'Trophy-Free Dataset')
```

    
    === Numerical feature profile for Full Dataset ===
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>group</th>
      <th>skewness</th>
      <th>zero_pct</th>
      <th>missing_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17</th>
      <td>donation_ratio</td>
      <td>Activity</td>
      <td>883.319323</td>
      <td>80.792108</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>donations</td>
      <td>Activity</td>
      <td>391.451202</td>
      <td>73.309522</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>58</th>
      <td>donations_diff_from_clan_mean</td>
      <td>Activity</td>
      <td>365.867884</td>
      <td>30.106722</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>donation_balance</td>
      <td>Activity</td>
      <td>349.795288</td>
      <td>65.915900</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>donations_received</td>
      <td>Activity</td>
      <td>151.424792</td>
      <td>73.063884</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>66</th>
      <td>capital_contributions_clan_pct</td>
      <td>Clan Composition</td>
      <td>0.000000</td>
      <td>100.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>64</th>
      <td>capital_contributions_diff_from_clan_mean</td>
      <td>Clan Composition</td>
      <td>0.000000</td>
      <td>100.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>65</th>
      <td>capital_contributions_ratio_to_clan_mean</td>
      <td>Clan Composition</td>
      <td>0.000000</td>
      <td>100.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>capital_contributions</td>
      <td>Other</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>clan_mean_capital_contributions</td>
      <td>Clan Composition</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>100.0</td>
    </tr>
  </tbody>
</table>
<p>76 rows × 5 columns</p>
</div>


    
    === Numerical feature profile for Trophy-Free Dataset ===
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>group</th>
      <th>skewness</th>
      <th>zero_pct</th>
      <th>missing_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>donation_ratio</td>
      <td>Activity</td>
      <td>883.319323</td>
      <td>80.792108</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>donations</td>
      <td>Activity</td>
      <td>391.451202</td>
      <td>73.309522</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>50</th>
      <td>donations_diff_from_clan_mean</td>
      <td>Activity</td>
      <td>365.867884</td>
      <td>30.106722</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>donation_balance</td>
      <td>Activity</td>
      <td>349.795288</td>
      <td>65.915900</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>donations_received</td>
      <td>Activity</td>
      <td>151.424792</td>
      <td>73.063884</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>58</th>
      <td>capital_contributions_clan_pct</td>
      <td>Clan Composition</td>
      <td>0.000000</td>
      <td>100.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>56</th>
      <td>capital_contributions_diff_from_clan_mean</td>
      <td>Clan Composition</td>
      <td>0.000000</td>
      <td>100.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>57</th>
      <td>capital_contributions_ratio_to_clan_mean</td>
      <td>Clan Composition</td>
      <td>0.000000</td>
      <td>100.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>capital_contributions</td>
      <td>Other</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>clan_mean_capital_contributions</td>
      <td>Clan Composition</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>100.0</td>
    </tr>
  </tbody>
</table>
<p>68 rows × 5 columns</p>
</div>


## 5. Feature-Target Relationships (Regression)

For each variant, Mutual Information, Pearson correlation, and Spearman correlation with `clan_rank` are computed. Scatter plots are shown for the top 6 features by Mutual Information.


```python
# === 5. Feature-Target Relationships (Regression) ===

def compute_feature_target_metrics(data: pd.DataFrame, name: str) -> dict:
    """Computes MI, Pearson and Spearman correlations for numeric features vs target."""
    # Limpiar filas con target nulo o valores infinitos
    clean_data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[TARGET]).copy()
    target = clean_data[TARGET]
    
    numeric_cols = clean_data.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET in numeric_cols:
        numeric_cols.remove(TARGET)

    # Impute median temporarily for Mutual Information
    X = clean_data[numeric_cols].copy()
    
    # Eliminar columnas completamente vacías si las hubiera
    X = X.dropna(how='all', axis=1)
    valid_cols = X.columns.tolist()

    for col in valid_cols:
        if X[col].isna().any():
            med = X[col].median()
            X[col] = X[col].fillna(med if pd.notna(med) else 0)

    mi_scores = mutual_info_regression(X, target, random_state=42)
    mi_df = pd.DataFrame({'feature': valid_cols, 'mutual_info': mi_scores})
    mi_df = mi_df.sort_values('mutual_info', ascending=False).reset_index(drop=True)

    pearson = []
    spearman = []
    for col in valid_cols:
        pearson_r = clean_data[col].corr(target, method='pearson')
        spearman_r = clean_data[col].corr(target, method='spearman')
        pearson.append((col, pearson_r))
        spearman.append((col, spearman_r))

    pearson_df = pd.DataFrame(pearson, columns=['feature', 'pearson_r']).sort_values('pearson_r', key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    spearman_df = pd.DataFrame(spearman, columns=['feature', 'spearman_r']).sort_values('spearman_r', key=lambda s: s.abs(), ascending=False).reset_index(drop=True)

    print('')
    print(f'=== Feature-target metrics for {name} ===')
    print('Top 10 Mutual Information:')
    display(mi_df.head(10))
    print('')
    print('Top 10 Pearson correlation:')
    display(pearson_df.head(10))
    print('')
    print('Top 10 Spearman correlation:')
    display(spearman_df.head(10))

    # Scatter plots for top 6 MI features
    top_features = mi_df.head(6)['feature'].tolist()
    if top_features:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        for ax, var in zip(axes, top_features):
            ax.scatter(clean_data[var], clean_data[TARGET], alpha=0.6)
            ax.set_title(f'{var} vs clan_rank')
            ax.set_xlabel(var)
            ax.set_ylabel(TARGET)
        for ax in axes[len(top_features):]:
            ax.axis('off')
        plt.suptitle(f'Top 6 MI features for {name}')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

    return {'mi': mi_df, 'pearson': pearson_df, 'spearman': spearman_df}

metrics_full = compute_feature_target_metrics(df_full, 'Full Dataset')
metrics_no_trophies = compute_feature_target_metrics(df_no_trophies, 'Trophy-Free Dataset')
```

    
    === Feature-target metrics for Full Dataset ===
    Top 10 Mutual Information:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>mutual_info</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>members</td>
      <td>0.467676</td>
    </tr>
    <tr>
      <th>1</th>
      <td>attack_wins_clan_pct</td>
      <td>0.423052</td>
    </tr>
    <tr>
      <th>2</th>
      <td>town_hall_level_clan_pct</td>
      <td>0.419419</td>
    </tr>
    <tr>
      <th>3</th>
      <td>defense_wins_clan_pct</td>
      <td>0.388326</td>
    </tr>
    <tr>
      <th>4</th>
      <td>exp_level_clan_pct</td>
      <td>0.352160</td>
    </tr>
    <tr>
      <th>5</th>
      <td>donations_clan_pct</td>
      <td>0.348760</td>
    </tr>
    <tr>
      <th>6</th>
      <td>donations_received_clan_pct</td>
      <td>0.308286</td>
    </tr>
    <tr>
      <th>7</th>
      <td>town_hall_level_diff_from_clan_mean</td>
      <td>0.286278</td>
    </tr>
    <tr>
      <th>8</th>
      <td>war_stars_clan_pct</td>
      <td>0.281213</td>
    </tr>
    <tr>
      <th>9</th>
      <td>trophies_clan_pct</td>
      <td>0.276252</td>
    </tr>
  </tbody>
</table>
</div>


    
    Top 10 Pearson correlation:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>pearson_r</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>members</td>
      <td>0.633797</td>
    </tr>
    <tr>
      <th>1</th>
      <td>clan_capital_points</td>
      <td>0.460138</td>
    </tr>
    <tr>
      <th>2</th>
      <td>clan_points</td>
      <td>0.454273</td>
    </tr>
    <tr>
      <th>3</th>
      <td>town_hall_level_clan_pct</td>
      <td>-0.420877</td>
    </tr>
    <tr>
      <th>4</th>
      <td>exp_level_clan_pct</td>
      <td>-0.382472</td>
    </tr>
    <tr>
      <th>5</th>
      <td>donations_clan_pct</td>
      <td>-0.360929</td>
    </tr>
    <tr>
      <th>6</th>
      <td>clan_level</td>
      <td>0.357667</td>
    </tr>
    <tr>
      <th>7</th>
      <td>clan_mean_town_hall_level</td>
      <td>0.341561</td>
    </tr>
    <tr>
      <th>8</th>
      <td>war_stars_clan_pct</td>
      <td>-0.337008</td>
    </tr>
    <tr>
      <th>9</th>
      <td>exp_level_diff_from_clan_mean</td>
      <td>-0.330000</td>
    </tr>
  </tbody>
</table>
</div>


    
    Top 10 Spearman correlation:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>spearman_r</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>members</td>
      <td>0.620904</td>
    </tr>
    <tr>
      <th>1</th>
      <td>attack_wins_clan_pct</td>
      <td>-0.568895</td>
    </tr>
    <tr>
      <th>2</th>
      <td>defense_wins_clan_pct</td>
      <td>-0.515443</td>
    </tr>
    <tr>
      <th>3</th>
      <td>clan_points</td>
      <td>0.490260</td>
    </tr>
    <tr>
      <th>4</th>
      <td>town_hall_level_clan_pct</td>
      <td>-0.481195</td>
    </tr>
    <tr>
      <th>5</th>
      <td>clan_capital_points</td>
      <td>0.456444</td>
    </tr>
    <tr>
      <th>6</th>
      <td>donations_clan_pct</td>
      <td>-0.442348</td>
    </tr>
    <tr>
      <th>7</th>
      <td>exp_level_clan_pct</td>
      <td>-0.441006</td>
    </tr>
    <tr>
      <th>8</th>
      <td>donations_diff_from_clan_mean</td>
      <td>-0.410980</td>
    </tr>
    <tr>
      <th>9</th>
      <td>town_hall_level_diff_from_clan_mean</td>
      <td>-0.402084</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](eda_p2_files/eda_p2_10_6.png)
    


    
    === Feature-target metrics for Trophy-Free Dataset ===
    Top 10 Mutual Information:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>mutual_info</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>members</td>
      <td>0.468411</td>
    </tr>
    <tr>
      <th>1</th>
      <td>attack_wins_clan_pct</td>
      <td>0.420851</td>
    </tr>
    <tr>
      <th>2</th>
      <td>town_hall_level_clan_pct</td>
      <td>0.418714</td>
    </tr>
    <tr>
      <th>3</th>
      <td>defense_wins_clan_pct</td>
      <td>0.389441</td>
    </tr>
    <tr>
      <th>4</th>
      <td>exp_level_clan_pct</td>
      <td>0.353433</td>
    </tr>
    <tr>
      <th>5</th>
      <td>donations_clan_pct</td>
      <td>0.349768</td>
    </tr>
    <tr>
      <th>6</th>
      <td>donations_received_clan_pct</td>
      <td>0.307359</td>
    </tr>
    <tr>
      <th>7</th>
      <td>town_hall_level_diff_from_clan_mean</td>
      <td>0.286037</td>
    </tr>
    <tr>
      <th>8</th>
      <td>war_stars_clan_pct</td>
      <td>0.282604</td>
    </tr>
    <tr>
      <th>9</th>
      <td>town_hall_level_ratio_to_clan_mean</td>
      <td>0.203446</td>
    </tr>
  </tbody>
</table>
</div>


    
    Top 10 Pearson correlation:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>pearson_r</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>members</td>
      <td>0.633797</td>
    </tr>
    <tr>
      <th>1</th>
      <td>clan_capital_points</td>
      <td>0.460138</td>
    </tr>
    <tr>
      <th>2</th>
      <td>clan_points</td>
      <td>0.454273</td>
    </tr>
    <tr>
      <th>3</th>
      <td>town_hall_level_clan_pct</td>
      <td>-0.420877</td>
    </tr>
    <tr>
      <th>4</th>
      <td>exp_level_clan_pct</td>
      <td>-0.382472</td>
    </tr>
    <tr>
      <th>5</th>
      <td>donations_clan_pct</td>
      <td>-0.360929</td>
    </tr>
    <tr>
      <th>6</th>
      <td>clan_level</td>
      <td>0.357667</td>
    </tr>
    <tr>
      <th>7</th>
      <td>clan_mean_town_hall_level</td>
      <td>0.341561</td>
    </tr>
    <tr>
      <th>8</th>
      <td>war_stars_clan_pct</td>
      <td>-0.337008</td>
    </tr>
    <tr>
      <th>9</th>
      <td>exp_level_diff_from_clan_mean</td>
      <td>-0.330000</td>
    </tr>
  </tbody>
</table>
</div>


    
    Top 10 Spearman correlation:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>spearman_r</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>members</td>
      <td>0.620904</td>
    </tr>
    <tr>
      <th>1</th>
      <td>attack_wins_clan_pct</td>
      <td>-0.568895</td>
    </tr>
    <tr>
      <th>2</th>
      <td>defense_wins_clan_pct</td>
      <td>-0.515443</td>
    </tr>
    <tr>
      <th>3</th>
      <td>clan_points</td>
      <td>0.490260</td>
    </tr>
    <tr>
      <th>4</th>
      <td>town_hall_level_clan_pct</td>
      <td>-0.481195</td>
    </tr>
    <tr>
      <th>5</th>
      <td>clan_capital_points</td>
      <td>0.456444</td>
    </tr>
    <tr>
      <th>6</th>
      <td>donations_clan_pct</td>
      <td>-0.442348</td>
    </tr>
    <tr>
      <th>7</th>
      <td>exp_level_clan_pct</td>
      <td>-0.441006</td>
    </tr>
    <tr>
      <th>8</th>
      <td>donations_diff_from_clan_mean</td>
      <td>-0.410980</td>
    </tr>
    <tr>
      <th>9</th>
      <td>town_hall_level_diff_from_clan_mean</td>
      <td>-0.402084</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](eda_p2_files/eda_p2_10_13.png)
    


## 6. Multicollinearity and Redundancy

Pearson and Spearman correlation matrices are computed among numerical predictors. Feature pairs exceeding |r| > 0.85 are reported and visual heatmaps are shown.


```python
# === 6. Multicollinearity and Redundancy ===

def collinearity_report(data: pd.DataFrame, name: str) -> None:
    """Prints high-correlation pairs and displays correlation heatmaps."""
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET in numeric_cols:
        numeric_cols.remove(TARGET)
    corr_data = data[numeric_cols].copy().replace([np.inf, -np.inf], np.nan)
    for col in corr_data.columns:
        if corr_data[col].isna().any():
            corr_data[col] = corr_data[col].fillna(corr_data[col].median())

    pearson_corr = corr_data.corr(method='pearson')
    spearman_corr = corr_data.corr(method='spearman')

    threshold = 0.85

    def get_high_corr_pairs(corr_matrix):
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        pairs = []
        for col in upper.columns:
            for idx in upper.index:
                val = upper.loc[idx, col]
                if pd.notna(val) and abs(val) > threshold:
                    pairs.append((idx, col, val))
        return pd.DataFrame(pairs, columns=['feature_1', 'feature_2', 'corr'])

    pearson_pairs = get_high_corr_pairs(pearson_corr)
    spearman_pairs = get_high_corr_pairs(spearman_corr)

    print('')
    print(f'=== Collinearity report for {name} ===')
    print(f'Pairs with |Pearson| > {threshold}:')
    display(pearson_pairs if len(pearson_pairs) > 0 else None)
    print('')
    print(f'Pairs with |Spearman| > {threshold}:')
    display(spearman_pairs if len(spearman_pairs) > 0 else None)

    plt.figure(figsize=(12, 10))
    sns.heatmap(pearson_corr, annot=False, cmap='coolwarm', center=0, square=True)
    plt.title(f'Pearson correlation matrix - {name}')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 10))
    sns.heatmap(spearman_corr, annot=False, cmap='coolwarm', center=0, square=True)
    plt.title(f'Spearman correlation matrix - {name}')
    plt.tight_layout()
    plt.show()

collinearity_report(df_full, 'Full Dataset')
collinearity_report(df_no_trophies, 'Trophy-Free Dataset')
```

    
    === Collinearity report for Full Dataset ===
    Pairs with |Pearson| > 0.85:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature_1</th>
      <th>feature_2</th>
      <th>corr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>town_hall_level</td>
      <td>exp_level</td>
      <td>0.891220</td>
    </tr>
    <tr>
      <th>1</th>
      <td>exp_level</td>
      <td>builder_base_trophies</td>
      <td>0.873612</td>
    </tr>
    <tr>
      <th>2</th>
      <td>exp_level</td>
      <td>best_trophies</td>
      <td>0.934167</td>
    </tr>
    <tr>
      <th>3</th>
      <td>exp_level</td>
      <td>builder_hall_level</td>
      <td>0.851683</td>
    </tr>
    <tr>
      <th>4</th>
      <td>builder_base_trophies</td>
      <td>builder_hall_level</td>
      <td>0.892938</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>96</th>
      <td>donations</td>
      <td>donations_diff_from_clan_mean</td>
      <td>0.980155</td>
    </tr>
    <tr>
      <th>97</th>
      <td>donation_balance</td>
      <td>donations_diff_from_clan_mean</td>
      <td>0.981631</td>
    </tr>
    <tr>
      <th>98</th>
      <td>attack_wins</td>
      <td>attack_wins_diff_from_clan_mean</td>
      <td>0.921821</td>
    </tr>
    <tr>
      <th>99</th>
      <td>combat_activity_total</td>
      <td>attack_wins_diff_from_clan_mean</td>
      <td>0.917247</td>
    </tr>
    <tr>
      <th>100</th>
      <td>defense_wins</td>
      <td>defense_wins_diff_from_clan_mean</td>
      <td>0.954631</td>
    </tr>
  </tbody>
</table>
<p>101 rows × 3 columns</p>
</div>


    
    Pairs with |Spearman| > 0.85:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature_1</th>
      <th>feature_2</th>
      <th>corr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>town_hall_level</td>
      <td>exp_level</td>
      <td>0.897365</td>
    </tr>
    <tr>
      <th>1</th>
      <td>exp_level</td>
      <td>builder_base_trophies</td>
      <td>0.889479</td>
    </tr>
    <tr>
      <th>2</th>
      <td>exp_level</td>
      <td>best_trophies</td>
      <td>0.939114</td>
    </tr>
    <tr>
      <th>3</th>
      <td>exp_level</td>
      <td>war_stars</td>
      <td>0.879204</td>
    </tr>
    <tr>
      <th>4</th>
      <td>town_hall_level</td>
      <td>builder_hall_level</td>
      <td>0.850558</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>134</th>
      <td>war_stars_ratio_to_clan_mean</td>
      <td>war_stars_clan_pct</td>
      <td>0.888435</td>
    </tr>
    <tr>
      <th>135</th>
      <td>attack_wins</td>
      <td>attack_wins_ratio_to_clan_mean</td>
      <td>0.999872</td>
    </tr>
    <tr>
      <th>136</th>
      <td>defense_wins</td>
      <td>defense_wins_ratio_to_clan_mean</td>
      <td>0.999722</td>
    </tr>
    <tr>
      <th>137</th>
      <td>combat_activity_total</td>
      <td>defense_wins_ratio_to_clan_mean</td>
      <td>0.941415</td>
    </tr>
    <tr>
      <th>138</th>
      <td>defense_wins_diff_from_clan_mean</td>
      <td>defense_wins_clan_pct</td>
      <td>0.859708</td>
    </tr>
  </tbody>
</table>
<p>139 rows × 3 columns</p>
</div>



    
![png](eda_p2_files/eda_p2_12_4.png)
    



    
![png](eda_p2_files/eda_p2_12_5.png)
    


    
    === Collinearity report for Trophy-Free Dataset ===
    Pairs with |Pearson| > 0.85:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature_1</th>
      <th>feature_2</th>
      <th>corr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>town_hall_level</td>
      <td>exp_level</td>
      <td>0.891220</td>
    </tr>
    <tr>
      <th>1</th>
      <td>exp_level</td>
      <td>builder_base_trophies</td>
      <td>0.873612</td>
    </tr>
    <tr>
      <th>2</th>
      <td>exp_level</td>
      <td>builder_hall_level</td>
      <td>0.851683</td>
    </tr>
    <tr>
      <th>3</th>
      <td>builder_base_trophies</td>
      <td>builder_hall_level</td>
      <td>0.892938</td>
    </tr>
    <tr>
      <th>4</th>
      <td>exp_level</td>
      <td>best_builder_base_trophies</td>
      <td>0.895712</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>90</th>
      <td>donations</td>
      <td>donations_diff_from_clan_mean</td>
      <td>0.980155</td>
    </tr>
    <tr>
      <th>91</th>
      <td>donation_balance</td>
      <td>donations_diff_from_clan_mean</td>
      <td>0.981631</td>
    </tr>
    <tr>
      <th>92</th>
      <td>attack_wins</td>
      <td>attack_wins_diff_from_clan_mean</td>
      <td>0.921821</td>
    </tr>
    <tr>
      <th>93</th>
      <td>combat_activity_total</td>
      <td>attack_wins_diff_from_clan_mean</td>
      <td>0.917247</td>
    </tr>
    <tr>
      <th>94</th>
      <td>defense_wins</td>
      <td>defense_wins_diff_from_clan_mean</td>
      <td>0.954631</td>
    </tr>
  </tbody>
</table>
<p>95 rows × 3 columns</p>
</div>


    
    Pairs with |Spearman| > 0.85:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature_1</th>
      <th>feature_2</th>
      <th>corr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>town_hall_level</td>
      <td>exp_level</td>
      <td>0.897365</td>
    </tr>
    <tr>
      <th>1</th>
      <td>exp_level</td>
      <td>builder_base_trophies</td>
      <td>0.889479</td>
    </tr>
    <tr>
      <th>2</th>
      <td>exp_level</td>
      <td>war_stars</td>
      <td>0.879204</td>
    </tr>
    <tr>
      <th>3</th>
      <td>town_hall_level</td>
      <td>builder_hall_level</td>
      <td>0.850558</td>
    </tr>
    <tr>
      <th>4</th>
      <td>exp_level</td>
      <td>builder_hall_level</td>
      <td>0.873722</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>124</th>
      <td>war_stars_ratio_to_clan_mean</td>
      <td>war_stars_clan_pct</td>
      <td>0.888435</td>
    </tr>
    <tr>
      <th>125</th>
      <td>attack_wins</td>
      <td>attack_wins_ratio_to_clan_mean</td>
      <td>0.999872</td>
    </tr>
    <tr>
      <th>126</th>
      <td>defense_wins</td>
      <td>defense_wins_ratio_to_clan_mean</td>
      <td>0.999722</td>
    </tr>
    <tr>
      <th>127</th>
      <td>combat_activity_total</td>
      <td>defense_wins_ratio_to_clan_mean</td>
      <td>0.941415</td>
    </tr>
    <tr>
      <th>128</th>
      <td>defense_wins_diff_from_clan_mean</td>
      <td>defense_wins_clan_pct</td>
      <td>0.859708</td>
    </tr>
  </tbody>
</table>
<p>129 rows × 3 columns</p>
</div>



    
![png](eda_p2_files/eda_p2_12_10.png)
    



    
![png](eda_p2_files/eda_p2_12_11.png)
    


## 7. Outlier and Extreme Case Detection

IQR-based outlier percentages are reported for every numerical feature. The top 10 extreme records for influential features are inspected qualitatively.


```python
# === 7. Outlier and Extreme Case Detection ===

def iqr_outlier_report(data: pd.DataFrame, name: str) -> pd.DataFrame:
    """Computes IQR outlier percentages for all numerical features."""
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET in numeric_cols:
        numeric_cols.remove(TARGET)
    records = []
    for col in numeric_cols:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        mask = (data[col] < lower_bound) | (data[col] > upper_bound)
        records.append({
            'feature': col,
            'q1': q1,
            'q3': q3,
            'IQR': iqr,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'n_outliers': int(mask.sum()),
            'pct_outliers': 100.0 * mask.sum() / len(data)
        })
    report = pd.DataFrame(records).sort_values('n_outliers', ascending=False)
    print('')
    print(f'=== IQR outlier report for {name} ===')
    display(report)
    return report

outliers_full = iqr_outlier_report(df_full, 'Full Dataset')
outliers_no_trophies = iqr_outlier_report(df_no_trophies, 'Trophy-Free Dataset')

# Qualitative inspection of top 10 extreme records for influential features
for variant_name, data in [('Full', df_full), ('Trophy-Free', df_no_trophies)]:
    print('')
    print(f'=== Top 10 extreme records for {variant_name} variant ===')
    for var in ['donations', 'trophies', 'attack_wins']:
        if var in data.columns:
            top_vals = data.nlargest(10, var)[[var, TARGET]].reset_index(drop=True)
            print(f'Extreme values for {var}:')
            display(top_vals)
```

    
    === IQR outlier report for Full Dataset ===
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>q1</th>
      <th>q3</th>
      <th>IQR</th>
      <th>lower_bound</th>
      <th>upper_bound</th>
      <th>n_outliers</th>
      <th>pct_outliers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16</th>
      <td>donation_balance</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>285424</td>
      <td>34.084100</td>
    </tr>
    <tr>
      <th>70</th>
      <td>attack_wins_diff_from_clan_mean</td>
      <td>-0.045455</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>-0.113636</td>
      <td>0.068182</td>
      <td>206748</td>
      <td>24.688952</td>
    </tr>
    <tr>
      <th>6</th>
      <td>donations</td>
      <td>0.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>-15.000000</td>
      <td>25.000000</td>
      <td>202411</td>
      <td>24.171046</td>
    </tr>
    <tr>
      <th>59</th>
      <td>donations_ratio_to_clan_mean</td>
      <td>0.000000</td>
      <td>0.093371</td>
      <td>0.093371</td>
      <td>-0.140056</td>
      <td>0.233427</td>
      <td>196596</td>
      <td>23.476644</td>
    </tr>
    <tr>
      <th>49</th>
      <td>trophies_diff_from_clan_mean</td>
      <td>-163.257143</td>
      <td>57.800000</td>
      <td>221.057143</td>
      <td>-494.842857</td>
      <td>389.385714</td>
      <td>172236</td>
      <td>20.567678</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>51</th>
      <td>trophies_clan_pct</td>
      <td>0.333333</td>
      <td>0.737171</td>
      <td>0.403838</td>
      <td>-0.272423</td>
      <td>1.342928</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>66</th>
      <td>capital_contributions_clan_pct</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>65</th>
      <td>capital_contributions_ratio_to_clan_mean</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>64</th>
      <td>capital_contributions_diff_from_clan_mean</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>69</th>
      <td>war_stars_clan_pct</td>
      <td>0.285714</td>
      <td>0.777778</td>
      <td>0.492063</td>
      <td>-0.452381</td>
      <td>1.515873</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>76 rows × 8 columns</p>
</div>


    
    === IQR outlier report for Trophy-Free Dataset ===
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>q1</th>
      <th>q3</th>
      <th>IQR</th>
      <th>lower_bound</th>
      <th>upper_bound</th>
      <th>n_outliers</th>
      <th>pct_outliers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>donation_balance</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>285424</td>
      <td>34.084100</td>
    </tr>
    <tr>
      <th>62</th>
      <td>attack_wins_diff_from_clan_mean</td>
      <td>-4.545455e-02</td>
      <td>0.000000e+00</td>
      <td>0.045455</td>
      <td>-1.136364e-01</td>
      <td>6.818182e-02</td>
      <td>206748</td>
      <td>24.688952</td>
    </tr>
    <tr>
      <th>5</th>
      <td>donations</td>
      <td>0.000000e+00</td>
      <td>1.000000e+01</td>
      <td>10.000000</td>
      <td>-1.500000e+01</td>
      <td>2.500000e+01</td>
      <td>202411</td>
      <td>24.171046</td>
    </tr>
    <tr>
      <th>51</th>
      <td>donations_ratio_to_clan_mean</td>
      <td>0.000000e+00</td>
      <td>9.337068e-02</td>
      <td>0.093371</td>
      <td>-1.400560e-01</td>
      <td>2.334267e-01</td>
      <td>196596</td>
      <td>23.476644</td>
    </tr>
    <tr>
      <th>2</th>
      <td>league_id</td>
      <td>2.900000e+07</td>
      <td>2.900000e+07</td>
      <td>0.000000</td>
      <td>2.900000e+07</td>
      <td>2.900000e+07</td>
      <td>169465</td>
      <td>20.236777</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>43</th>
      <td>members</td>
      <td>1.400000e+01</td>
      <td>3.900000e+01</td>
      <td>25.000000</td>
      <td>-2.350000e+01</td>
      <td>7.650000e+01</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>58</th>
      <td>capital_contributions_clan_pct</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>57</th>
      <td>capital_contributions_ratio_to_clan_mean</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>56</th>
      <td>capital_contributions_diff_from_clan_mean</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>61</th>
      <td>war_stars_clan_pct</td>
      <td>2.857143e-01</td>
      <td>7.777778e-01</td>
      <td>0.492063</td>
      <td>-4.523810e-01</td>
      <td>1.515873e+00</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>68 rows × 8 columns</p>
</div>


    
    === Top 10 extreme records for Full variant ===
    Extreme values for donations:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>donations</th>
      <th>clan_rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3161641</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2400715</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2368337</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1266761</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1139948</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1075534</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>833804</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>757437</td>
      <td>5</td>
    </tr>
    <tr>
      <th>8</th>
      <td>573620</td>
      <td>5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>535634</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>


    Extreme values for trophies:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trophies</th>
      <th>clan_rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5479</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5353</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5322</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5315</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5297</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5297</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5293</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5287</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5283</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5282</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


    Extreme values for attack_wins:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>attack_wins</th>
      <th>clan_rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>913</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>910</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>910</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>908</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>907</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>905</td>
      <td>10</td>
    </tr>
    <tr>
      <th>6</th>
      <td>905</td>
      <td>4</td>
    </tr>
    <tr>
      <th>7</th>
      <td>902</td>
      <td>6</td>
    </tr>
    <tr>
      <th>8</th>
      <td>902</td>
      <td>6</td>
    </tr>
    <tr>
      <th>9</th>
      <td>902</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


    
    === Top 10 extreme records for Trophy-Free variant ===
    Extreme values for donations:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>donations</th>
      <th>clan_rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3161641</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2400715</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2368337</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1266761</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1139948</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1075534</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>833804</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>757437</td>
      <td>5</td>
    </tr>
    <tr>
      <th>8</th>
      <td>573620</td>
      <td>5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>535634</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>


    Extreme values for attack_wins:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>attack_wins</th>
      <th>clan_rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>913</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>910</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>910</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>908</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>907</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>905</td>
      <td>10</td>
    </tr>
    <tr>
      <th>6</th>
      <td>905</td>
      <td>4</td>
    </tr>
    <tr>
      <th>7</th>
      <td>902</td>
      <td>6</td>
    </tr>
    <tr>
      <th>8</th>
      <td>902</td>
      <td>6</td>
    </tr>
    <tr>
      <th>9</th>
      <td>902</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


## 8. Modeling Implications and Executive Summary

The findings are consolidated into an executable summary dictionary and explicit preprocessing, validation, pruning, and modeling recommendations are provided.


```python
# === 8. Summary dictionary and recommendations ===
target = df_full[TARGET]
max_pct = (target.value_counts(normalize=True) * 100).max()
min_pct = (target.value_counts(normalize=True) * 100).min()
summary = {
    'target': TARGET,
    'full_rows': df_full.shape[0],
    'full_cols': df_full.shape[1],
    'trophy_free_rows': df_no_trophies.shape[0],
    'trophy_free_cols': df_no_trophies.shape[1],
    'trophy_columns_removed': trophy_cols,
    'target_skewness': float(stats.skew(target.dropna())),
    'target_kurtosis': float(stats.kurtosis(target.dropna())),
    'target_imbalance_ratio': float(max_pct / max(min_pct, 1e-9)) if min_pct > 0 else float('inf'),
    'top_mi_features_full': metrics_full['mi'].head(10).to_dict('records'),
    'top_mi_features_trophy_free': metrics_no_trophies['mi'].head(10).to_dict('records'),
    'full_high_pearson_pairs': collinearity_report_df_full if 'collinearity_report_df_full' in globals() else [],
    'trophy_free_high_pearson_pairs': collinearity_report_df_no_trophies if 'collinearity_report_df_no_trophies' in globals() else [],
    'outlier_max_pct_full': float(outliers_full['pct_outliers'].max()) if len(outliers_full) > 0 else 0.0,
    'outlier_max_pct_trophy_free': float(outliers_no_trophies['pct_outliers'].max()) if len(outliers_no_trophies) > 0 else 0.0
}
print('=== EXECUTIVE SUMMARY EDA P2 ===')
for k, v in summary.items():
    print(f'{k}: {v}')
```

    === EXECUTIVE SUMMARY EDA P2 ===
    target: clan_rank
    full_rows: 837411
    full_cols: 87
    trophy_free_rows: 837411
    trophy_free_cols: 79
    trophy_columns_removed: ['required_trophies', 'best_trophies', 'trophies_diff_from_clan_mean', 'trophies', 'clan_mean_trophies', 'progression_ratio_trophies', 'trophies_clan_pct', 'trophies_ratio_to_clan_mean']
    target_skewness: 1.0297496835595223
    target_kurtosis: 0.3159766696334416
    target_imbalance_ratio: 58.93189557321226
    top_mi_features_full: [{'feature': 'members', 'mutual_info': 0.46767648354904434}, {'feature': 'attack_wins_clan_pct', 'mutual_info': 0.42305198249539266}, {'feature': 'town_hall_level_clan_pct', 'mutual_info': 0.41941942863882}, {'feature': 'defense_wins_clan_pct', 'mutual_info': 0.38832643953887214}, {'feature': 'exp_level_clan_pct', 'mutual_info': 0.35215971151390946}, {'feature': 'donations_clan_pct', 'mutual_info': 0.34875990034386106}, {'feature': 'donations_received_clan_pct', 'mutual_info': 0.30828640101531235}, {'feature': 'town_hall_level_diff_from_clan_mean', 'mutual_info': 0.2862778483643851}, {'feature': 'war_stars_clan_pct', 'mutual_info': 0.2812125050904051}, {'feature': 'trophies_clan_pct', 'mutual_info': 0.27625155494801845}]
    top_mi_features_trophy_free: [{'feature': 'members', 'mutual_info': 0.46841078055025154}, {'feature': 'attack_wins_clan_pct', 'mutual_info': 0.4208511999671387}, {'feature': 'town_hall_level_clan_pct', 'mutual_info': 0.41871426015956814}, {'feature': 'defense_wins_clan_pct', 'mutual_info': 0.38944059747196746}, {'feature': 'exp_level_clan_pct', 'mutual_info': 0.35343277458574107}, {'feature': 'donations_clan_pct', 'mutual_info': 0.34976774547956424}, {'feature': 'donations_received_clan_pct', 'mutual_info': 0.30735919993587224}, {'feature': 'town_hall_level_diff_from_clan_mean', 'mutual_info': 0.2860374931606078}, {'feature': 'war_stars_clan_pct', 'mutual_info': 0.28260388948717186}, {'feature': 'town_hall_level_ratio_to_clan_mean', 'mutual_info': 0.20344622640991794}]
    full_high_pearson_pairs: []
    trophy_free_high_pearson_pairs: []
    outlier_max_pct_full: 34.084099683429045
    outlier_max_pct_trophy_free: 34.084099683429045
    

### Preprocessing Recommendations

- **Imputation**
  - Numeric: median for highly skewed features or when outliers are prevalent; otherwise, mean.
  - Categorical: most frequent category (mode); consider an explicit `"missing"` category if the missing rate is high.
  - Drop columns with more than 40-50% missing values, unless they carry critical domain information.

- **Transformations**
  - Apply `log1p` to severely skewed predictors (`|skew| > 1`) such as donations, loot metrics, or trophy counts.
  - Use `RobustScaler` for features with extreme outliers; `StandardScaler` for the rest.

- **Target Transformation**
  - Based on the skewness diagnostic, transform `clan_rank` with `log1p` if `|skew| > 0.5` and evaluate inverse transform for interpretation.

### Cross-Validation Strategy

- **StratifiedKFold on rank bins**  
  - Bin target values into quantile-based groups and use these bins as stratification labels to preserve distribution in each fold.
- **GroupKFold**  
  - If multiple rows share the same clan entity, use `clan_tag` as the grouping variable to prevent data leakage.
- **Metrics**  
  - `MAE`: scales interpretability in original units.  
  - `RMSE`: penalizes large errors more strongly.  
  - `R²`: proportion of variance explained.

### Feature Pruning

- Remove zero-variance and nearly constant features (`mode frequency >= 0.95`).
- For each collinear pair (`|Pearson| > 0.85` or `|Spearman| > 0.85`), retain the feature with higher Mutual Information.
- If MI is very close, prefer the feature with lower measurement cost or higher interpretability.
- For tree-based models, correlation pruning is less critical, but can reduce training time and improve stability.

### Recommended Baseline Models

1. **Ridge Regression**  
   - `Ridge(alpha=1.0, positive=False)`  
   - Good baseline for regularized linear relationships, especially with high-dimensional data.

2. **Random Forest Regressor**  
   - `RandomForestRegressor(n_estimators=200, random_state=42, max_depth=None)`  
   - Robust to non-linear patterns and outliers; provides feature importance.

3. **XGBoost Regressor**  
   - `XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.05, random_state=42)`  
   - Strong performance with hyperparameter tuning and early stopping.

The above recommendations should directly feed the modeling phase for Problem 2.

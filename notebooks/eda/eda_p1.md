# EDA - Problem 1: Player Role Classification

Exploratory data analysis and diagnostic notebook for the Clash of Clans role classification dataset.

The dataset is expected at `data/processed/role_classification.parquet` and contains one row per player-clan relationship.

## 1. Setup and Data Loading

The necessary libraries are imported and the dataset is loaded from Parquet. Basic integrity and data types are verified.


```python
# === 1. Setup and Data Loading ===
import warnings
warnings.filterwarnings('ignore')

%pip install pyarrow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from pathlib import Path
from collections import defaultdict

sns.set_theme(style="whitegrid", palette="viridis")
%matplotlib inline

# Resolve the dataset path from the current working directory or its parents
def find_project_root() -> Path:
    """Searches for the project root by checking the existence of the parquet file."""
    candidates = [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]
    for cand in candidates:
        if (cand / 'data' / 'datasets' / 'role_classification.parquet').exists():
            return cand
    raise FileNotFoundError(
        "Could not find data/datasets/role_classification.parquet. "
        "Adjust the path or run the notebook from the project root."
    )

root = find_project_root()
DATA_PATH = root / 'data' / 'datasets' / 'role_classification.parquet'
print(f"Project root directory: {root}")
print(f"Dataset: {DATA_PATH}")

# Load the dataset
df = pd.read_parquet(DATA_PATH)
print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
df.head()
```

    Requirement already satisfied: pyarrow in c:\Users\Usuario\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages (25.0.1)
    Note: you may need to restart the kernel to use updated packages.
    

    
    [notice] A new release of pip is available: 26.1.2 -> 26.2.1
    [notice] To update, run: python.exe -m pip install --upgrade pip
    

    Project root directory: c:\Users\Usuario\Desktop\Clash of Clans ML Lab
    Dataset: c:\Users\Usuario\Desktop\Clash of Clans ML Lab\data\datasets\role_classification.parquet
    Dataset loaded with 837411 rows and 66 columns.
    




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
      <th>trophies_diff_from_clan_mean</th>
      <th>trophies_ratio_to_clan_mean</th>
      <th>exp_level_diff_from_clan_mean</th>
      <th>exp_level_ratio_to_clan_mean</th>
      <th>town_hall_level_diff_from_clan_mean</th>
      <th>town_hall_level_ratio_to_clan_mean</th>
      <th>war_stars_diff_from_clan_mean</th>
      <th>war_stars_ratio_to_clan_mean</th>
      <th>...</th>
      <th>hero_mean_completion_ratio</th>
      <th>spell_count</th>
      <th>spell_mean_level</th>
      <th>spell_mean_completion_ratio</th>
      <th>equipment_count</th>
      <th>equipment_mean_level</th>
      <th>equipment_mean_completion_ratio</th>
      <th>achievement_count</th>
      <th>achievement_completion_ratio</th>
      <th>role</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>#Y8GGPRUPR</td>
      <td>#200022JVL</td>
      <td>930.163265</td>
      <td>4.231566</td>
      <td>48.571429</td>
      <td>1.219355</td>
      <td>0.142857</td>
      <td>1.008</td>
      <td>1235.020408</td>
      <td>1.550125</td>
      <td>...</td>
      <td>0.977473</td>
      <td>18.0</td>
      <td>6.166667</td>
      <td>0.835504</td>
      <td>36.0</td>
      <td>20.833333</td>
      <td>0.993827</td>
      <td>53.0</td>
      <td>10.612056</td>
      <td>member</td>
    </tr>
    <tr>
      <th>1</th>
      <td>#9PQV8JPCP</td>
      <td>#200022JVL</td>
      <td>757.163265</td>
      <td>3.630530</td>
      <td>27.571429</td>
      <td>1.124516</td>
      <td>0.142857</td>
      <td>1.008</td>
      <td>1274.020408</td>
      <td>1.567498</td>
      <td>...</td>
      <td>0.978571</td>
      <td>17.0</td>
      <td>7.176471</td>
      <td>0.957049</td>
      <td>39.0</td>
      <td>19.692308</td>
      <td>0.934473</td>
      <td>53.0</td>
      <td>11.132148</td>
      <td>admin</td>
    </tr>
    <tr>
      <th>2</th>
      <td>#P8RLQVJL</td>
      <td>#200022JVL</td>
      <td>571.163265</td>
      <td>2.984331</td>
      <td>69.571429</td>
      <td>1.314194</td>
      <td>0.142857</td>
      <td>1.008</td>
      <td>4128.020408</td>
      <td>2.838779</td>
      <td>...</td>
      <td>1.000000</td>
      <td>18.0</td>
      <td>6.833333</td>
      <td>0.913360</td>
      <td>42.0</td>
      <td>20.285714</td>
      <td>0.941799</td>
      <td>53.0</td>
      <td>15.574746</td>
      <td>admin</td>
    </tr>
    <tr>
      <th>3</th>
      <td>#2VU80892Y</td>
      <td>#200022JVL</td>
      <td>386.163265</td>
      <td>2.341605</td>
      <td>44.571429</td>
      <td>1.201290</td>
      <td>0.142857</td>
      <td>1.008</td>
      <td>1210.020408</td>
      <td>1.538989</td>
      <td>...</td>
      <td>0.973872</td>
      <td>18.0</td>
      <td>5.277778</td>
      <td>0.713745</td>
      <td>36.0</td>
      <td>19.472222</td>
      <td>0.937243</td>
      <td>53.0</td>
      <td>8.975956</td>
      <td>coLeader</td>
    </tr>
    <tr>
      <th>4</th>
      <td>#98CPPQYUC</td>
      <td>#200022JVL</td>
      <td>-175.836735</td>
      <td>0.389109</td>
      <td>28.571429</td>
      <td>1.129032</td>
      <td>0.142857</td>
      <td>1.008</td>
      <td>372.020408</td>
      <td>1.165712</td>
      <td>...</td>
      <td>0.995455</td>
      <td>18.0</td>
      <td>6.277778</td>
      <td>0.843059</td>
      <td>41.0</td>
      <td>18.195122</td>
      <td>0.859982</td>
      <td>53.0</td>
      <td>6.261092</td>
      <td>admin</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 66 columns</p>
</div>




```python
# General integrity information
print("=== General information ===")
df.info(show_counts=True)

print("\n=== Data types ===")
display(df.dtypes.to_frame(name='dtype'))
```

    === General information ===
    <class 'pandas.DataFrame'>
    RangeIndex: 837411 entries, 0 to 837410
    Data columns (total 66 columns):
     #   Column                                     Non-Null Count   Dtype  
    ---  ------                                     --------------   -----  
     0   player_tag                                 837411 non-null  str    
     1   clan_tag                                   837411 non-null  str    
     2   trophies_diff_from_clan_mean               837411 non-null  float64
     3   trophies_ratio_to_clan_mean                837411 non-null  float64
     4   exp_level_diff_from_clan_mean              837411 non-null  float64
     5   exp_level_ratio_to_clan_mean               837411 non-null  float64
     6   town_hall_level_diff_from_clan_mean        837411 non-null  float64
     7   town_hall_level_ratio_to_clan_mean         837411 non-null  float64
     8   war_stars_diff_from_clan_mean              837411 non-null  float64
     9   war_stars_ratio_to_clan_mean               837411 non-null  float64
     10  capital_contributions_diff_from_clan_mean  837411 non-null  float64
     11  capital_contributions_ratio_to_clan_mean   837411 non-null  float64
     12  donations_diff_from_clan_mean              837411 non-null  float64
     13  donations_ratio_to_clan_mean               837411 non-null  float64
     14  donations_received_diff_from_clan_mean     837411 non-null  float64
     15  donations_received_ratio_to_clan_mean      837411 non-null  float64
     16  attack_wins_diff_from_clan_mean            837411 non-null  float64
     17  attack_wins_ratio_to_clan_mean             837411 non-null  float64
     18  defense_wins_diff_from_clan_mean           837411 non-null  float64
     19  defense_wins_ratio_to_clan_mean            837411 non-null  float64
     20  trophies_clan_pct                          837411 non-null  float64
     21  exp_level_clan_pct                         837411 non-null  float64
     22  war_stars_clan_pct                         837411 non-null  float64
     23  clan_level                                 837411 non-null  int64  
     24  clan_points                                837411 non-null  int64  
     25  clan_capital_points                        837411 non-null  int64  
     26  members                                    837411 non-null  int64  
     27  required_trophies                          837411 non-null  int64  
     28  war_frequency                              837411 non-null  str    
     29  war_league                                 837411 non-null  str    
     30  capital_league                             837411 non-null  str    
     31  type                                       837411 non-null  str    
     32  is_family_friendly                         837411 non-null  bool   
     33  town_hall_level                            837411 non-null  float64
     34  exp_level                                  837411 non-null  float64
     35  trophies                                   837411 non-null  float64
     36  best_trophies                              837411 non-null  float64
     37  war_stars                                  837411 non-null  float64
     38  attack_wins                                837411 non-null  float64
     39  defense_wins                               837411 non-null  float64
     40  builder_hall_level                         837411 non-null  float64
     41  builder_base_trophies                      837411 non-null  float64
     42  best_builder_base_trophies                 837411 non-null  float64
     43  donations                                  837411 non-null  float64
     44  donations_received                         837411 non-null  float64
     45  clan_capital_contributions                 837411 non-null  float64
     46  donation_balance                           837411 non-null  float64
     47  donation_ratio                             837411 non-null  float64
     48  combat_activity_total                      837411 non-null  float64
     49  progression_ratio_trophies                 837411 non-null  float64
     50  builder_progression_ratio                  837411 non-null  float64
     51  troop_count                                837411 non-null  float64
     52  troop_mean_level                           837411 non-null  float64
     53  troop_mean_completion_ratio                837411 non-null  float64
     54  hero_count                                 837411 non-null  float64
     55  hero_mean_level                            837411 non-null  float64
     56  hero_mean_completion_ratio                 837411 non-null  float64
     57  spell_count                                837411 non-null  float64
     58  spell_mean_level                           837411 non-null  float64
     59  spell_mean_completion_ratio                837411 non-null  float64
     60  equipment_count                            837411 non-null  float64
     61  equipment_mean_level                       837411 non-null  float64
     62  equipment_mean_completion_ratio            837411 non-null  float64
     63  achievement_count                          837411 non-null  float64
     64  achievement_completion_ratio               837411 non-null  float64
     65  role                                       837411 non-null  str    
    dtypes: bool(1), float64(53), int64(5), str(7)
    memory usage: 514.6 MB
    
    === Data types ===
    


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
      <th>trophies_diff_from_clan_mean</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>trophies_ratio_to_clan_mean</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>exp_level_diff_from_clan_mean</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>equipment_mean_level</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>equipment_mean_completion_ratio</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>achievement_count</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>achievement_completion_ratio</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>role</th>
      <td>str</td>
    </tr>
  </tbody>
</table>
<p>66 rows × 1 columns</p>
</div>


## 2. Basic Dataset Audit

Exact dimensions, data types, missing values, duplicates, and nearly constant features are checked.


```python
# === 2. Basic Dataset Audit ===
print("Exact dimensions:")
print(f"  - Rows: {df.shape[0]}")
print(f"  - Columns: {df.shape[1]}")

# Missing values
missing = df.isna().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'missing': missing, 'percentage': missing_pct})
print("\nMissing values by column (only columns with missing values):")
display(missing_df[missing_df['missing'] > 0].sort_values('percentage', ascending=False))

# Exact duplicate rows
dup_rows = df.duplicated().sum()
print(f"\nExact duplicate rows: {dup_rows}")

# Nearly constant features
threshold = 0.95
constant_features = []
for col in df.columns:
    value_counts = df[col].value_counts(dropna=False)
    top_freq = value_counts.iloc[0] / len(df) if len(value_counts) > 0 else 1.0
    if top_freq >= threshold:
        constant_features.append((col, 'nearly_constant', top_freq))
    if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique(dropna=True) <= 1:
        constant_features.append((col, 'zero_variance', top_freq))

if constant_features:
    print("\nFeatures with a single value or nearly constant (>=95% single value):")
    display(pd.DataFrame(constant_features, columns=['column', 'type', 'mode_frequency']))
else:
    print("\nNo nearly constant features exceeding 95% of a single value were detected.")
```

    Exact dimensions:
      - Rows: 837411
      - Columns: 66
    
    Missing values by column (only columns with missing values):
    


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
  </tbody>
</table>
</div>


    
    Exact duplicate rows: 0
    
    Features with a single value or nearly constant (>=95% single value):
    


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
      <td>capital_contributions_diff_from_clan_mean</td>
      <td>nearly_constant</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>capital_contributions_diff_from_clan_mean</td>
      <td>zero_variance</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>capital_contributions_ratio_to_clan_mean</td>
      <td>nearly_constant</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>capital_contributions_ratio_to_clan_mean</td>
      <td>zero_variance</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>attack_wins_ratio_to_clan_mean</td>
      <td>nearly_constant</td>
      <td>0.977350</td>
    </tr>
    <tr>
      <th>5</th>
      <td>defense_wins_ratio_to_clan_mean</td>
      <td>nearly_constant</td>
      <td>0.961056</td>
    </tr>
    <tr>
      <th>6</th>
      <td>attack_wins</td>
      <td>nearly_constant</td>
      <td>0.977350</td>
    </tr>
    <tr>
      <th>7</th>
      <td>defense_wins</td>
      <td>nearly_constant</td>
      <td>0.961056</td>
    </tr>
    <tr>
      <th>8</th>
      <td>combat_activity_total</td>
      <td>nearly_constant</td>
      <td>0.956284</td>
    </tr>
    <tr>
      <th>9</th>
      <td>achievement_count</td>
      <td>nearly_constant</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>achievement_count</td>
      <td>zero_variance</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>


## 3. Target Analysis (`role`)

The distribution of the target variable is analyzed and class imbalance is diagnosed.


```python
TARGET = 'role'

# Absolute and percentage counts
target_counts = df[TARGET].value_counts(dropna=False)
target_pct = df[TARGET].value_counts(dropna=False, normalize=True) * 100
target_summary = pd.DataFrame({
    'count': target_counts,
    'percentage': target_pct
})
print("Target distribution:")
display(target_summary)

# Bar plot
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x=TARGET, order=target_counts.index)
plt.title(f'Distribution of the target variable: {TARGET}')
plt.xlabel(TARGET)
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Class imbalance diagnostics
print("\nClass imbalance diagnostics:")
max_pct = target_pct.max()
min_pct = target_pct.min()
if max_pct > 60:
    print("- High imbalance: the majority class exceeds 60%.")
    print("- Consider class_weight='balanced', SMOTE, or choose metrics such as F1-macro / PR-AUC instead of accuracy.")
elif max_pct > 45:
    print("- Moderate imbalance: evaluate metric impact and consider robust stratification.")
else:
    print("- Relatively balanced distribution; accuracy/macro-F1 can be monitored.")
```

    Target distribution:
    


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
      <th>count</th>
      <th>percentage</th>
    </tr>
    <tr>
      <th>role</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>member</th>
      <td>346329</td>
      <td>41.357111</td>
    </tr>
    <tr>
      <th>admin</th>
      <td>236271</td>
      <td>28.214461</td>
    </tr>
    <tr>
      <th>coLeader</th>
      <td>202902</td>
      <td>24.229679</td>
    </tr>
    <tr>
      <th>leader</th>
      <td>51909</td>
      <td>6.198748</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](eda_p1_files/eda_p1_7_2.png)
    


    
    Class imbalance diagnostics:
    - Relatively balanced distribution; accuracy/macro-F1 can be monitored.
    

## 4. Functional Feature Analysis

Conceptual grouping of variables, skewness analysis, and distribution of critical variables.


```python
# === 4. Functional Feature Analysis ===

def conceptual_group(col: str) -> str:
    """Assigns a column to a conceptual group based on its name."""
    col_l = col.lower()
    if any(k in col_l for k in ['troop', 'hero', 'spell', 'equipment', 'builder_hall', 'town_hall', 'exp_level', 'achievement']):
        return 'Progression'
    if any(k in col_l for k in ['donat', 'attack', 'defense', 'war_stars', 'versus_battle']):
        return 'Activity'
    if any(k in col_l for k in ['loot', 'gold', 'elixir', 'dark_elixir', 'clan_games']):
        return 'Economy'
    if any(k in col_l for k in ['troph', 'clan_rank', 'war_win', 'clan_war_league', 'legend']):
        return 'Clan/Competition Metrics'
    return 'Other'

group_to_cols = defaultdict(list)
for col in df.columns:
    group_to_cols[conceptual_group(col)].append(col)

print("Conceptual grouping of features:")
for grp, cols in group_to_cols.items():
    print(f"\n{grp}:")
    for c in cols:
        print(f"  - {c}")

# Numeric variables excluding the target
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if TARGET in numeric_cols:
    numeric_cols.remove(TARGET)

# Skewness, percentage of zeros, and percentage of NaN
skew_df = pd.DataFrame({
    'skewness': df[numeric_cols].skew(),
    'zero_pct': (df[numeric_cols] == 0).mean() * 100,
    'nan_pct': df[numeric_cols].isna().mean() * 100
}).sort_values('skewness', key=lambda s: s.abs(), ascending=False)

print("\nSkewness and presence of zeros/NaN in numeric variables:")
display(skew_df)

# Visualization of critical distributions
critical_vars = ['donations', 'attack_wins', 'trophies']
present_critical = [v for v in critical_vars if v in df.columns]
if present_critical:
    fig, axes = plt.subplots(1, len(present_critical), figsize=(5 * len(present_critical), 4))
    if len(present_critical) == 1:
        axes = [axes]
    for ax, var in zip(axes, present_critical):
        sns.histplot(data=df, x=var, kde=True, ax=ax)
        ax.set_title(f'Distribution of {var}')
    plt.tight_layout()
    plt.show()
else:
    print("Expected critical variables (donations, attack_wins, trophies) were not found.")
```

    Conceptual grouping of features:
    
    Other:
      - player_tag
      - clan_tag
      - capital_contributions_diff_from_clan_mean
      - capital_contributions_ratio_to_clan_mean
      - clan_level
      - clan_points
      - clan_capital_points
      - members
      - war_frequency
      - war_league
      - capital_league
      - type
      - is_family_friendly
      - clan_capital_contributions
      - combat_activity_total
      - builder_progression_ratio
      - role
    
    Clan/Competition Metrics:
      - trophies_diff_from_clan_mean
      - trophies_ratio_to_clan_mean
      - trophies_clan_pct
      - required_trophies
      - trophies
      - best_trophies
      - builder_base_trophies
      - best_builder_base_trophies
      - progression_ratio_trophies
    
    Progression:
      - exp_level_diff_from_clan_mean
      - exp_level_ratio_to_clan_mean
      - town_hall_level_diff_from_clan_mean
      - town_hall_level_ratio_to_clan_mean
      - exp_level_clan_pct
      - town_hall_level
      - exp_level
      - builder_hall_level
      - troop_count
      - troop_mean_level
      - troop_mean_completion_ratio
      - hero_count
      - hero_mean_level
      - hero_mean_completion_ratio
      - spell_count
      - spell_mean_level
      - spell_mean_completion_ratio
      - equipment_count
      - equipment_mean_level
      - equipment_mean_completion_ratio
      - achievement_count
      - achievement_completion_ratio
    
    Activity:
      - war_stars_diff_from_clan_mean
      - war_stars_ratio_to_clan_mean
      - donations_diff_from_clan_mean
      - donations_ratio_to_clan_mean
      - donations_received_diff_from_clan_mean
      - donations_received_ratio_to_clan_mean
      - attack_wins_diff_from_clan_mean
      - attack_wins_ratio_to_clan_mean
      - defense_wins_diff_from_clan_mean
      - defense_wins_ratio_to_clan_mean
      - war_stars_clan_pct
      - war_stars
      - attack_wins
      - defense_wins
      - donations
      - donations_received
      - donation_balance
      - donation_ratio
    
    Skewness and presence of zeros/NaN in numeric variables:
    


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
      <th>skewness</th>
      <th>zero_pct</th>
      <th>nan_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>donation_ratio</th>
      <td>883.319323</td>
      <td>80.792108</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>donations</th>
      <td>391.451305</td>
      <td>73.318478</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>donations_diff_from_clan_mean</th>
      <td>365.867788</td>
      <td>30.106602</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>donation_balance</th>
      <td>349.795288</td>
      <td>65.915900</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>donations_received</th>
      <td>151.429835</td>
      <td>73.068063</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>donations_received_diff_from_clan_mean</th>
      <td>79.421448</td>
      <td>29.644105</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>attack_wins</th>
      <td>35.560693</td>
      <td>97.735043</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>combat_activity_total</th>
      <td>34.937844</td>
      <td>95.628431</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>attack_wins_diff_from_clan_mean</th>
      <td>28.944551</td>
      <td>71.602714</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>defense_wins</th>
      <td>14.548574</td>
      <td>96.105616</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>defense_wins_diff_from_clan_mean</th>
      <td>13.505352</td>
      <td>59.011883</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>attack_wins_ratio_to_clan_mean</th>
      <td>12.219311</td>
      <td>97.735043</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>defense_wins_ratio_to_clan_mean</th>
      <td>9.851055</td>
      <td>96.105616</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>progression_ratio_trophies</th>
      <td>7.602090</td>
      <td>62.936479</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>donations_ratio_to_clan_mean</th>
      <td>6.986924</td>
      <td>73.309522</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>donations_received_ratio_to_clan_mean</th>
      <td>6.338193</td>
      <td>73.063884</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>trophies_ratio_to_clan_mean</th>
      <td>6.261242</td>
      <td>53.461204</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>trophies</th>
      <td>5.361737</td>
      <td>62.893490</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>war_stars_ratio_to_clan_mean</th>
      <td>4.968075</td>
      <td>9.734288</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>builder_progression_ratio</th>
      <td>-3.547297</td>
      <td>3.261242</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>achievement_completion_ratio</th>
      <td>2.436429</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>war_stars</th>
      <td>2.089524</td>
      <td>9.734288</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>clan_capital_contributions</th>
      <td>2.086583</td>
      <td>11.825734</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>trophies_diff_from_clan_mean</th>
      <td>1.969441</td>
      <td>7.265608</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>clan_points</th>
      <td>1.652575</td>
      <td>5.742342</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>exp_level_ratio_to_clan_mean</th>
      <td>1.508346</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>required_trophies</th>
      <td>1.389214</td>
      <td>46.138037</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>war_stars_diff_from_clan_mean</th>
      <td>1.196036</td>
      <td>0.601258</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>spell_count</th>
      <td>-1.023029</td>
      <td>2.684703</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>clan_capital_points</th>
      <td>0.894702</td>
      <td>26.284226</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>hero_count</th>
      <td>-0.802314</td>
      <td>5.297399</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>builder_hall_level</th>
      <td>-0.758551</td>
      <td>3.238792</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>spell_mean_level</th>
      <td>-0.534585</td>
      <td>2.684703</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>town_hall_level</th>
      <td>-0.533949</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>town_hall_level_diff_from_clan_mean</th>
      <td>-0.488968</td>
      <td>2.606844</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>troop_mean_completion_ratio</th>
      <td>0.482339</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>clan_level</th>
      <td>0.413863</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>trophies_clan_pct</th>
      <td>0.412409</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>equipment_mean_level</th>
      <td>0.301593</td>
      <td>13.530393</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>hero_mean_completion_ratio</th>
      <td>0.282757</td>
      <td>5.297399</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>troop_mean_level</th>
      <td>0.217141</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>equipment_mean_completion_ratio</th>
      <td>0.207833</td>
      <td>13.530393</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>best_builder_base_trophies</th>
      <td>0.201818</td>
      <td>3.261242</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>spell_mean_completion_ratio</th>
      <td>-0.172377</td>
      <td>2.684703</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>members</th>
      <td>0.156845</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>equipment_count</th>
      <td>-0.141249</td>
      <td>13.530393</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>town_hall_level_ratio_to_clan_mean</th>
      <td>0.130784</td>
      <td>0.006448</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>troop_count</th>
      <td>-0.102463</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>best_trophies</th>
      <td>0.091443</td>
      <td>1.868736</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>exp_level_diff_from_clan_mean</th>
      <td>-0.090196</td>
      <td>0.186766</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>exp_level</th>
      <td>0.050676</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>hero_mean_level</th>
      <td>-0.023540</td>
      <td>5.297399</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>war_stars_clan_pct</th>
      <td>0.020411</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>builder_base_trophies</th>
      <td>-0.019563</td>
      <td>3.251808</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>exp_level_clan_pct</th>
      <td>-0.004262</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>capital_contributions_ratio_to_clan_mean</th>
      <td>0.000000</td>
      <td>100.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>capital_contributions_diff_from_clan_mean</th>
      <td>0.000000</td>
      <td>100.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>achievement_count</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](eda_p1_files/eda_p1_9_2.png)
    


## 5. Feature–Target Relationship

Class separation is analyzed and nonlinear predictive power is estimated using Mutual Information and Kruskal-Wallis.


```python
# === 5. Feature–Target Relationship ===

# Encode target for Mutual Information
target_encoded = pd.factorize(df[TARGET])[0]

# Prepare numerical data by temporarily imputing the median only for MI
df_num_for_mi = df[numeric_cols].copy()
for col in numeric_cols:
    if df_num_for_mi[col].isna().any():
        df_num_for_mi[col] = df_num_for_mi[col].fillna(df_num_for_mi[col].median())

# Mutual Information
mi_scores = mutual_info_classif(df_num_for_mi, target_encoded, random_state=42)
mi_df = pd.DataFrame({'feature': numeric_cols, 'mutual_info': mi_scores})
mi_df = mi_df.sort_values('mutual_info', ascending=False).reset_index(drop=True)
print("Nonlinear predictive power (Mutual Information) by numeric feature:")
display(mi_df.head(20))

# Kruskal-Wallis
kw_results = []
for col in numeric_cols:
    groups = []
    for role in df[TARGET].dropna().unique():
        group_vals = df.loc[df[TARGET] == role, col].dropna()
        if len(group_vals) > 5:
            groups.append(group_vals)
    if len(groups) >= 2:
        try:
            stat, p = stats.kruskal(*groups)
            kw_results.append((col, stat, p))
        except Exception:
            pass

kw_df = pd.DataFrame(kw_results, columns=['feature', 'kruskal_stat', 'p_value'])
kw_df['p_bonferroni'] = kw_df['p_value'] * len(numeric_cols)
kw_df = kw_df.sort_values('p_value').reset_index(drop=True)
print("\nKruskal-Wallis test (numeric variables vs role):")
display(kw_df.head(20))

# Boxplots of features with the highest Mutual Information
top_mi_features = mi_df.head(6)['feature'].tolist()
if top_mi_features:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for ax, var in zip(axes, top_mi_features):
        sns.boxplot(data=df, x=TARGET, y=var, ax=ax)
        ax.set_title(f'{var} by {TARGET}')
        ax.set_xlabel(TARGET)
        ax.set_ylabel(var)
        plt.setp(ax.get_xticklabels(), rotation=45)
    # Hide empty axes if fewer than 6 features
    for ax in axes[len(top_mi_features):]:
        ax.axis('off')
    plt.tight_layout()
    plt.show()
else:
    print("No numeric features were found to plot.")
```

    Nonlinear predictive power (Mutual Information) by numeric feature:
    


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
      <td>donations_diff_from_clan_mean</td>
      <td>0.143917</td>
    </tr>
    <tr>
      <th>1</th>
      <td>donations_received_diff_from_clan_mean</td>
      <td>0.130772</td>
    </tr>
    <tr>
      <th>2</th>
      <td>town_hall_level_ratio_to_clan_mean</td>
      <td>0.114983</td>
    </tr>
    <tr>
      <th>3</th>
      <td>trophies_diff_from_clan_mean</td>
      <td>0.105480</td>
    </tr>
    <tr>
      <th>4</th>
      <td>war_stars_diff_from_clan_mean</td>
      <td>0.090231</td>
    </tr>
    <tr>
      <th>5</th>
      <td>war_stars_ratio_to_clan_mean</td>
      <td>0.089187</td>
    </tr>
    <tr>
      <th>6</th>
      <td>exp_level_ratio_to_clan_mean</td>
      <td>0.085726</td>
    </tr>
    <tr>
      <th>7</th>
      <td>exp_level_diff_from_clan_mean</td>
      <td>0.085472</td>
    </tr>
    <tr>
      <th>8</th>
      <td>war_stars_clan_pct</td>
      <td>0.085422</td>
    </tr>
    <tr>
      <th>9</th>
      <td>town_hall_level_diff_from_clan_mean</td>
      <td>0.083063</td>
    </tr>
    <tr>
      <th>10</th>
      <td>exp_level_clan_pct</td>
      <td>0.078907</td>
    </tr>
    <tr>
      <th>11</th>
      <td>clan_capital_points</td>
      <td>0.068985</td>
    </tr>
    <tr>
      <th>12</th>
      <td>war_stars</td>
      <td>0.063114</td>
    </tr>
    <tr>
      <th>13</th>
      <td>clan_points</td>
      <td>0.062788</td>
    </tr>
    <tr>
      <th>14</th>
      <td>achievement_count</td>
      <td>0.053759</td>
    </tr>
    <tr>
      <th>15</th>
      <td>spell_mean_completion_ratio</td>
      <td>0.045886</td>
    </tr>
    <tr>
      <th>16</th>
      <td>members</td>
      <td>0.043157</td>
    </tr>
    <tr>
      <th>17</th>
      <td>troop_mean_completion_ratio</td>
      <td>0.043051</td>
    </tr>
    <tr>
      <th>18</th>
      <td>spell_count</td>
      <td>0.042943</td>
    </tr>
    <tr>
      <th>19</th>
      <td>exp_level</td>
      <td>0.042457</td>
    </tr>
  </tbody>
</table>
</div>


    
    Kruskal-Wallis test (numeric variables vs role):
    


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
      <th>kruskal_stat</th>
      <th>p_value</th>
      <th>p_bonferroni</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>trophies_diff_from_clan_mean</td>
      <td>1924.813639</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>exp_level_diff_from_clan_mean</td>
      <td>78440.927836</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>exp_level_ratio_to_clan_mean</td>
      <td>77140.020501</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>town_hall_level_diff_from_clan_mean</td>
      <td>71404.451990</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>war_stars_diff_from_clan_mean</td>
      <td>73333.481528</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>town_hall_level_ratio_to_clan_mean</td>
      <td>69302.030722</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>war_stars_ratio_to_clan_mean</td>
      <td>101672.692785</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>donations_diff_from_clan_mean</td>
      <td>36690.786953</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>attack_wins_diff_from_clan_mean</td>
      <td>19159.129773</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>donations_ratio_to_clan_mean</td>
      <td>43229.232092</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>donations_received_diff_from_clan_mean</td>
      <td>25414.381909</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>donations_received_ratio_to_clan_mean</td>
      <td>31949.707701</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>defense_wins_diff_from_clan_mean</td>
      <td>19114.545755</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>attack_wins_ratio_to_clan_mean</td>
      <td>3961.963250</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>defense_wins_ratio_to_clan_mean</td>
      <td>4800.749613</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>trophies_clan_pct</td>
      <td>4220.673051</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>town_hall_level</td>
      <td>53851.726653</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>exp_level_clan_pct</td>
      <td>86357.337833</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>war_stars_clan_pct</td>
      <td>95335.942744</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>clan_level</td>
      <td>32877.533647</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](eda_p1_files/eda_p1_11_4.png)
    


## 6. Multicollinearity and Redundancy

Pearson and Spearman correlation matrices are computed to identify pairs with |r| > 0.85.


```python
# === 6. Multicollinearity and Redundancy ===

df_corr = df[numeric_cols].copy()
df_corr = df_corr.replace([np.inf, -np.inf], np.nan)
for col in df_corr.columns:
    if df_corr[col].isna().any():
        df_corr[col] = df_corr[col].fillna(df_corr[col].median())

pearson_corr = df_corr.corr(method='pearson')
spearman_corr = df_corr.corr(method='spearman')

threshold_corr = 0.85

def get_high_corr_pairs(corr_matrix, threshold=threshold_corr):
    """Returns pairs of variables with correlation above the threshold (upper triangle)."""
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

print("Pairs with |Pearson| > 0.85:")
display(pearson_pairs if len(pearson_pairs) > 0 else None)

print("\nPairs with |Spearman| > 0.85:")
display(spearman_pairs if len(spearman_pairs) > 0 else None)

# Pearson correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(pearson_corr, annot=False, cmap='coolwarm', center=0, square=True)
plt.title('Pearson correlation matrix (numeric variables)')
plt.tight_layout()
plt.show()
```

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
      <td>exp_level_diff_from_clan_mean</td>
      <td>exp_level_ratio_to_clan_mean</td>
      <td>0.900149</td>
    </tr>
    <tr>
      <th>1</th>
      <td>exp_level_ratio_to_clan_mean</td>
      <td>town_hall_level_ratio_to_clan_mean</td>
      <td>0.859917</td>
    </tr>
    <tr>
      <th>2</th>
      <td>town_hall_level_diff_from_clan_mean</td>
      <td>town_hall_level_ratio_to_clan_mean</td>
      <td>0.964016</td>
    </tr>
    <tr>
      <th>3</th>
      <td>exp_level_diff_from_clan_mean</td>
      <td>exp_level_clan_pct</td>
      <td>0.894825</td>
    </tr>
    <tr>
      <th>4</th>
      <td>clan_points</td>
      <td>clan_capital_points</td>
      <td>0.891741</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>93</th>
      <td>hero_mean_level</td>
      <td>equipment_mean_completion_ratio</td>
      <td>0.862480</td>
    </tr>
    <tr>
      <th>94</th>
      <td>hero_mean_completion_ratio</td>
      <td>equipment_mean_completion_ratio</td>
      <td>0.883966</td>
    </tr>
    <tr>
      <th>95</th>
      <td>equipment_count</td>
      <td>equipment_mean_completion_ratio</td>
      <td>0.901594</td>
    </tr>
    <tr>
      <th>96</th>
      <td>equipment_mean_level</td>
      <td>equipment_mean_completion_ratio</td>
      <td>0.997023</td>
    </tr>
    <tr>
      <th>97</th>
      <td>troop_mean_completion_ratio</td>
      <td>achievement_completion_ratio</td>
      <td>0.868364</td>
    </tr>
  </tbody>
</table>
<p>98 rows × 3 columns</p>
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
      <td>exp_level_diff_from_clan_mean</td>
      <td>exp_level_ratio_to_clan_mean</td>
      <td>0.971851</td>
    </tr>
    <tr>
      <th>1</th>
      <td>town_hall_level_diff_from_clan_mean</td>
      <td>town_hall_level_ratio_to_clan_mean</td>
      <td>0.993523</td>
    </tr>
    <tr>
      <th>2</th>
      <td>exp_level_diff_from_clan_mean</td>
      <td>exp_level_clan_pct</td>
      <td>0.933134</td>
    </tr>
    <tr>
      <th>3</th>
      <td>exp_level_ratio_to_clan_mean</td>
      <td>exp_level_clan_pct</td>
      <td>0.916646</td>
    </tr>
    <tr>
      <th>4</th>
      <td>war_stars_diff_from_clan_mean</td>
      <td>war_stars_clan_pct</td>
      <td>0.859458</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>127</th>
      <td>spell_count</td>
      <td>achievement_completion_ratio</td>
      <td>0.930316</td>
    </tr>
    <tr>
      <th>128</th>
      <td>spell_mean_completion_ratio</td>
      <td>achievement_completion_ratio</td>
      <td>0.864668</td>
    </tr>
    <tr>
      <th>129</th>
      <td>equipment_count</td>
      <td>achievement_completion_ratio</td>
      <td>0.894912</td>
    </tr>
    <tr>
      <th>130</th>
      <td>equipment_mean_level</td>
      <td>achievement_completion_ratio</td>
      <td>0.895241</td>
    </tr>
    <tr>
      <th>131</th>
      <td>equipment_mean_completion_ratio</td>
      <td>achievement_completion_ratio</td>
      <td>0.889145</td>
    </tr>
  </tbody>
</table>
<p>132 rows × 3 columns</p>
</div>



    
![png](eda_p1_files/eda_p1_13_4.png)
    


## 7. Outlier and Extreme Case Detection

Outliers are identified using the IQR rule and extreme examples are documented.


```python
# === 7. Outlier and Extreme Case Detection ===

outlier_records = []
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    mask = (df[col] < lower_bound) | (df[col] > upper_bound)
    n_outliers = int(mask.sum())
    outlier_records.append({
        'feature': col,
        'q1': q1,
        'q3': q3,
        'IQR': iqr,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'n_outliers': n_outliers,
        'pct_outliers': 100.0 * n_outliers / len(df)
    })

outlier_df = pd.DataFrame(outlier_records).sort_values('n_outliers', ascending=False)
print("IQR outlier summary:")
display(outlier_df)

# Qualitative analysis for critical variables
for var in ['donations', 'trophies', 'attack_wins']:
    if var in df.columns:
        top_vals = df.nlargest(10, var)[[var, TARGET]].reset_index(drop=True)
        print(f"\nTop 10 extreme values of {var}:")
        display(top_vals)
```

    IQR outlier summary:
    


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
      <th>39</th>
      <td>donation_balance</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>285424</td>
      <td>34.084100</td>
    </tr>
    <tr>
      <th>14</th>
      <td>attack_wins_diff_from_clan_mean</td>
      <td>-0.045455</td>
      <td>0.000000e+00</td>
      <td>4.545455e-02</td>
      <td>-1.136364e-01</td>
      <td>6.818182e-02</td>
      <td>206706</td>
      <td>24.683937</td>
    </tr>
    <tr>
      <th>36</th>
      <td>donations</td>
      <td>0.000000</td>
      <td>1.000000e+01</td>
      <td>1.000000e+01</td>
      <td>-1.500000e+01</td>
      <td>2.500000e+01</td>
      <td>202354</td>
      <td>24.164240</td>
    </tr>
    <tr>
      <th>11</th>
      <td>donations_ratio_to_clan_mean</td>
      <td>0.000000</td>
      <td>9.341472e-02</td>
      <td>9.341472e-02</td>
      <td>-1.401221e-01</td>
      <td>2.335368e-01</td>
      <td>196591</td>
      <td>23.476047</td>
    </tr>
    <tr>
      <th>0</th>
      <td>trophies_diff_from_clan_mean</td>
      <td>-163.000000</td>
      <td>5.809762e+01</td>
      <td>2.210976e+02</td>
      <td>-4.946464e+02</td>
      <td>3.897440e+02</td>
      <td>172131</td>
      <td>20.555140</td>
    </tr>
    <tr>
      <th>13</th>
      <td>donations_received_ratio_to_clan_mean</td>
      <td>0.000000</td>
      <td>3.188769e-01</td>
      <td>3.188769e-01</td>
      <td>-4.783153e-01</td>
      <td>7.971922e-01</td>
      <td>169079</td>
      <td>20.190683</td>
    </tr>
    <tr>
      <th>12</th>
      <td>donations_received_diff_from_clan_mean</td>
      <td>-65.708333</td>
      <td>0.000000e+00</td>
      <td>6.570833e+01</td>
      <td>-1.642708e+02</td>
      <td>9.856250e+01</td>
      <td>165152</td>
      <td>19.721738</td>
    </tr>
    <tr>
      <th>40</th>
      <td>donation_ratio</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>160849</td>
      <td>19.207892</td>
    </tr>
    <tr>
      <th>16</th>
      <td>defense_wins_diff_from_clan_mean</td>
      <td>-0.172414</td>
      <td>0.000000e+00</td>
      <td>1.724138e-01</td>
      <td>-4.310345e-01</td>
      <td>2.586207e-01</td>
      <td>159135</td>
      <td>19.003213</td>
    </tr>
    <tr>
      <th>37</th>
      <td>donations_received</td>
      <td>0.000000</td>
      <td>4.550000e+01</td>
      <td>4.550000e+01</td>
      <td>-6.825000e+01</td>
      <td>1.137500e+02</td>
      <td>149080</td>
      <td>17.802489</td>
    </tr>
    <tr>
      <th>10</th>
      <td>donations_diff_from_clan_mean</td>
      <td>-72.736842</td>
      <td>0.000000e+00</td>
      <td>7.273684e+01</td>
      <td>-1.818421e+02</td>
      <td>1.091053e+02</td>
      <td>148160</td>
      <td>17.692626</td>
    </tr>
    <tr>
      <th>42</th>
      <td>progression_ratio_trophies</td>
      <td>0.000000</td>
      <td>3.977480e-02</td>
      <td>3.977480e-02</td>
      <td>-5.966220e-02</td>
      <td>9.943700e-02</td>
      <td>147627</td>
      <td>17.628978</td>
    </tr>
    <tr>
      <th>28</th>
      <td>trophies</td>
      <td>0.000000</td>
      <td>1.240000e+02</td>
      <td>1.240000e+02</td>
      <td>-1.860000e+02</td>
      <td>3.100000e+02</td>
      <td>139105</td>
      <td>16.611318</td>
    </tr>
    <tr>
      <th>6</th>
      <td>war_stars_diff_from_clan_mean</td>
      <td>-361.702941</td>
      <td>2.136667e+02</td>
      <td>5.753696e+02</td>
      <td>-1.224757e+03</td>
      <td>1.076721e+03</td>
      <td>98646</td>
      <td>11.779879</td>
    </tr>
    <tr>
      <th>43</th>
      <td>builder_progression_ratio</td>
      <td>0.905104</td>
      <td>9.843750e-01</td>
      <td>7.927118e-02</td>
      <td>7.861970e-01</td>
      <td>1.103282e+00</td>
      <td>73423</td>
      <td>8.767857</td>
    </tr>
    <tr>
      <th>38</th>
      <td>clan_capital_contributions</td>
      <td>25602.500000</td>
      <td>1.315764e+06</td>
      <td>1.290161e+06</td>
      <td>-1.909639e+06</td>
      <td>3.251005e+06</td>
      <td>71036</td>
      <td>8.482812</td>
    </tr>
    <tr>
      <th>1</th>
      <td>trophies_ratio_to_clan_mean</td>
      <td>0.000000</td>
      <td>1.253235e+00</td>
      <td>1.253235e+00</td>
      <td>-1.879853e+00</td>
      <td>3.133088e+00</td>
      <td>62505</td>
      <td>7.464077</td>
    </tr>
    <tr>
      <th>30</th>
      <td>war_stars</td>
      <td>55.000000</td>
      <td>1.110000e+03</td>
      <td>1.055000e+03</td>
      <td>-1.527500e+03</td>
      <td>2.692500e+03</td>
      <td>59121</td>
      <td>7.059974</td>
    </tr>
    <tr>
      <th>5</th>
      <td>town_hall_level_ratio_to_clan_mean</td>
      <td>0.905660</td>
      <td>1.101928e+00</td>
      <td>1.962680e-01</td>
      <td>6.112584e-01</td>
      <td>1.396330e+00</td>
      <td>58675</td>
      <td>7.006715</td>
    </tr>
    <tr>
      <th>57</th>
      <td>achievement_completion_ratio</td>
      <td>0.611019</td>
      <td>3.471336e+00</td>
      <td>2.860317e+00</td>
      <td>-3.679457e+00</td>
      <td>7.761811e+00</td>
      <td>47405</td>
      <td>5.660900</td>
    </tr>
    <tr>
      <th>50</th>
      <td>spell_count</td>
      <td>11.000000</td>
      <td>1.700000e+01</td>
      <td>6.000000e+00</td>
      <td>2.000000e+00</td>
      <td>2.600000e+01</td>
      <td>38442</td>
      <td>4.590577</td>
    </tr>
    <tr>
      <th>41</th>
      <td>combat_activity_total</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>36608</td>
      <td>4.371569</td>
    </tr>
    <tr>
      <th>4</th>
      <td>town_hall_level_diff_from_clan_mean</td>
      <td>-1.181818</td>
      <td>1.360000e+00</td>
      <td>2.541818e+00</td>
      <td>-4.994545e+00</td>
      <td>5.172727e+00</td>
      <td>36031</td>
      <td>4.302666</td>
    </tr>
    <tr>
      <th>17</th>
      <td>defense_wins_ratio_to_clan_mean</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>32612</td>
      <td>3.894384</td>
    </tr>
    <tr>
      <th>32</th>
      <td>defense_wins</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>32612</td>
      <td>3.894384</td>
    </tr>
    <tr>
      <th>7</th>
      <td>war_stars_ratio_to_clan_mean</td>
      <td>0.184630</td>
      <td>1.419521e+00</td>
      <td>1.234891e+00</td>
      <td>-1.667706e+00</td>
      <td>3.271857e+00</td>
      <td>32247</td>
      <td>3.850797</td>
    </tr>
    <tr>
      <th>25</th>
      <td>required_trophies</td>
      <td>0.000000</td>
      <td>1.800000e+03</td>
      <td>1.800000e+03</td>
      <td>-2.700000e+03</td>
      <td>4.500000e+03</td>
      <td>29991</td>
      <td>3.581396</td>
    </tr>
    <tr>
      <th>3</th>
      <td>exp_level_ratio_to_clan_mean</td>
      <td>0.760369</td>
      <td>1.203098e+00</td>
      <td>4.427298e-01</td>
      <td>9.627391e-02</td>
      <td>1.867193e+00</td>
      <td>26324</td>
      <td>3.143498</td>
    </tr>
    <tr>
      <th>51</th>
      <td>spell_mean_level</td>
      <td>3.333333</td>
      <td>5.200000e+00</td>
      <td>1.866667e+00</td>
      <td>5.333333e-01</td>
      <td>8.000000e+00</td>
      <td>22491</td>
      <td>2.685778</td>
    </tr>
    <tr>
      <th>15</th>
      <td>attack_wins_ratio_to_clan_mean</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>18967</td>
      <td>2.264957</td>
    </tr>
    <tr>
      <th>31</th>
      <td>attack_wins</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>18967</td>
      <td>2.264957</td>
    </tr>
    <tr>
      <th>2</th>
      <td>exp_level_diff_from_clan_mean</td>
      <td>-28.300000</td>
      <td>2.940863e+01</td>
      <td>5.770863e+01</td>
      <td>-1.148629e+02</td>
      <td>1.159716e+02</td>
      <td>16616</td>
      <td>1.984211</td>
    </tr>
    <tr>
      <th>22</th>
      <td>clan_points</td>
      <td>5200.000000</td>
      <td>6.475000e+04</td>
      <td>5.955000e+04</td>
      <td>-8.412500e+04</td>
      <td>1.540750e+05</td>
      <td>12439</td>
      <td>1.485412</td>
    </tr>
    <tr>
      <th>26</th>
      <td>town_hall_level</td>
      <td>11.000000</td>
      <td>1.600000e+01</td>
      <td>5.000000e+00</td>
      <td>3.500000e+00</td>
      <td>2.350000e+01</td>
      <td>8066</td>
      <td>0.963207</td>
    </tr>
    <tr>
      <th>35</th>
      <td>best_builder_base_trophies</td>
      <td>1531.000000</td>
      <td>3.343000e+03</td>
      <td>1.812000e+03</td>
      <td>-1.187000e+03</td>
      <td>6.061000e+03</td>
      <td>1725</td>
      <td>0.205992</td>
    </tr>
    <tr>
      <th>23</th>
      <td>clan_capital_points</td>
      <td>0.000000</td>
      <td>2.294000e+03</td>
      <td>2.294000e+03</td>
      <td>-3.441000e+03</td>
      <td>5.735000e+03</td>
      <td>90</td>
      <td>0.010747</td>
    </tr>
    <tr>
      <th>27</th>
      <td>exp_level</td>
      <td>84.000000</td>
      <td>2.030000e+02</td>
      <td>1.190000e+02</td>
      <td>-9.450000e+01</td>
      <td>3.815000e+02</td>
      <td>69</td>
      <td>0.008240</td>
    </tr>
    <tr>
      <th>20</th>
      <td>war_stars_clan_pct</td>
      <td>0.285714</td>
      <td>7.777778e-01</td>
      <td>4.920635e-01</td>
      <td>-4.523810e-01</td>
      <td>1.515873e+00</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>trophies_clan_pct</td>
      <td>0.333333</td>
      <td>7.380952e-01</td>
      <td>4.047619e-01</td>
      <td>-2.738095e-01</td>
      <td>1.345238e+00</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>capital_contributions_ratio_to_clan_mean</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>capital_contributions_diff_from_clan_mean</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>clan_level</td>
      <td>6.000000</td>
      <td>2.000000e+01</td>
      <td>1.400000e+01</td>
      <td>-1.500000e+01</td>
      <td>4.100000e+01</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>members</td>
      <td>14.000000</td>
      <td>3.900000e+01</td>
      <td>2.500000e+01</td>
      <td>-2.350000e+01</td>
      <td>7.650000e+01</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>exp_level_clan_pct</td>
      <td>0.285714</td>
      <td>7.826087e-01</td>
      <td>4.968944e-01</td>
      <td>-4.596273e-01</td>
      <td>1.527950e+00</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>29</th>
      <td>best_trophies</td>
      <td>1442.000000</td>
      <td>4.495000e+03</td>
      <td>3.053000e+03</td>
      <td>-3.137500e+03</td>
      <td>9.074500e+03</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>33</th>
      <td>builder_hall_level</td>
      <td>5.000000</td>
      <td>1.000000e+01</td>
      <td>5.000000e+00</td>
      <td>-2.500000e+00</td>
      <td>1.750000e+01</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>34</th>
      <td>builder_base_trophies</td>
      <td>1457.000000</td>
      <td>3.076000e+03</td>
      <td>1.619000e+03</td>
      <td>-9.715000e+02</td>
      <td>5.504500e+03</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>46</th>
      <td>troop_mean_completion_ratio</td>
      <td>0.249099</td>
      <td>5.075526e-01</td>
      <td>2.584535e-01</td>
      <td>-1.385810e-01</td>
      <td>8.952329e-01</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>44</th>
      <td>troop_count</td>
      <td>43.000000</td>
      <td>7.500000e+01</td>
      <td>3.200000e+01</td>
      <td>-5.000000e+00</td>
      <td>1.230000e+02</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>45</th>
      <td>troop_mean_level</td>
      <td>3.285714</td>
      <td>6.307692e+00</td>
      <td>3.021978e+00</td>
      <td>-1.247253e+00</td>
      <td>1.084066e+01</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>49</th>
      <td>hero_mean_completion_ratio</td>
      <td>0.210224</td>
      <td>6.678553e-01</td>
      <td>4.576315e-01</td>
      <td>-4.762234e-01</td>
      <td>1.354302e+00</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>48</th>
      <td>hero_mean_level</td>
      <td>20.000000</td>
      <td>4.887500e+01</td>
      <td>2.887500e+01</td>
      <td>-2.331250e+01</td>
      <td>9.218750e+01</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>47</th>
      <td>hero_count</td>
      <td>4.000000</td>
      <td>8.000000e+00</td>
      <td>4.000000e+00</td>
      <td>-2.000000e+00</td>
      <td>1.400000e+01</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>52</th>
      <td>spell_mean_completion_ratio</td>
      <td>0.373504</td>
      <td>6.586420e-01</td>
      <td>2.851377e-01</td>
      <td>-5.420228e-02</td>
      <td>1.086349e+00</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>53</th>
      <td>equipment_count</td>
      <td>10.000000</td>
      <td>3.200000e+01</td>
      <td>2.200000e+01</td>
      <td>-2.300000e+01</td>
      <td>6.500000e+01</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>54</th>
      <td>equipment_mean_level</td>
      <td>3.812500</td>
      <td>1.217647e+01</td>
      <td>8.363971e+00</td>
      <td>-8.733456e+00</td>
      <td>2.472243e+01</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>55</th>
      <td>equipment_mean_completion_ratio</td>
      <td>0.203704</td>
      <td>5.919753e-01</td>
      <td>3.882716e-01</td>
      <td>-3.787037e-01</td>
      <td>1.174383e+00</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>56</th>
      <td>achievement_count</td>
      <td>53.000000</td>
      <td>5.300000e+01</td>
      <td>0.000000e+00</td>
      <td>5.300000e+01</td>
      <td>5.300000e+01</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>


    
    Top 10 extreme values of donations:
    


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
      <th>role</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3161641.0</td>
      <td>member</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2400715.0</td>
      <td>coLeader</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2368337.0</td>
      <td>leader</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1266761.0</td>
      <td>coLeader</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1139948.0</td>
      <td>member</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1075534.0</td>
      <td>coLeader</td>
    </tr>
    <tr>
      <th>6</th>
      <td>833804.0</td>
      <td>leader</td>
    </tr>
    <tr>
      <th>7</th>
      <td>757437.0</td>
      <td>member</td>
    </tr>
    <tr>
      <th>8</th>
      <td>573620.0</td>
      <td>coLeader</td>
    </tr>
    <tr>
      <th>9</th>
      <td>535634.0</td>
      <td>coLeader</td>
    </tr>
  </tbody>
</table>
</div>


    
    Top 10 extreme values of trophies:
    


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
      <th>role</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5353.0</td>
      <td>coLeader</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5322.0</td>
      <td>admin</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5315.0</td>
      <td>coLeader</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5297.0</td>
      <td>coLeader</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5297.0</td>
      <td>coLeader</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5293.0</td>
      <td>coLeader</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5287.0</td>
      <td>admin</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5283.0</td>
      <td>coLeader</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5282.0</td>
      <td>admin</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5280.0</td>
      <td>admin</td>
    </tr>
  </tbody>
</table>
</div>


    
    Top 10 extreme values of attack_wins:
    


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
      <th>role</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>913.0</td>
      <td>coLeader</td>
    </tr>
    <tr>
      <th>1</th>
      <td>910.0</td>
      <td>coLeader</td>
    </tr>
    <tr>
      <th>2</th>
      <td>910.0</td>
      <td>admin</td>
    </tr>
    <tr>
      <th>3</th>
      <td>908.0</td>
      <td>coLeader</td>
    </tr>
    <tr>
      <th>4</th>
      <td>907.0</td>
      <td>admin</td>
    </tr>
    <tr>
      <th>5</th>
      <td>905.0</td>
      <td>member</td>
    </tr>
    <tr>
      <th>6</th>
      <td>905.0</td>
      <td>coLeader</td>
    </tr>
    <tr>
      <th>7</th>
      <td>902.0</td>
      <td>member</td>
    </tr>
    <tr>
      <th>8</th>
      <td>902.0</td>
      <td>coLeader</td>
    </tr>
    <tr>
      <th>9</th>
      <td>902.0</td>
      <td>member</td>
    </tr>
  </tbody>
</table>
</div>


## 8. Modeling Implications (Conclusions for the Modeling Phase)

### Structured summary of findings and recommendations


```python
# === Automatic executive summary ===
summary = {
    'n_rows': df.shape[0],
    'n_cols': df.shape[1],
    'missing_total': int(missing[missing > 0].sum()),
    'constant_features': [c[0] for c in constant_features],
    'target_distribution': target_counts.to_dict(),
    'target_imbalance_ratio': float(max_pct / max(min_pct, 1e-9)),
    'top_mi_features': mi_df.head(10).to_dict('records'),
    'pearson_high_corr_pairs': pearson_pairs.to_dict('records') if len(pearson_pairs) > 0 else [],
    'spearman_high_corr_pairs': spearman_pairs.to_dict('records') if len(spearman_pairs) > 0 else [],
    'outliers_max_pct': float(outlier_df['pct_outliers'].max()) if len(outlier_df) > 0 else 0.0,
    'outliers_most_common': outlier_df.head(1).to_dict('records') if len(outlier_df) > 0 else []
}

print("=== EXECUTIVE SUMMARY EDA P1 ===")
for k, v in summary.items():
    print(f"{k}: {v}")
```

    === EXECUTIVE SUMMARY EDA P1 ===
    n_rows: 837411
    n_cols: 66
    missing_total: 0
    constant_features: ['capital_contributions_diff_from_clan_mean', 'capital_contributions_diff_from_clan_mean', 'capital_contributions_ratio_to_clan_mean', 'capital_contributions_ratio_to_clan_mean', 'attack_wins_ratio_to_clan_mean', 'defense_wins_ratio_to_clan_mean', 'attack_wins', 'defense_wins', 'combat_activity_total', 'achievement_count', 'achievement_count']
    target_distribution: {'member': 346329, 'admin': 236271, 'coLeader': 202902, 'leader': 51909}
    target_imbalance_ratio: 6.671848812344679
    top_mi_features: [{'feature': 'donations_diff_from_clan_mean', 'mutual_info': 0.1439169472014976}, {'feature': 'donations_received_diff_from_clan_mean', 'mutual_info': 0.13077239221819248}, {'feature': 'town_hall_level_ratio_to_clan_mean', 'mutual_info': 0.11498305556848232}, {'feature': 'trophies_diff_from_clan_mean', 'mutual_info': 0.10547974285196071}, {'feature': 'war_stars_diff_from_clan_mean', 'mutual_info': 0.09023114219920414}, {'feature': 'war_stars_ratio_to_clan_mean', 'mutual_info': 0.08918693557631308}, {'feature': 'exp_level_ratio_to_clan_mean', 'mutual_info': 0.08572585957065382}, {'feature': 'exp_level_diff_from_clan_mean', 'mutual_info': 0.08547222902448626}, {'feature': 'war_stars_clan_pct', 'mutual_info': 0.08542196117284107}, {'feature': 'town_hall_level_diff_from_clan_mean', 'mutual_info': 0.0830633295450034}]
    pearson_high_corr_pairs: [{'feature_1': 'exp_level_diff_from_clan_mean', 'feature_2': 'exp_level_ratio_to_clan_mean', 'corr': 0.9001490118134395}, {'feature_1': 'exp_level_ratio_to_clan_mean', 'feature_2': 'town_hall_level_ratio_to_clan_mean', 'corr': 0.8599173124148108}, {'feature_1': 'town_hall_level_diff_from_clan_mean', 'feature_2': 'town_hall_level_ratio_to_clan_mean', 'corr': 0.964016038346303}, {'feature_1': 'exp_level_diff_from_clan_mean', 'feature_2': 'exp_level_clan_pct', 'corr': 0.8948246538108309}, {'feature_1': 'clan_points', 'feature_2': 'clan_capital_points', 'corr': 0.8917409795571107}, {'feature_1': 'town_hall_level', 'feature_2': 'exp_level', 'corr': 0.8916838476534488}, {'feature_1': 'exp_level', 'feature_2': 'best_trophies', 'corr': 0.9341833775287842}, {'feature_1': 'attack_wins_diff_from_clan_mean', 'feature_2': 'attack_wins', 'corr': 0.9218215508045353}, {'feature_1': 'defense_wins_diff_from_clan_mean', 'feature_2': 'defense_wins', 'corr': 0.9546949465312266}, {'feature_1': 'exp_level', 'feature_2': 'builder_hall_level', 'corr': 0.8516966764951563}, {'feature_1': 'exp_level', 'feature_2': 'builder_base_trophies', 'corr': 0.8736213709655044}, {'feature_1': 'builder_hall_level', 'feature_2': 'builder_base_trophies', 'corr': 0.8929433676010627}, {'feature_1': 'exp_level', 'feature_2': 'best_builder_base_trophies', 'corr': 0.8957228217273948}, {'feature_1': 'builder_hall_level', 'feature_2': 'best_builder_base_trophies', 'corr': 0.8841580151320565}, {'feature_1': 'builder_base_trophies', 'feature_2': 'best_builder_base_trophies', 'corr': 0.9740852728815064}, {'feature_1': 'donations_diff_from_clan_mean', 'feature_2': 'donations', 'corr': 0.9801551980455804}, {'feature_1': 'donations_diff_from_clan_mean', 'feature_2': 'donation_balance', 'corr': 0.9816305304088451}, {'feature_1': 'donations', 'feature_2': 'donation_balance', 'corr': 0.9636461509650384}, {'feature_1': 'attack_wins_diff_from_clan_mean', 'feature_2': 'combat_activity_total', 'corr': 0.917247114144462}, {'feature_1': 'attack_wins', 'feature_2': 'combat_activity_total', 'corr': 0.9951420260016343}, {'feature_1': 'town_hall_level', 'feature_2': 'troop_count', 'corr': 0.9508703860167353}, {'feature_1': 'exp_level', 'feature_2': 'troop_count', 'corr': 0.9414520025577721}, {'feature_1': 'best_trophies', 'feature_2': 'troop_count', 'corr': 0.8793673524618906}, {'feature_1': 'builder_hall_level', 'feature_2': 'troop_count', 'corr': 0.887620831494113}, {'feature_1': 'builder_base_trophies', 'feature_2': 'troop_count', 'corr': 0.8538611372417254}, {'feature_1': 'best_builder_base_trophies', 'feature_2': 'troop_count', 'corr': 0.860502768492204}, {'feature_1': 'exp_level', 'feature_2': 'troop_mean_level', 'corr': 0.9440262624290604}, {'feature_1': 'best_trophies', 'feature_2': 'troop_mean_level', 'corr': 0.8807941679454331}, {'feature_1': 'builder_base_trophies', 'feature_2': 'troop_mean_level', 'corr': 0.8913563748741234}, {'feature_1': 'best_builder_base_trophies', 'feature_2': 'troop_mean_level', 'corr': 0.9110821671089121}, {'feature_1': 'troop_count', 'feature_2': 'troop_mean_level', 'corr': 0.8977635078315117}, {'feature_1': 'town_hall_level', 'feature_2': 'troop_mean_completion_ratio', 'corr': 0.8516216284931183}, {'feature_1': 'exp_level', 'feature_2': 'troop_mean_completion_ratio', 'corr': 0.9460566676955994}, {'feature_1': 'best_trophies', 'feature_2': 'troop_mean_completion_ratio', 'corr': 0.8903269163341349}, {'feature_1': 'builder_base_trophies', 'feature_2': 'troop_mean_completion_ratio', 'corr': 0.8557379666976954}, {'feature_1': 'best_builder_base_trophies', 'feature_2': 'troop_mean_completion_ratio', 'corr': 0.8808023399020682}, {'feature_1': 'troop_count', 'feature_2': 'troop_mean_completion_ratio', 'corr': 0.9139077366450583}, {'feature_1': 'troop_mean_level', 'feature_2': 'troop_mean_completion_ratio', 'corr': 0.9876677248278652}, {'feature_1': 'town_hall_level', 'feature_2': 'hero_count', 'corr': 0.9454622088061055}, {'feature_1': 'exp_level', 'feature_2': 'hero_count', 'corr': 0.8631302079148179}, {'feature_1': 'builder_hall_level', 'feature_2': 'hero_count', 'corr': 0.8951349501496545}, {'feature_1': 'troop_count', 'feature_2': 'hero_count', 'corr': 0.9274227163895379}, {'feature_1': 'town_hall_level', 'feature_2': 'hero_mean_level', 'corr': 0.8554304937968595}, {'feature_1': 'exp_level', 'feature_2': 'hero_mean_level', 'corr': 0.9324228794433526}, {'feature_1': 'best_trophies', 'feature_2': 'hero_mean_level', 'corr': 0.8788593064491675}, {'feature_1': 'troop_count', 'feature_2': 'hero_mean_level', 'corr': 0.9043299480243826}, {'feature_1': 'troop_mean_level', 'feature_2': 'hero_mean_level', 'corr': 0.9237604830571727}, {'feature_1': 'troop_mean_completion_ratio', 'feature_2': 'hero_mean_level', 'corr': 0.93377407898453}, {'feature_1': 'town_hall_level', 'feature_2': 'hero_mean_completion_ratio', 'corr': 0.8663346838918746}, {'feature_1': 'exp_level', 'feature_2': 'hero_mean_completion_ratio', 'corr': 0.9436360177326656}, {'feature_1': 'best_trophies', 'feature_2': 'hero_mean_completion_ratio', 'corr': 0.8910921662949787}, {'feature_1': 'builder_base_trophies', 'feature_2': 'hero_mean_completion_ratio', 'corr': 0.8508791324058064}, {'feature_1': 'best_builder_base_trophies', 'feature_2': 'hero_mean_completion_ratio', 'corr': 0.8701925077391266}, {'feature_1': 'troop_count', 'feature_2': 'hero_mean_completion_ratio', 'corr': 0.9298378905139261}, {'feature_1': 'troop_mean_level', 'feature_2': 'hero_mean_completion_ratio', 'corr': 0.9371788819251587}, {'feature_1': 'troop_mean_completion_ratio', 'feature_2': 'hero_mean_completion_ratio', 'corr': 0.952624830802343}, {'feature_1': 'hero_mean_level', 'feature_2': 'hero_mean_completion_ratio', 'corr': 0.9843447314259481}, {'feature_1': 'town_hall_level', 'feature_2': 'spell_count', 'corr': 0.9267785442436567}, {'feature_1': 'exp_level', 'feature_2': 'spell_count', 'corr': 0.8817538218862572}, {'feature_1': 'builder_hall_level', 'feature_2': 'spell_count', 'corr': 0.8500199650924027}, {'feature_1': 'troop_count', 'feature_2': 'spell_count', 'corr': 0.9257360333363442}, {'feature_1': 'hero_count', 'feature_2': 'spell_count', 'corr': 0.9364741903829369}, {'feature_1': 'hero_mean_level', 'feature_2': 'spell_count', 'corr': 0.8819059409324196}, {'feature_1': 'hero_mean_completion_ratio', 'feature_2': 'spell_count', 'corr': 0.8623803968592061}, {'feature_1': 'troop_mean_level', 'feature_2': 'spell_mean_level', 'corr': 0.854621652112353}, {'feature_1': 'troop_mean_completion_ratio', 'feature_2': 'spell_mean_level', 'corr': 0.854519484515336}, {'feature_1': 'exp_level', 'feature_2': 'spell_mean_completion_ratio', 'corr': 0.8700304830876484}, {'feature_1': 'troop_mean_level', 'feature_2': 'spell_mean_completion_ratio', 'corr': 0.9047402881861749}, {'feature_1': 'troop_mean_completion_ratio', 'feature_2': 'spell_mean_completion_ratio', 'corr': 0.911934433746959}, {'feature_1': 'hero_mean_level', 'feature_2': 'spell_mean_completion_ratio', 'corr': 0.8883165257362778}, {'feature_1': 'hero_mean_completion_ratio', 'feature_2': 'spell_mean_completion_ratio', 'corr': 0.8701682283361974}, {'feature_1': 'spell_mean_level', 'feature_2': 'spell_mean_completion_ratio', 'corr': 0.9835972270017781}, {'feature_1': 'town_hall_level', 'feature_2': 'equipment_count', 'corr': 0.9107148797009506}, {'feature_1': 'exp_level', 'feature_2': 'equipment_count', 'corr': 0.8543900401047002}, {'feature_1': 'troop_count', 'feature_2': 'equipment_count', 'corr': 0.9275694657285501}, {'feature_1': 'troop_mean_completion_ratio', 'feature_2': 'equipment_count', 'corr': 0.8707682969901166}, {'feature_1': 'hero_count', 'feature_2': 'equipment_count', 'corr': 0.914031142084032}, {'feature_1': 'hero_mean_level', 'feature_2': 'equipment_count', 'corr': 0.8714720502303855}, {'feature_1': 'hero_mean_completion_ratio', 'feature_2': 'equipment_count', 'corr': 0.8890528322301133}, {'feature_1': 'spell_count', 'feature_2': 'equipment_count', 'corr': 0.8842424347422378}, {'feature_1': 'town_hall_level', 'feature_2': 'equipment_mean_level', 'corr': 0.8559871556404147}, {'feature_1': 'exp_level', 'feature_2': 'equipment_mean_level', 'corr': 0.8606223290173156}, {'feature_1': 'troop_count', 'feature_2': 'equipment_mean_level', 'corr': 0.8926413853583949}, {'feature_1': 'troop_mean_level', 'feature_2': 'equipment_mean_level', 'corr': 0.87359607140898}, {'feature_1': 'troop_mean_completion_ratio', 'feature_2': 'equipment_mean_level', 'corr': 0.9026603708430297}, {'feature_1': 'hero_mean_level', 'feature_2': 'equipment_mean_level', 'corr': 0.8708853318226154}, {'feature_1': 'hero_mean_completion_ratio', 'feature_2': 'equipment_mean_level', 'corr': 0.8954005337683707}, {'feature_1': 'equipment_count', 'feature_2': 'equipment_mean_level', 'corr': 0.9164109758920717}, {'feature_1': 'town_hall_level', 'feature_2': 'equipment_mean_completion_ratio', 'corr': 0.8529062714205827}, {'feature_1': 'exp_level', 'feature_2': 'equipment_mean_completion_ratio', 'corr': 0.8561643434521129}, {'feature_1': 'troop_count', 'feature_2': 'equipment_mean_completion_ratio', 'corr': 0.8855816967114907}, {'feature_1': 'troop_mean_level', 'feature_2': 'equipment_mean_completion_ratio', 'corr': 0.8683944398832283}, {'feature_1': 'troop_mean_completion_ratio', 'feature_2': 'equipment_mean_completion_ratio', 'corr': 0.8926947900571742}, {'feature_1': 'hero_mean_level', 'feature_2': 'equipment_mean_completion_ratio', 'corr': 0.8624801032187406}, {'feature_1': 'hero_mean_completion_ratio', 'feature_2': 'equipment_mean_completion_ratio', 'corr': 0.8839663793529899}, {'feature_1': 'equipment_count', 'feature_2': 'equipment_mean_completion_ratio', 'corr': 0.901593937279917}, {'feature_1': 'equipment_mean_level', 'feature_2': 'equipment_mean_completion_ratio', 'corr': 0.9970225564599938}, {'feature_1': 'troop_mean_completion_ratio', 'feature_2': 'achievement_completion_ratio', 'corr': 0.8683639278465286}]
    spearman_high_corr_pairs: [{'feature_1': 'exp_level_diff_from_clan_mean', 'feature_2': 'exp_level_ratio_to_clan_mean', 'corr': 0.971851180228231}, {'feature_1': 'town_hall_level_diff_from_clan_mean', 'feature_2': 'town_hall_level_ratio_to_clan_mean', 'corr': 0.993523081145209}, {'feature_1': 'exp_level_diff_from_clan_mean', 'feature_2': 'exp_level_clan_pct', 'corr': 0.9331341019518617}, {'feature_1': 'exp_level_ratio_to_clan_mean', 'feature_2': 'exp_level_clan_pct', 'corr': 0.9166457694613056}, {'feature_1': 'war_stars_diff_from_clan_mean', 'feature_2': 'war_stars_clan_pct', 'corr': 0.8594576241687177}, {'feature_1': 'war_stars_ratio_to_clan_mean', 'feature_2': 'war_stars_clan_pct', 'corr': 0.8882432327973688}, {'feature_1': 'clan_points', 'feature_2': 'clan_capital_points', 'corr': 0.9070394521976096}, {'feature_1': 'town_hall_level', 'feature_2': 'exp_level', 'corr': 0.8972839561355849}, {'feature_1': 'exp_level', 'feature_2': 'best_trophies', 'corr': 0.9391245511582926}, {'feature_1': 'exp_level', 'feature_2': 'war_stars', 'corr': 0.8791672320743088}, {'feature_1': 'attack_wins_ratio_to_clan_mean', 'feature_2': 'attack_wins', 'corr': 0.9998723760833487}, {'feature_1': 'defense_wins_ratio_to_clan_mean', 'feature_2': 'defense_wins', 'corr': 0.9997220976784419}, {'feature_1': 'town_hall_level', 'feature_2': 'builder_hall_level', 'corr': 0.8505853353880947}, {'feature_1': 'exp_level', 'feature_2': 'builder_hall_level', 'corr': 0.8737475274500364}, {'feature_1': 'exp_level', 'feature_2': 'builder_base_trophies', 'corr': 0.8894952048628155}, {'feature_1': 'builder_hall_level', 'feature_2': 'builder_base_trophies', 'corr': 0.913198558476526}, {'feature_1': 'exp_level', 'feature_2': 'best_builder_base_trophies', 'corr': 0.9105762670737384}, {'feature_1': 'best_trophies', 'feature_2': 'best_builder_base_trophies', 'corr': 0.8594162628073104}, {'feature_1': 'builder_hall_level', 'feature_2': 'best_builder_base_trophies', 'corr': 0.9249776094074618}, {'feature_1': 'builder_base_trophies', 'feature_2': 'best_builder_base_trophies', 'corr': 0.9850791250454726}, {'feature_1': 'donations_ratio_to_clan_mean', 'feature_2': 'donations', 'corr': 0.9890722786820347}, {'feature_1': 'donations_received_ratio_to_clan_mean', 'feature_2': 'donations_received', 'corr': 0.9839102179125062}, {'feature_1': 'defense_wins_ratio_to_clan_mean', 'feature_2': 'combat_activity_total', 'corr': 0.9414153254355659}, {'feature_1': 'defense_wins', 'feature_2': 'combat_activity_total', 'corr': 0.941626691475901}, {'feature_1': 'trophies', 'feature_2': 'progression_ratio_trophies', 'corr': 0.9865359433177927}, {'feature_1': 'town_hall_level', 'feature_2': 'troop_count', 'corr': 0.9586231701982441}, {'feature_1': 'exp_level', 'feature_2': 'troop_count', 'corr': 0.9470805053159714}, {'feature_1': 'best_trophies', 'feature_2': 'troop_count', 'corr': 0.8856315142428698}, {'feature_1': 'builder_hall_level', 'feature_2': 'troop_count', 'corr': 0.9031862569417787}, {'feature_1': 'builder_base_trophies', 'feature_2': 'troop_count', 'corr': 0.8692136323241492}, {'feature_1': 'best_builder_base_trophies', 'feature_2': 'troop_count', 'corr': 0.8847899482287029}, {'feature_1': 'exp_level', 'feature_2': 'troop_mean_level', 'corr': 0.9451216218483022}, {'feature_1': 'best_trophies', 'feature_2': 'troop_mean_level', 'corr': 0.8829425110972489}, {'feature_1': 'builder_hall_level', 'feature_2': 'troop_mean_level', 'corr': 0.8574204594036301}, {'feature_1': 'builder_base_trophies', 'feature_2': 'troop_mean_level', 'corr': 0.9065022712459555}, {'feature_1': 'best_builder_base_trophies', 'feature_2': 'troop_mean_level', 'corr': 0.9218678435470662}, {'feature_1': 'troop_count', 'feature_2': 'troop_mean_level', 'corr': 0.9082413181460244}, {'feature_1': 'town_hall_level', 'feature_2': 'troop_mean_completion_ratio', 'corr': 0.8795264549216455}, {'feature_1': 'exp_level', 'feature_2': 'troop_mean_completion_ratio', 'corr': 0.9578386803203193}, {'feature_1': 'best_trophies', 'feature_2': 'troop_mean_completion_ratio', 'corr': 0.8977433269428355}, {'feature_1': 'builder_hall_level', 'feature_2': 'troop_mean_completion_ratio', 'corr': 0.8563743969512647}, {'feature_1': 'builder_base_trophies', 'feature_2': 'troop_mean_completion_ratio', 'corr': 0.8822873880295851}, {'feature_1': 'best_builder_base_trophies', 'feature_2': 'troop_mean_completion_ratio', 'corr': 0.8984425316259248}, {'feature_1': 'troop_count', 'feature_2': 'troop_mean_completion_ratio', 'corr': 0.9399556735955354}, {'feature_1': 'troop_mean_level', 'feature_2': 'troop_mean_completion_ratio', 'corr': 0.9893697298505719}, {'feature_1': 'town_hall_level', 'feature_2': 'hero_count', 'corr': 0.9429797880426316}, {'feature_1': 'exp_level', 'feature_2': 'hero_count', 'corr': 0.8856424267946135}, {'feature_1': 'builder_hall_level', 'feature_2': 'hero_count', 'corr': 0.899104695187841}, {'feature_1': 'troop_count', 'feature_2': 'hero_count', 'corr': 0.9482678428331847}, {'feature_1': 'troop_mean_level', 'feature_2': 'hero_count', 'corr': 0.8518225152707274}, {'feature_1': 'troop_mean_completion_ratio', 'feature_2': 'hero_count', 'corr': 0.8840125006927603}, {'feature_1': 'exp_level', 'feature_2': 'hero_mean_level', 'corr': 0.9306187024108057}, {'feature_1': 'best_trophies', 'feature_2': 'hero_mean_level', 'corr': 0.8791677331110601}, {'feature_1': 'troop_count', 'feature_2': 'hero_mean_level', 'corr': 0.9078254332666449}, {'feature_1': 'troop_mean_level', 'feature_2': 'hero_mean_level', 'corr': 0.9173088898649976}, {'feature_1': 'troop_mean_completion_ratio', 'feature_2': 'hero_mean_level', 'corr': 0.9391753463380493}, {'feature_1': 'town_hall_level', 'feature_2': 'hero_mean_completion_ratio', 'corr': 0.886141246900681}, {'feature_1': 'exp_level', 'feature_2': 'hero_mean_completion_ratio', 'corr': 0.9496174322125518}, {'feature_1': 'best_trophies', 'feature_2': 'hero_mean_completion_ratio', 'corr': 0.8926525058196169}, {'feature_1': 'builder_hall_level', 'feature_2': 'hero_mean_completion_ratio', 'corr': 0.862646603930529}, {'feature_1': 'builder_base_trophies', 'feature_2': 'hero_mean_completion_ratio', 'corr': 0.8715574305617396}, {'feature_1': 'best_builder_base_trophies', 'feature_2': 'hero_mean_completion_ratio', 'corr': 0.8879783221273047}, {'feature_1': 'troop_count', 'feature_2': 'hero_mean_completion_ratio', 'corr': 0.9425904765936148}, {'feature_1': 'troop_mean_level', 'feature_2': 'hero_mean_completion_ratio', 'corr': 0.93534408397509}, {'feature_1': 'troop_mean_completion_ratio', 'feature_2': 'hero_mean_completion_ratio', 'corr': 0.9544816623671756}, {'feature_1': 'hero_count', 'feature_2': 'hero_mean_completion_ratio', 'corr': 0.8832812807090521}, {'feature_1': 'hero_mean_level', 'feature_2': 'hero_mean_completion_ratio', 'corr': 0.9891785928560608}, {'feature_1': 'town_hall_level', 'feature_2': 'spell_count', 'corr': 0.9240723678932197}, {'feature_1': 'exp_level', 'feature_2': 'spell_count', 'corr': 0.9164754025816017}, {'feature_1': 'best_trophies', 'feature_2': 'spell_count', 'corr': 0.8529371360503715}, {'feature_1': 'troop_count', 'feature_2': 'spell_count', 'corr': 0.9672887415674352}, {'feature_1': 'troop_mean_level', 'feature_2': 'spell_count', 'corr': 0.8828189833619289}, {'feature_1': 'troop_mean_completion_ratio', 'feature_2': 'spell_count', 'corr': 0.923041195337966}, {'feature_1': 'hero_count', 'feature_2': 'spell_count', 'corr': 0.9293998082757967}, {'feature_1': 'hero_mean_level', 'feature_2': 'spell_count', 'corr': 0.9032017891182189}, {'feature_1': 'hero_mean_completion_ratio', 'feature_2': 'spell_count', 'corr': 0.928855470904161}, {'feature_1': 'troop_mean_level', 'feature_2': 'spell_mean_level', 'corr': 0.8575727518316636}, {'feature_1': 'troop_mean_completion_ratio', 'feature_2': 'spell_mean_level', 'corr': 0.8723763625698564}, {'feature_1': 'exp_level', 'feature_2': 'spell_mean_completion_ratio', 'corr': 0.8688259450086104}, {'feature_1': 'troop_mean_level', 'feature_2': 'spell_mean_completion_ratio', 'corr': 0.9013279313512609}, {'feature_1': 'troop_mean_completion_ratio', 'feature_2': 'spell_mean_completion_ratio', 'corr': 0.9222535662249286}, {'feature_1': 'hero_mean_level', 'feature_2': 'spell_mean_completion_ratio', 'corr': 0.8876234695155565}, {'feature_1': 'hero_mean_completion_ratio', 'feature_2': 'spell_mean_completion_ratio', 'corr': 0.8816970238459448}, {'feature_1': 'spell_mean_level', 'feature_2': 'spell_mean_completion_ratio', 'corr': 0.9838860386827117}, {'feature_1': 'town_hall_level', 'feature_2': 'equipment_count', 'corr': 0.9262641824675679}, {'feature_1': 'exp_level', 'feature_2': 'equipment_count', 'corr': 0.8596906527564733}, {'feature_1': 'clan_capital_contributions', 'feature_2': 'equipment_count', 'corr': 0.8591395794111405}, {'feature_1': 'troop_count', 'feature_2': 'equipment_count', 'corr': 0.9352310986984033}, {'feature_1': 'troop_mean_completion_ratio', 'feature_2': 'equipment_count', 'corr': 0.8880118368124763}, {'feature_1': 'hero_count', 'feature_2': 'equipment_count', 'corr': 0.935470190311612}, {'feature_1': 'hero_mean_level', 'feature_2': 'equipment_count', 'corr': 0.873289921553544}, {'feature_1': 'hero_mean_completion_ratio', 'feature_2': 'equipment_count', 'corr': 0.8988720555322406}, {'feature_1': 'spell_count', 'feature_2': 'equipment_count', 'corr': 0.9367340657058053}, {'feature_1': 'town_hall_level', 'feature_2': 'equipment_mean_level', 'corr': 0.8840027574728276}, {'feature_1': 'exp_level', 'feature_2': 'equipment_mean_level', 'corr': 0.8549941168879648}, {'feature_1': 'troop_count', 'feature_2': 'equipment_mean_level', 'corr': 0.9074032743820426}, {'feature_1': 'troop_mean_level', 'feature_2': 'equipment_mean_level', 'corr': 0.8578198726755685}, {'feature_1': 'troop_mean_completion_ratio', 'feature_2': 'equipment_mean_level', 'corr': 0.8906921253208341}, {'feature_1': 'hero_count', 'feature_2': 'equipment_mean_level', 'corr': 0.8860222645167982}, {'feature_1': 'hero_mean_level', 'feature_2': 'equipment_mean_level', 'corr': 0.864744182064843}, {'feature_1': 'hero_mean_completion_ratio', 'feature_2': 'equipment_mean_level', 'corr': 0.8869666684543199}, {'feature_1': 'spell_count', 'feature_2': 'equipment_mean_level', 'corr': 0.8974397904781009}, {'feature_1': 'equipment_count', 'feature_2': 'equipment_mean_level', 'corr': 0.9301429409459304}, {'feature_1': 'town_hall_level', 'feature_2': 'equipment_mean_completion_ratio', 'corr': 0.8739674612816193}, {'feature_1': 'exp_level', 'feature_2': 'equipment_mean_completion_ratio', 'corr': 0.8507047251174301}, {'feature_1': 'troop_count', 'feature_2': 'equipment_mean_completion_ratio', 'corr': 0.8977473002688984}, {'feature_1': 'troop_mean_level', 'feature_2': 'equipment_mean_completion_ratio', 'corr': 0.8542463724803991}, {'feature_1': 'troop_mean_completion_ratio', 'feature_2': 'equipment_mean_completion_ratio', 'corr': 0.8842495054509139}, {'feature_1': 'hero_count', 'feature_2': 'equipment_mean_completion_ratio', 'corr': 0.8741360576451189}, {'feature_1': 'hero_mean_level', 'feature_2': 'equipment_mean_completion_ratio', 'corr': 0.8548639363188749}, {'feature_1': 'hero_mean_completion_ratio', 'feature_2': 'equipment_mean_completion_ratio', 'corr': 0.8771105185176332}, {'feature_1': 'spell_count', 'feature_2': 'equipment_mean_completion_ratio', 'corr': 0.8852348517866596}, {'feature_1': 'equipment_count', 'feature_2': 'equipment_mean_completion_ratio', 'corr': 0.909238254640528}, {'feature_1': 'equipment_mean_level', 'feature_2': 'equipment_mean_completion_ratio', 'corr': 0.9972131424551625}, {'feature_1': 'town_hall_level', 'feature_2': 'achievement_completion_ratio', 'corr': 0.9204915666802086}, {'feature_1': 'exp_level', 'feature_2': 'achievement_completion_ratio', 'corr': 0.9692850582883229}, {'feature_1': 'best_trophies', 'feature_2': 'achievement_completion_ratio', 'corr': 0.9248746359405307}, {'feature_1': 'war_stars', 'feature_2': 'achievement_completion_ratio', 'corr': 0.8532721041496405}, {'feature_1': 'builder_hall_level', 'feature_2': 'achievement_completion_ratio', 'corr': 0.873988948466854}, {'feature_1': 'builder_base_trophies', 'feature_2': 'achievement_completion_ratio', 'corr': 0.8718676899814491}, {'feature_1': 'best_builder_base_trophies', 'feature_2': 'achievement_completion_ratio', 'corr': 0.892497970783502}, {'feature_1': 'troop_count', 'feature_2': 'achievement_completion_ratio', 'corr': 0.9575161455188681}, {'feature_1': 'troop_mean_level', 'feature_2': 'achievement_completion_ratio', 'corr': 0.9320923907325278}, {'feature_1': 'troop_mean_completion_ratio', 'feature_2': 'achievement_completion_ratio', 'corr': 0.9538101539045183}, {'feature_1': 'hero_count', 'feature_2': 'achievement_completion_ratio', 'corr': 0.9075282897487531}, {'feature_1': 'hero_mean_level', 'feature_2': 'achievement_completion_ratio', 'corr': 0.9229140635639714}, {'feature_1': 'hero_mean_completion_ratio', 'feature_2': 'achievement_completion_ratio', 'corr': 0.9458862947836342}, {'feature_1': 'spell_count', 'feature_2': 'achievement_completion_ratio', 'corr': 0.9303156188823306}, {'feature_1': 'spell_mean_completion_ratio', 'feature_2': 'achievement_completion_ratio', 'corr': 0.8646678679769749}, {'feature_1': 'equipment_count', 'feature_2': 'achievement_completion_ratio', 'corr': 0.8949124517737489}, {'feature_1': 'equipment_mean_level', 'feature_2': 'achievement_completion_ratio', 'corr': 0.8952411301933015}, {'feature_1': 'equipment_mean_completion_ratio', 'feature_2': 'achievement_completion_ratio', 'corr': 0.8891446028829528}]
    outliers_max_pct: 34.084099683429045
    outliers_most_common: [{'feature': 'donation_balance', 'q1': 0.0, 'q3': 0.0, 'IQR': 0.0, 'lower_bound': 0.0, 'upper_bound': 0.0, 'n_outliers': 285424, 'pct_outliers': 34.084099683429045}]
    

### Preprocessing Pipeline

- **Imputation**  
  - Numeric variables: use the median if skewness is high or the percentage of outliers is large; otherwise, use the mean.  
  - Categorical variables: use the mode.  
  - If more than 40–50% of values in a column are missing, consider dropping it or creating an explicit `"missing"` category.

- **Transformations**  
  - Apply `log1p` to features with severe skewness (`|skew| > 1`) and presence of very high values, such as `donations`, `trophies`, or loot-related metrics.  
  - Use `RobustScaler` for variables with extreme outliers; `StandardScaler` for the rest.  
  - Encode the `role` target with `LabelEncoder` (or `OrdinalEncoder` if preserving semantic order is preferred).

### Validation Strategy

- Use **`StratifiedKFold`** with 5 or 10 folds to preserve class distribution in each partition.  
- For highly imbalanced datasets, avoid relying on `accuracy`. Prioritize:  
  - `F1-macro`  
  - `Cohen's Kappa`  
  - `PR-AUC` (for multiclass problems, use `average='macro'` or One-vs-Rest).  
  - Row-normalized confusion matrix.

### Feature Pruning

- Remove **zero variance** or **nearly constant** features (top frequency ≥ 95%).  
- Remove one feature from each pair with `|Pearson| > 0.85` or `|Spearman| > 0.85`, keeping the one with higher mutual information or interpretability.  
- Review outlier concentration: if `pct_outliers` is very high (>10–15%), consider treating that variable with a robust transformation instead of removing records.

### Recommended Baseline

1. **Multiclass Logistic Regression**  
   - `LogisticRegression(multi_class='multinomial', solver='saga', class_weight='balanced', max_iter=1000)`  
   - A good choice as a fast and interpretable baseline.  

2. **Random Forest**  
   - Robust to nonlinearities and outliers; use `class_weight='balanced_subsample'`.  
   - Evaluate feature importance as a complement to MI analysis.

3. **Gradient Boosting / XGBoost / LightGBM**  
   - More powerful models if the dataset is large; use `scale_pos_weight` or `class_weight` for imbalance and hyperparameter search with nested cross-validation.

These conclusions should directly feed the feature engineering and modeling phase of Problem 1.

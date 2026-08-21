# Problem 4: Clan Performance Classification EDA
This notebook performs Exploratory Data Analysis (EDA) for the Clan Performance Classification problem.
It focuses on structural, compositional, infrastructure, and player progression features while enforcing strict leakage prevention.


## 1. Setup & Dynamic Environment



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.preprocessing import LabelEncoder

sns.set_theme(style='whitegrid', palette='viridis')
plt.rcParams['figure.dpi'] = 100

from pathlib import Path

def find_project_root(marker='data'):
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent
    return current

PROJECT_ROOT = find_project_root()
DATA_PATH = PROJECT_ROOT / 'data' / 'datasets' / 'clan_performance_classification.parquet'

df = pd.read_parquet(DATA_PATH)

print('Dataset loaded from:', DATA_PATH)
print('Shape:', df.shape)
print()
print('Columns:')
print(df.columns.tolist())
print()
print('Data types:')
print(df.dtypes)

```

    Dataset loaded from: c:\Users\Usuario\Desktop\Clash of Clans ML Lab\data\datasets\clan_performance_classification.parquet
    Shape: (31289, 51)
    
    Columns:
    ['clan_tag', 'clan_level', 'clan_points', 'clan_capital_points', 'members', 'required_trophies', 'war_frequency', 'war_league', 'capital_league', 'type', 'is_family_friendly', 'location_id', 'location_name', 'mean_town_hall_level', 'median_town_hall_level', 'std_town_hall_level', 'mean_exp_level', 'median_exp_level', 'std_exp_level', 'mean_trophies', 'median_trophies', 'std_trophies', 'mean_donations', 'median_donations', 'std_donations', 'mean_donations_received', 'median_donations_received', 'std_donations_received', 'mean_clan_capital_contributions', 'median_clan_capital_contributions', 'std_clan_capital_contributions', 'mean_troop_mean_level', 'mean_troop_mean_completion_ratio', 'mean_hero_mean_level', 'mean_hero_mean_completion_ratio', 'mean_spell_mean_level', 'mean_spell_mean_completion_ratio', 'mean_equipment_mean_level', 'mean_equipment_mean_completion_ratio', 'mean_achievement_completion_ratio', 'th18_percentage', 'th17_plus_percentage', 'member_count', 'sum_donations', 'sum_donations_received', 'sum_clan_capital_contributions', 'donation_balance', 'donation_ratio', 'donation_rate', 'capital_contribution_rate', 'performance_class']
    
    Data types:
    clan_tag                                    str
    clan_level                                int64
    clan_points                               int64
    clan_capital_points                       int64
    members                                   int64
    required_trophies                         int64
    war_frequency                               str
    war_league                                  str
    capital_league                              str
    type                                        str
    is_family_friendly                         bool
    location_id                             float64
    location_name                               str
    mean_town_hall_level                    float64
    median_town_hall_level                  float64
    std_town_hall_level                     float64
    mean_exp_level                          float64
    median_exp_level                        float64
    std_exp_level                           float64
    mean_trophies                           float64
    median_trophies                         float64
    std_trophies                            float64
    mean_donations                          float64
    median_donations                        float64
    std_donations                           float64
    mean_donations_received                 float64
    median_donations_received               float64
    std_donations_received                  float64
    mean_clan_capital_contributions         float64
    median_clan_capital_contributions       float64
    std_clan_capital_contributions          float64
    mean_troop_mean_level                   float64
    mean_troop_mean_completion_ratio        float64
    mean_hero_mean_level                    float64
    mean_hero_mean_completion_ratio         float64
    mean_spell_mean_level                   float64
    mean_spell_mean_completion_ratio        float64
    mean_equipment_mean_level               float64
    mean_equipment_mean_completion_ratio    float64
    mean_achievement_completion_ratio       float64
    th18_percentage                         float64
    th17_plus_percentage                    float64
    member_count                              int64
    sum_donations                           float64
    sum_donations_received                  float64
    sum_clan_capital_contributions          float64
    donation_balance                        float64
    donation_ratio                          float64
    donation_rate                           float64
    capital_contribution_rate               float64
    performance_class                           str
    dtype: object
    

## 2. Data Integrity & Leakage Prevention Audit



```python
print('Dimensions:', df.shape)
print('Duplicate rows:', df.duplicated().sum())

missing = df.isna().sum()
missing_pct = (missing / len(df)) * 100
print()
print('Missing values (top 10):')
print(missing_pct[missing_pct > 0].sort_values(ascending=False).head(10))

TARGET = 'performance_class'
CONTINUOUS_TARGET = 'war_success_rate'

LEAKAGE_COLS = ['war_wins', 'war_losses', 'war_ties', 'war_win_streak']

IDENTIFIER_COLS = [c for c in df.columns if c.lower() in {'tag', 'clan_tag', 'name', 'player_tag', 'clan_name'}]

drop_cols = [c for c in df.columns if (c in LEAKAGE_COLS) or (c in IDENTIFIER_COLS) or c == TARGET or c == CONTINUOUS_TARGET]
X = df.drop(columns=drop_cols).copy()
y = df[TARGET].astype(str)

print()
print('Dropped predictor columns (leakage / identifiers / target):')
print([c for c in df.columns if c not in X.columns])

zero_variance = X.columns[X.nunique(dropna=False) <= 1].tolist()
numeric_X = X.select_dtypes(include=[np.number])
quasi_constant = numeric_X.columns[numeric_X.var(skipna=True) < 1e-6].tolist()
print()
print('Zero-variance features:', zero_variance)
print('Quasi-constant features (variance < 1e-6):', quasi_constant)

X = X.drop(columns=list(set(zero_variance + quasi_constant)))
print()
print('X shape after data integrity filtering:', X.shape)

```

    Dimensions: (31289, 51)
    Duplicate rows: 0
    
    Missing values (top 10):
    Series([], dtype: float64)
    
    Dropped predictor columns (leakage / identifiers / target):
    ['clan_tag', 'performance_class']
    
    Zero-variance features: []
    Quasi-constant features (variance < 1e-6): []
    
    X shape after data integrity filtering: (31289, 49)
    

## 3. Target Analysis & Threshold Justification (war_success_rate & performance_class)



```python
if CONTINUOUS_TARGET in df.columns:
    ws = pd.to_numeric(df[CONTINUOUS_TARGET], errors='coerce')
    print('war_success_rate summary:')
    print(ws.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_string())
    print()
    print('Skewness:', round(stats.skew(ws.dropna()), 4))
    terciles = ws.quantile([1/3, 2/3]).values
    print('Tercile cutoffs (33.3%, 66.7%):', [round(t, 4) for t in terciles])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(ws.dropna(), kde=True, ax=axes[0], color='teal')
    axes[0].set_title('Distribution of war_success_rate')
    axes[0].axvline(terciles[0], color='red', linestyle='--', label='33% cutoff')
    axes[0].axvline(terciles[1], color='red', linestyle='--', label='66% cutoff')
    axes[0].legend()

    class_counts = y.value_counts().sort_index()
    axes[1].bar(class_counts.index, class_counts.values, color=sns.color_palette('viridis', len(class_counts)))
    axes[1].set_title('Performance Class Frequency')
    axes[1].set_ylabel('Count')
    plt.tight_layout()
    plt.show()

    total = len(y)
    balance = (class_counts / total) * 100
    minority = balance.min()
    majority = balance.max()
    print()
    print('Class distribution (%):')
    print(balance.round(2))
    print()
    print(f'Minority class proportion: {minority:.2f}%')
    print(f'Minority-to-majority ratio: {minority / majority:.3f}')
else:
    print(f'{CONTINUOUS_TARGET} not found in dataset.')

```

    war_success_rate not found in dataset.
    

## 4. Feature Group Profiling



```python
feature_groups = {
    'Town Hall & Composition': [
        'mean_TH', 'median_TH', 'std_TH', 'TH18_percentage',
    ],
    'Player Progression': [
        'mean_EXP', 'mean_hero_level', 'mean_troop_progress',
        'mean_spell_progress', 'mean_equipment_progress',
    ],
    'Infrastructure & Activity': [
        'clan_level', 'member_count', 'donation_rate',
        'capital_contribution_rate', 'clan_capital_points',
    ],
}

rows = []
for group_name, col_candidates in feature_groups.items():
    cols_present = [c for c in col_candidates if c in X.columns]
    if not cols_present:
        continue
    group_data = X[cols_present]
    missing_pct = group_data.isna().mean().mean() * 100
    zero_pct = (group_data == 0).sum().sum() / (group_data.shape[0] * len(cols_present)) * 100
    numeric_group = group_data.select_dtypes(include=[np.number])
    avg_abs_skew = numeric_group.skew().abs().mean() if len(numeric_group.columns) > 0 else np.nan
    rows.append({
        'group': group_name,
        'n_features': len(cols_present),
        'missing_pct': round(missing_pct, 2),
        'zero_pct': round(zero_pct, 2),
        'avg_abs_skew': round(avg_abs_skew, 4) if not np.isnan(avg_abs_skew) else np.nan,
    })

profile_df = pd.DataFrame(rows)
print('Feature group profile:')
print(profile_df.to_string(index=False))

```

    Feature group profile:
                        group  n_features  missing_pct  zero_pct  avg_abs_skew
    Infrastructure & Activity           5          0.0     20.48       12.8127
    

## 5. Feature-Target Relationships (Classification)



```python
X_processed = X.copy()

numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
X_processed[numeric_cols] = X_processed[numeric_cols].fillna(X_processed[numeric_cols].median())

object_cols = X_processed.select_dtypes(include=['object']).columns
for c in object_cols:
    X_processed[c] = X_processed[c].fillna('missing').astype('category').cat.codes

le = LabelEncoder()
y_encoded = le.fit_transform(y)

mi_scores = mutual_info_classif(X_processed, y_encoded, random_state=42)
mi_series = pd.Series(mi_scores, index=X_processed.columns).sort_values(ascending=False)

numeric_imputed = X_processed.select_dtypes(include=[np.number])
f_scores, _ = f_classif(numeric_imputed, y_encoded)
f_series = pd.Series(f_scores, index=numeric_imputed.columns).sort_values(ascending=False)

print('Top 10 Mutual Information Scores:')
print(mi_series.head(10).round(4).to_string())

print()
print('Top 10 ANOVA F-Scores:')
print(f_series.head(10).round(4).to_string())

top_mi_features = mi_series.head(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, feat in enumerate(top_mi_features):
    if feat in numeric_imputed.columns:
        for cls in le.classes_:
            idx = np.where(le.classes_ == cls)[0][0]
            subset = numeric_imputed.loc[y_encoded == idx, feat]
            axes[i].hist(subset, alpha=0.5, label=cls, color=sns.color_palette('viridis', len(le.classes_))[idx], density=True)
        axes[i].set_title(f'Distribution of {feat} by class')
        axes[i].legend()
    else:
        axes[i].set_visible(False)
plt.tight_layout()
plt.show()

```

    C:\Users\Usuario\AppData\Local\Temp\ipykernel_3416\1554551760.py:6: Pandas4Warning: For backward compatibility, 'str' dtypes are included by select_dtypes when 'object' dtype is specified. This behavior is deprecated and will be removed in a future version. Explicitly pass 'str' to `include` to select them, or to `exclude` to remove them and silence this warning.
    See https://pandas.pydata.org/docs/user_guide/migration-3-strings.html#string-migration-select-dtypes for details on how to write code that works with pandas 2 and 3.
      object_cols = X_processed.select_dtypes(include=['object']).columns
    

    Top 10 Mutual Information Scores:
    war_league                              0.0692
    clan_level                              0.0442
    mean_spell_mean_level                   0.0380
    mean_spell_mean_completion_ratio        0.0335
    location_name                           0.0320
    mean_equipment_mean_completion_ratio    0.0282
    mean_exp_level                          0.0281
    location_id                             0.0278
    sum_donations_received                  0.0263
    mean_hero_mean_level                    0.0263
    
    Top 10 ANOVA F-Scores:
    war_league                              1408.4535
    mean_spell_mean_completion_ratio        1042.4681
    mean_spell_mean_level                   1040.6639
    clan_level                               997.8021
    mean_hero_mean_level                     812.8529
    mean_troop_mean_completion_ratio         676.9242
    mean_hero_mean_completion_ratio          675.0405
    mean_exp_level                           661.7533
    mean_equipment_mean_level                638.4292
    mean_equipment_mean_completion_ratio     617.7370
    


    
![png](eda_p4_files/eda_p4_10_2.png)
    


## 6. Collinearity & Feature Redundancy



```python
numeric_corr = numeric_imputed.corr(method='pearson')

upper_tri = numeric_corr.where(np.triu(np.ones(numeric_corr.shape), k=1).astype(bool))
high_pairs = [
    (col, row, upper_tri.loc[row, col])
    for col in upper_tri.columns
    for row in upper_tri.index
    if abs(upper_tri.loc[row, col]) > 0.85 and not pd.isna(upper_tri.loc[row, col])
]
high_pairs = sorted(high_pairs, key=lambda x: abs(x[2]), reverse=True)

print('Feature pairs with |r| > 0.85:')
if high_pairs:
    for col, row, val in high_pairs:
        print(f'{row:30s} - {col:30s}: {val:.4f}')
else:
    print('None found.')

plt.figure(figsize=(12, 10))
sns.heatmap(numeric_corr, cmap='viridis', center=0, vmin=-1, vmax=1, square=True, linewidths=0.5)
plt.title('Pearson Correlation Heatmap of Numerical Features')
plt.tight_layout()
plt.show()

```

    Feature pairs with |r| > 0.85:
    members                        - member_count                  : 1.0000
    mean_donations                 - donation_rate                 : 1.0000
    mean_clan_capital_contributions - capital_contribution_rate     : 0.9999
    mean_equipment_mean_level      - mean_equipment_mean_completion_ratio: 0.9984
    clan_capital_points            - capital_league                : 0.9975
    sum_donations                  - sum_donations_received        : 0.9975
    mean_donations                 - mean_donations_received       : 0.9958
    mean_donations_received        - donation_rate                 : 0.9958
    mean_troop_mean_level          - mean_troop_mean_completion_ratio: 0.9906
    mean_hero_mean_level           - mean_hero_mean_completion_ratio: 0.9897
    mean_spell_mean_level          - mean_spell_mean_completion_ratio: 0.9884
    mean_exp_level                 - median_exp_level              : 0.9818
    mean_troop_mean_completion_ratio - mean_hero_mean_completion_ratio: 0.9814
    std_donations                  - sum_donations                 : 0.9795
    mean_town_hall_level           - median_town_hall_level        : 0.9790
    std_donations                  - sum_donations_received        : 0.9761
    mean_exp_level                 - mean_troop_mean_level         : 0.9738
    mean_troop_mean_level          - mean_hero_mean_completion_ratio: 0.9700
    mean_exp_level                 - mean_troop_mean_completion_ratio: 0.9696
    mean_troop_mean_completion_ratio - mean_hero_mean_level          : 0.9688
    mean_clan_capital_contributions - median_clan_capital_contributions: 0.9681
    mean_exp_level                 - mean_hero_mean_completion_ratio: 0.9680
    median_clan_capital_contributions - capital_contribution_rate     : 0.9679
    mean_exp_level                 - mean_hero_mean_level          : 0.9663
    mean_troop_mean_level          - mean_hero_mean_level          : 0.9642
    median_exp_level               - mean_troop_mean_level         : 0.9540
    median_exp_level               - mean_troop_mean_completion_ratio: 0.9509
    median_exp_level               - mean_hero_mean_completion_ratio: 0.9497
    mean_troop_mean_completion_ratio - mean_spell_mean_completion_ratio: 0.9491
    mean_hero_mean_level           - mean_spell_mean_completion_ratio: 0.9470
    mean_troop_mean_level          - mean_spell_mean_completion_ratio: 0.9466
    median_exp_level               - mean_hero_mean_level          : 0.9461
    mean_donations_received        - median_donations_received     : 0.9448
    sum_donations                  - donation_rate                 : 0.9386
    mean_donations_received        - sum_donations_received        : 0.9385
    mean_donations                 - sum_donations                 : 0.9384
    mean_donations                 - median_donations_received     : 0.9380
    median_donations_received      - donation_rate                 : 0.9380
    mean_town_hall_level           - mean_exp_level                : 0.9364
    mean_donations_received        - sum_donations                 : 0.9360
    mean_troop_mean_completion_ratio - mean_equipment_mean_level     : 0.9357
    sum_donations_received         - donation_rate                 : 0.9356
    mean_donations                 - sum_donations_received        : 0.9353
    mean_troop_mean_completion_ratio - mean_equipment_mean_completion_ratio: 0.9321
    mean_hero_mean_completion_ratio - mean_equipment_mean_level     : 0.9318
    mean_hero_mean_completion_ratio - mean_spell_mean_completion_ratio: 0.9310
    mean_town_hall_level           - mean_hero_mean_completion_ratio: 0.9303
    mean_town_hall_level           - mean_hero_mean_level          : 0.9301
    mean_exp_level                 - mean_spell_mean_completion_ratio: 0.9289
    mean_hero_mean_completion_ratio - mean_equipment_mean_completion_ratio: 0.9281
    std_donations                  - donation_rate                 : 0.9277
    mean_donations                 - std_donations                 : 0.9275
    mean_troop_mean_completion_ratio - mean_achievement_completion_ratio: 0.9270
    mean_town_hall_level           - mean_troop_mean_completion_ratio: 0.9256
    std_donations                  - mean_donations_received       : 0.9243
    th18_percentage                - th17_plus_percentage          : 0.9239
    mean_town_hall_level           - mean_troop_mean_level         : 0.9204
    median_town_hall_level         - median_exp_level              : 0.9183
    median_town_hall_level         - mean_exp_level                : 0.9182
    clan_points                    - sum_clan_capital_contributions: 0.9160
    mean_town_hall_level           - median_exp_level              : 0.9145
    mean_hero_mean_level           - mean_equipment_mean_level     : 0.9136
    mean_hero_mean_level           - mean_equipment_mean_completion_ratio: 0.9135
    median_town_hall_level         - mean_hero_mean_completion_ratio: 0.9128
    mean_town_hall_level           - mean_equipment_mean_completion_ratio: 0.9126
    mean_town_hall_level           - mean_spell_mean_completion_ratio: 0.9123
    mean_hero_mean_completion_ratio - mean_achievement_completion_ratio: 0.9114
    median_town_hall_level         - mean_hero_mean_level          : 0.9109
    mean_town_hall_level           - mean_equipment_mean_level     : 0.9102
    mean_troop_mean_level          - mean_spell_mean_level         : 0.9092
    median_town_hall_level         - mean_troop_mean_completion_ratio: 0.9081
    mean_troop_mean_level          - mean_equipment_mean_completion_ratio: 0.9075
    mean_troop_mean_level          - mean_equipment_mean_level     : 0.9069
    std_donations                  - median_donations_received     : 0.9054
    median_exp_level               - mean_spell_mean_completion_ratio: 0.9053
    mean_troop_mean_completion_ratio - mean_spell_mean_level         : 0.9051
    mean_hero_mean_level           - mean_spell_mean_level         : 0.9038
    median_town_hall_level         - mean_troop_mean_level         : 0.9004
    clan_points                    - clan_capital_points           : 0.9002
    median_donations_received      - sum_donations_received        : 0.8948
    median_town_hall_level         - mean_equipment_mean_completion_ratio: 0.8935
    clan_points                    - capital_league                : 0.8932
    mean_trophies                  - median_trophies               : 0.8928
    mean_spell_mean_completion_ratio - mean_equipment_mean_completion_ratio: 0.8925
    median_town_hall_level         - mean_equipment_mean_level     : 0.8921
    mean_spell_mean_completion_ratio - mean_equipment_mean_level     : 0.8921
    mean_troop_mean_level          - mean_achievement_completion_ratio: 0.8916
    median_town_hall_level         - mean_spell_mean_completion_ratio: 0.8915
    median_donations_received      - sum_donations                 : 0.8896
    mean_achievement_completion_ratio - capital_contribution_rate     : 0.8893
    mean_clan_capital_contributions - mean_achievement_completion_ratio: 0.8893
    mean_exp_level                 - mean_achievement_completion_ratio: 0.8885
    mean_exp_level                 - mean_spell_mean_level         : 0.8881
    mean_exp_level                 - mean_equipment_mean_completion_ratio: 0.8828
    mean_exp_level                 - mean_equipment_mean_level     : 0.8815
    mean_hero_mean_completion_ratio - mean_spell_mean_level         : 0.8810
    clan_capital_points            - sum_clan_capital_contributions: 0.8775
    mean_achievement_completion_ratio - th17_plus_percentage          : 0.8774
    mean_equipment_mean_level      - mean_achievement_completion_ratio: 0.8744
    mean_clan_capital_contributions - std_clan_capital_contributions: 0.8742
    std_clan_capital_contributions - capital_contribution_rate     : 0.8742
    mean_hero_mean_level           - mean_achievement_completion_ratio: 0.8739
    median_exp_level               - mean_achievement_completion_ratio: 0.8737
    capital_league                 - sum_clan_capital_contributions: 0.8717
    mean_town_hall_level           - mean_spell_mean_level         : 0.8714
    median_exp_level               - mean_spell_mean_level         : 0.8636
    mean_achievement_completion_ratio - th18_percentage               : 0.8621
    median_exp_level               - mean_equipment_mean_completion_ratio: 0.8617
    mean_equipment_mean_completion_ratio - mean_achievement_completion_ratio: 0.8614
    median_exp_level               - mean_equipment_mean_level     : 0.8610
    sum_clan_capital_contributions - capital_contribution_rate     : 0.8604
    mean_clan_capital_contributions - sum_clan_capital_contributions: 0.8602
    mean_clan_capital_contributions - mean_troop_mean_completion_ratio: 0.8521
    mean_troop_mean_completion_ratio - capital_contribution_rate     : 0.8521
    median_clan_capital_contributions - mean_achievement_completion_ratio: 0.8521
    


    
![png](eda_p4_files/eda_p4_12_1.png)
    


## 7. Modeling Strategy & Executive Summary



```python
if 'balance' not in locals():
    class_counts = y.value_counts().sort_index()
    balance = (class_counts / len(y)) * 100

summary = {
    'problem': 'Clan performance classification into low, medium, high tiers',
    'dataset': str(DATA_PATH.relative_to(PROJECT_ROOT)) if DATA_PATH.is_relative_to(PROJECT_ROOT) else str(DATA_PATH),
    'target': TARGET,
    'continuous_target_used_for_binning': CONTINUOUS_TARGET,
    'predictors_after_cleaning': X.shape[1],
    'leakage_columns_removed': [c for c in LEAKAGE_COLS if c in df.columns],
    'identifier_columns_removed': [c for c in IDENTIFIER_COLS if c in df.columns],
    'zero_or_quasi_constant_removed': list(set(zero_variance + quasi_constant)),
    'class_balance_ratios': balance.round(2).to_dict(),
    'top_mi_features': mi_series.head(10).round(4).to_dict(),
    'top_anova_features': f_series.head(10).round(4).to_dict(),
    'high_collinearity_pairs': [(str(a), str(b), float(v)) for a, b, v in high_pairs],
    'preprocessing_pipeline': [
        'Impute missing values (median for numeric, constant missing for categorical)',
        'Encode categorical features using ordinal/integer encoding or one-hot for nominal',
        'Scale numeric features with StandardScaler or RobustScaler to handle outliers',
        'Ensure leakage columns are excluded before any model training',
    ],
    'cv_scheme': 'StratifiedKFold(n_splits=5, shuffle=True, random_state=42)',
    'baseline_models': [
        'Logistic Regression (multinomial)',
        'Decision Tree Classifier',
        'Random Forest Classifier',
        'XGBoost Classifier',
        'MLP Classifier',
    ],
    'evaluation_metrics': [
        'Macro F1-score',
        'Weighted F1-score',
        'Accuracy',
        'Multi-class Log Loss',
        'Confusion Matrix analysis',
    ],
}

import json
print('Executive Summary:')
print(json.dumps(summary, indent=2, default=str))

```

    Executive Summary:
    {
      "problem": "Clan performance classification into low, medium, high tiers",
      "dataset": "data\\datasets\\clan_performance_classification.parquet",
      "target": "performance_class",
      "continuous_target_used_for_binning": "war_success_rate",
      "predictors_after_cleaning": 49,
      "leakage_columns_removed": [],
      "identifier_columns_removed": [
        "clan_tag"
      ],
      "zero_or_quasi_constant_removed": [],
      "class_balance_ratios": {
        "high": 33.25,
        "low": 33.32,
        "medium": 33.42
      },
      "top_mi_features": {
        "war_league": 0.0692,
        "clan_level": 0.0442,
        "mean_spell_mean_level": 0.038,
        "mean_spell_mean_completion_ratio": 0.0335,
        "location_name": 0.032,
        "mean_equipment_mean_completion_ratio": 0.0282,
        "mean_exp_level": 0.0281,
        "location_id": 0.0278,
        "sum_donations_received": 0.0263,
        "mean_hero_mean_level": 0.0263
      },
      "top_anova_features": {
        "war_league": 1408.4535,
        "mean_spell_mean_completion_ratio": 1042.4681,
        "mean_spell_mean_level": 1040.6639,
        "clan_level": 997.8021,
        "mean_hero_mean_level": 812.8529,
        "mean_troop_mean_completion_ratio": 676.9242,
        "mean_hero_mean_completion_ratio": 675.0405,
        "mean_exp_level": 661.7533,
        "mean_equipment_mean_level": 638.4292,
        "mean_equipment_mean_completion_ratio": 617.737
      },
      "high_collinearity_pairs": [
        [
          "member_count",
          "members",
          0.9999987196100075
        ],
        [
          "donation_rate",
          "mean_donations",
          0.9999539629910303
        ],
        [
          "capital_contribution_rate",
          "mean_clan_capital_contributions",
          0.9998772934271833
        ],
        [
          "mean_equipment_mean_completion_ratio",
          "mean_equipment_mean_level",
          0.9983611126077823
        ],
        [
          "capital_league",
          "clan_capital_points",
          0.9974670358612939
        ],
        [
          "sum_donations_received",
          "sum_donations",
          0.9974514779448072
        ],
        [
          "mean_donations_received",
          "mean_donations",
          0.9958437716412711
        ],
        [
          "donation_rate",
          "mean_donations_received",
          0.9957944923897667
        ],
        [
          "mean_troop_mean_completion_ratio",
          "mean_troop_mean_level",
          0.9905537890359362
        ],
        [
          "mean_hero_mean_completion_ratio",
          "mean_hero_mean_level",
          0.9896939182480741
        ],
        [
          "mean_spell_mean_completion_ratio",
          "mean_spell_mean_level",
          0.9884171185601821
        ],
        [
          "median_exp_level",
          "mean_exp_level",
          0.9818088524209279
        ],
        [
          "mean_hero_mean_completion_ratio",
          "mean_troop_mean_completion_ratio",
          0.9814048232460592
        ],
        [
          "sum_donations",
          "std_donations",
          0.9795121185735912
        ],
        [
          "median_town_hall_level",
          "mean_town_hall_level",
          0.9789699880549386
        ],
        [
          "sum_donations_received",
          "std_donations",
          0.976068191999473
        ],
        [
          "mean_troop_mean_level",
          "mean_exp_level",
          0.9737722212621678
        ],
        [
          "mean_hero_mean_completion_ratio",
          "mean_troop_mean_level",
          0.9699564211498651
        ],
        [
          "mean_troop_mean_completion_ratio",
          "mean_exp_level",
          0.969589254784319
        ],
        [
          "mean_hero_mean_level",
          "mean_troop_mean_completion_ratio",
          0.9688261058558773
        ],
        [
          "median_clan_capital_contributions",
          "mean_clan_capital_contributions",
          0.9680824726898613
        ],
        [
          "mean_hero_mean_completion_ratio",
          "mean_exp_level",
          0.9679698413749701
        ],
        [
          "capital_contribution_rate",
          "median_clan_capital_contributions",
          0.9679283372262766
        ],
        [
          "mean_hero_mean_level",
          "mean_exp_level",
          0.9662780793716778
        ],
        [
          "mean_hero_mean_level",
          "mean_troop_mean_level",
          0.964177503903047
        ],
        [
          "mean_troop_mean_level",
          "median_exp_level",
          0.9540141099256774
        ],
        [
          "mean_troop_mean_completion_ratio",
          "median_exp_level",
          0.9509137176287084
        ],
        [
          "mean_hero_mean_completion_ratio",
          "median_exp_level",
          0.949699442137694
        ],
        [
          "mean_spell_mean_completion_ratio",
          "mean_troop_mean_completion_ratio",
          0.9491310999731407
        ],
        [
          "mean_spell_mean_completion_ratio",
          "mean_hero_mean_level",
          0.9469878582773932
        ],
        [
          "mean_spell_mean_completion_ratio",
          "mean_troop_mean_level",
          0.9466419466389812
        ],
        [
          "mean_hero_mean_level",
          "median_exp_level",
          0.9461238970033116
        ],
        [
          "median_donations_received",
          "mean_donations_received",
          0.9447817151500219
        ],
        [
          "donation_rate",
          "sum_donations",
          0.938644628316269
        ],
        [
          "sum_donations_received",
          "mean_donations_received",
          0.9385063330051074
        ],
        [
          "sum_donations",
          "mean_donations",
          0.9383811898689541
        ],
        [
          "median_donations_received",
          "mean_donations",
          0.938036052428505
        ],
        [
          "donation_rate",
          "median_donations_received",
          0.9380046323100206
        ],
        [
          "mean_exp_level",
          "mean_town_hall_level",
          0.9363873743650943
        ],
        [
          "sum_donations",
          "mean_donations_received",
          0.9359857270159119
        ],
        [
          "mean_equipment_mean_level",
          "mean_troop_mean_completion_ratio",
          0.935729068562321
        ],
        [
          "donation_rate",
          "sum_donations_received",
          0.9355707891722868
        ],
        [
          "sum_donations_received",
          "mean_donations",
          0.935307685433069
        ],
        [
          "mean_equipment_mean_completion_ratio",
          "mean_troop_mean_completion_ratio",
          0.9321427918943657
        ],
        [
          "mean_equipment_mean_level",
          "mean_hero_mean_completion_ratio",
          0.9318333686870326
        ],
        [
          "mean_spell_mean_completion_ratio",
          "mean_hero_mean_completion_ratio",
          0.9309515220276339
        ],
        [
          "mean_hero_mean_completion_ratio",
          "mean_town_hall_level",
          0.9302804932791575
        ],
        [
          "mean_hero_mean_level",
          "mean_town_hall_level",
          0.930130045259148
        ],
        [
          "mean_spell_mean_completion_ratio",
          "mean_exp_level",
          0.92887960774193
        ],
        [
          "mean_equipment_mean_completion_ratio",
          "mean_hero_mean_completion_ratio",
          0.9281110035344491
        ],
        [
          "donation_rate",
          "std_donations",
          0.9276631824479901
        ],
        [
          "std_donations",
          "mean_donations",
          0.9274617514451567
        ],
        [
          "mean_achievement_completion_ratio",
          "mean_troop_mean_completion_ratio",
          0.9269896716180719
        ],
        [
          "mean_troop_mean_completion_ratio",
          "mean_town_hall_level",
          0.9256212761378594
        ],
        [
          "mean_donations_received",
          "std_donations",
          0.9242894387383332
        ],
        [
          "th17_plus_percentage",
          "th18_percentage",
          0.923942208405904
        ],
        [
          "mean_troop_mean_level",
          "mean_town_hall_level",
          0.9203630331244214
        ],
        [
          "median_exp_level",
          "median_town_hall_level",
          0.918282748103769
        ],
        [
          "mean_exp_level",
          "median_town_hall_level",
          0.9181668602878041
        ],
        [
          "sum_clan_capital_contributions",
          "clan_points",
          0.9159711953489312
        ],
        [
          "median_exp_level",
          "mean_town_hall_level",
          0.9145431499302263
        ],
        [
          "mean_equipment_mean_level",
          "mean_hero_mean_level",
          0.913610843617331
        ],
        [
          "mean_equipment_mean_completion_ratio",
          "mean_hero_mean_level",
          0.9134554953142464
        ],
        [
          "mean_hero_mean_completion_ratio",
          "median_town_hall_level",
          0.9128083516611185
        ],
        [
          "mean_equipment_mean_completion_ratio",
          "mean_town_hall_level",
          0.9125918868425773
        ],
        [
          "mean_spell_mean_completion_ratio",
          "mean_town_hall_level",
          0.9123040958372808
        ],
        [
          "mean_achievement_completion_ratio",
          "mean_hero_mean_completion_ratio",
          0.9113778586964301
        ],
        [
          "mean_hero_mean_level",
          "median_town_hall_level",
          0.9108720399434137
        ],
        [
          "mean_equipment_mean_level",
          "mean_town_hall_level",
          0.9101760789113504
        ],
        [
          "mean_spell_mean_level",
          "mean_troop_mean_level",
          0.9092423536800155
        ],
        [
          "mean_troop_mean_completion_ratio",
          "median_town_hall_level",
          0.9080825798580815
        ],
        [
          "mean_equipment_mean_completion_ratio",
          "mean_troop_mean_level",
          0.9074924860211492
        ],
        [
          "mean_equipment_mean_level",
          "mean_troop_mean_level",
          0.9069065841377916
        ],
        [
          "median_donations_received",
          "std_donations",
          0.9053958850052797
        ],
        [
          "mean_spell_mean_completion_ratio",
          "median_exp_level",
          0.9053129689600433
        ],
        [
          "mean_spell_mean_level",
          "mean_troop_mean_completion_ratio",
          0.9051152499196883
        ],
        [
          "mean_spell_mean_level",
          "mean_hero_mean_level",
          0.9038182732211075
        ],
        [
          "mean_troop_mean_level",
          "median_town_hall_level",
          0.9003637890169182
        ],
        [
          "clan_capital_points",
          "clan_points",
          0.9001944610848239
        ],
        [
          "sum_donations_received",
          "median_donations_received",
          0.8947587591459918
        ],
        [
          "mean_equipment_mean_completion_ratio",
          "median_town_hall_level",
          0.8935224112510168
        ],
        [
          "capital_league",
          "clan_points",
          0.8932233990508327
        ],
        [
          "median_trophies",
          "mean_trophies",
          0.8927576633535159
        ],
        [
          "mean_equipment_mean_completion_ratio",
          "mean_spell_mean_completion_ratio",
          0.8925409979780571
        ],
        [
          "mean_equipment_mean_level",
          "median_town_hall_level",
          0.8921479739845861
        ],
        [
          "mean_equipment_mean_level",
          "mean_spell_mean_completion_ratio",
          0.8920712700733143
        ],
        [
          "mean_achievement_completion_ratio",
          "mean_troop_mean_level",
          0.8915738054708215
        ],
        [
          "mean_spell_mean_completion_ratio",
          "median_town_hall_level",
          0.891531260755109
        ],
        [
          "sum_donations",
          "median_donations_received",
          0.8895963278933791
        ],
        [
          "capital_contribution_rate",
          "mean_achievement_completion_ratio",
          0.889292596402319
        ],
        [
          "mean_achievement_completion_ratio",
          "mean_clan_capital_contributions",
          0.8892836421314928
        ],
        [
          "mean_achievement_completion_ratio",
          "mean_exp_level",
          0.8884896673143202
        ],
        [
          "mean_spell_mean_level",
          "mean_exp_level",
          0.8880526691614137
        ],
        [
          "mean_equipment_mean_completion_ratio",
          "mean_exp_level",
          0.8828498806730448
        ],
        [
          "mean_equipment_mean_level",
          "mean_exp_level",
          0.8814662994852405
        ],
        [
          "mean_spell_mean_level",
          "mean_hero_mean_completion_ratio",
          0.8809715597549873
        ],
        [
          "sum_clan_capital_contributions",
          "clan_capital_points",
          0.8774702409024014
        ],
        [
          "th17_plus_percentage",
          "mean_achievement_completion_ratio",
          0.877417020458128
        ],
        [
          "mean_achievement_completion_ratio",
          "mean_equipment_mean_level",
          0.8744497037354427
        ],
        [
          "std_clan_capital_contributions",
          "mean_clan_capital_contributions",
          0.8741832301085524
        ],
        [
          "capital_contribution_rate",
          "std_clan_capital_contributions",
          0.874158407813108
        ],
        [
          "mean_achievement_completion_ratio",
          "mean_hero_mean_level",
          0.873859303900518
        ],
        [
          "mean_achievement_completion_ratio",
          "median_exp_level",
          0.8737163268009026
        ],
        [
          "sum_clan_capital_contributions",
          "capital_league",
          0.8717286921300385
        ],
        [
          "mean_spell_mean_level",
          "mean_town_hall_level",
          0.8713835529109082
        ],
        [
          "mean_spell_mean_level",
          "median_exp_level",
          0.8636384309487055
        ],
        [
          "th18_percentage",
          "mean_achievement_completion_ratio",
          0.862144281614095
        ],
        [
          "mean_equipment_mean_completion_ratio",
          "median_exp_level",
          0.8616985644287639
        ],
        [
          "mean_achievement_completion_ratio",
          "mean_equipment_mean_completion_ratio",
          0.8613694558563652
        ],
        [
          "mean_equipment_mean_level",
          "median_exp_level",
          0.8609923883551597
        ],
        [
          "capital_contribution_rate",
          "sum_clan_capital_contributions",
          0.8603784962466942
        ],
        [
          "sum_clan_capital_contributions",
          "mean_clan_capital_contributions",
          0.8601567103366632
        ],
        [
          "mean_troop_mean_completion_ratio",
          "mean_clan_capital_contributions",
          0.8521332637961737
        ],
        [
          "capital_contribution_rate",
          "mean_troop_mean_completion_ratio",
          0.8521077130944711
        ],
        [
          "mean_achievement_completion_ratio",
          "median_clan_capital_contributions",
          0.8520835459262922
        ]
      ],
      "preprocessing_pipeline": [
        "Impute missing values (median for numeric, constant missing for categorical)",
        "Encode categorical features using ordinal/integer encoding or one-hot for nominal",
        "Scale numeric features with StandardScaler or RobustScaler to handle outliers",
        "Ensure leakage columns are excluded before any model training"
      ],
      "cv_scheme": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
      "baseline_models": [
        "Logistic Regression (multinomial)",
        "Decision Tree Classifier",
        "Random Forest Classifier",
        "XGBoost Classifier",
        "MLP Classifier"
      ],
      "evaluation_metrics": [
        "Macro F1-score",
        "Weighted F1-score",
        "Accuracy",
        "Multi-class Log Loss",
        "Confusion Matrix analysis"
      ]
    }
    

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

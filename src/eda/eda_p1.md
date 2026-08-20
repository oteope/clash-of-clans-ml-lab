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


```python
# General integrity information
print("=== General information ===")
df.info(show_counts=True)

print("\n=== Data types ===")
display(df.dtypes.to_frame(name='dtype'))
```

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

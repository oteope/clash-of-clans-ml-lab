# Problem 5: Player Clustering & Archetype Discovery EDA
This notebook performs unsupervised Exploratory Data Analysis to discover natural player archetypes.
It focuses on player progression, activity, and social/economic features.


## 1. Setup & Dynamic Environment



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA

sns.set_theme(style='whitegrid', palette='viridis')
plt.rcParams['figure.dpi'] = 110
plt.rcParams['figure.figsize'] = (10, 6)

from pathlib import Path

def find_project_root(marker='data'):
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent
    return current

PROJECT_ROOT = find_project_root()
DATA_PATH = PROJECT_ROOT / 'data' / 'datasets' / 'player_clustering.parquet'

df = pd.read_parquet(DATA_PATH)

print('Dataset loaded from:', DATA_PATH)
print('Dimensions:', df.shape)
print()
print('Data types:')
print(df.dtypes)
print()
print('First 5 rows:')
print(df.head())

```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    Cell In[1], line 25
         21 
         22 PROJECT_ROOT = find_project_root()
         23 DATA_PATH = PROJECT_ROOT / 'data' / 'datasets' / 'players_clustering.parquet'
         24 
    ---> 25 df = pd.read_parquet(DATA_PATH)
         26 
         27 print('Dataset loaded from:', DATA_PATH)
         28 print('Dimensions:', df.shape)
    

    File c:\Users\Usuario\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\pandas\io\parquet.py:671, in read_parquet(path, engine, columns, storage_options, dtype_backend, filesystem, filters, to_pandas_kwargs, **kwargs)
        668 impl = get_engine(engine)
        669 check_dtype_backend(dtype_backend)
    --> 671 return impl.read(
        672     path,
        673     columns=columns,
        674     filters=filters,
        675     storage_options=storage_options,
        676     dtype_backend=dtype_backend,
        677     filesystem=filesystem,
        678     to_pandas_kwargs=to_pandas_kwargs,
        679     **kwargs,
        680 )
    

    File c:\Users\Usuario\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\pandas\io\parquet.py:253, in PyArrowImpl.read(self, path, columns, filters, dtype_backend, storage_options, filesystem, to_pandas_kwargs, **kwargs)
        240 def read(
        241     self,
        242     path,
       (...)    249     **kwargs,
        250 ) -> DataFrame:
        251     kwargs["use_pandas_metadata"] = True
    --> 253     path_or_handle, handles, filesystem = _get_path_or_handle(
        254         path,
        255         filesystem,
        256         storage_options=storage_options,
        257         mode="rb",
        258     )
        259     try:
        260         pa_table = self.api.parquet.read_table(
        261             path_or_handle,
        262             columns=columns,
       (...)    265             **kwargs,
        266         )
    

    File c:\Users\Usuario\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\pandas\io\parquet.py:141, in _get_path_or_handle(path, fs, storage_options, mode, is_dir)
        131 handles = None
        132 if (
        133     not fs
        134     and not is_dir
       (...)    139     # fsspec resources can also point to directories
        140     # this branch is used for example when reading from non-fsspec URLs
    --> 141     handles = get_handle(
        142         path_or_handle, mode, is_text=False, storage_options=storage_options
        143     )
        144     fs = None
        145     path_or_handle = handles.handle
    

    File c:\Users\Usuario\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\pandas\io\common.py:939, in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
        930         handle = open(
        931             handle,
        932             ioargs.mode,
       (...)    935             newline="",
        936         )
        937     else:
        938         # Binary mode
    --> 939         handle = open(handle, ioargs.mode)
        940     handles.append(handle)
        942 # Convert BytesIO or file objects passed with an encoding
    

    FileNotFoundError: [Errno 2] No such file or directory: 'c:\\Users\\Usuario\\Desktop\\Clash of Clans ML Lab\\data\\datasets\\players_clustering.parquet'


## 2. Data Integrity & Identifier Audit



```python
print('Dimensions:', df.shape)
print('Duplicate rows:', df.duplicated().sum())

missing = df.isna().sum()
missing_pct = (missing / len(df)) * 100
print()
print('Missing values (top 10):')
print(missing_pct[missing_pct > 0].sort_values(ascending=False).head(10))

identifier_candidates = {'tag', 'player_tag', 'name', 'player_name', 'clan_tag'}
identifier_cols = [c for c in df.columns if c.lower() in identifier_candidates]

high_cardinality_cols = [
    c for c in df.columns
    if df[c].nunique(dropna=True) > 0.9 * len(df)
]

drop_cols = list(dict.fromkeys(identifier_cols + high_cardinality_cols))
X = df.drop(columns=drop_cols).copy()

print()
print('Identifier columns dropped:', identifier_cols)
print('High-cardinality columns dropped:', [c for c in high_cardinality_cols if c not in identifier_cols])
print('Feature matrix shape after dropping identifiers:', X.shape)

zero_variance = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
quasi_constant = []
for c in X.columns:
    counts = X[c].value_counts(dropna=False)
    if len(counts) > 0 and counts.iloc[0] / len(X) >= 0.95:
        quasi_constant.append(c)

print()
print('Zero-variance features:', zero_variance)
print('Quasi-constant features (mode frequency >= 95%):', quasi_constant)

drop_invariant = list(set(zero_variance + quasi_constant))
X = X.drop(columns=drop_invariant).copy()
print('X shape after invariant filtering:', X.shape)

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
print('Numeric features for clustering:', len(numeric_cols))

```

## 3. Feature Distribution, Skewness & Scale Analysis



```python
numeric_X = X[numeric_cols]

summary_rows = []
for col in numeric_cols:
    s = numeric_X[col]
    summary_rows.append({
        'feature': col,
        'mean': s.mean(),
        'std': s.std(),
        'median': s.median(),
        'Q1': s.quantile(0.25),
        'Q3': s.quantile(0.75),
        'IQR': s.quantile(0.75) - s.quantile(0.25),
        'min': s.min(),
        'max': s.max(),
        'zero_pct': (s == 0).mean() * 100,
        'skewness': stats.skew(s.dropna()),
        'kurtosis': stats.kurtosis(s.dropna()),
    })

summary_df = pd.DataFrame(summary_rows).set_index('feature')
print('Statistical summary of numeric features:')
print(summary_df.round(4).to_string())

skewed_features = summary_df.loc[summary_df['skewness'].abs() > 1.0].index.tolist()
print()
print('Heavily skewed features (|skewness| > 1.0), consider log1p transformation:')
print(skewed_features)

scale_stats = summary_df[['std', 'IQR']].describe().round(4)
print()
print('Scale variance across features:')
print(scale_stats)
print('Justification: Features vary widely in scale, so StandardScaler or RobustScaler should be applied before clustering.')

```

## 4. Conceptual Feature Group Profiling



```python
feature_groups = {
    'Progression & Levels': [
        'town_hall_level', 'builder_hall_level', 'exp_level',
        'hero_progress', 'troop_progress', 'equipment_progress'
    ],
    'Activity & Battle': [
        'trophies', 'best_trophies', 'war_stars', 'attack_wins', 'defense_wins'
    ],
    'Social & Economy': [
        'donations', 'donations_received', 'capital_contributions'
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

## 5. Multicollinearity & Redundancy Check



```python
numeric_for_corr = numeric_X.copy()
numeric_for_corr = numeric_for_corr.fillna(numeric_for_corr.median())

pearson_corr = numeric_for_corr.corr(method='pearson')
spearman_corr = numeric_for_corr.corr(method='spearman')

def get_high_pairs(corr_matrix, threshold=0.85):
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    pairs = []
    for col in upper.columns:
        for row in upper.index:
            val = upper.loc[row, col]
            if pd.notna(val) and abs(val) > threshold:
                pairs.append((row, col, val))
    return sorted(pairs, key=lambda x: abs(x[2]), reverse=True)

pearson_pairs = get_high_pairs(pearson_corr)
spearman_pairs = get_high_pairs(spearman_corr)

print('Pearson high-correlation pairs (|r| > 0.85):')
if pearson_pairs:
    for a, b, v in pearson_pairs:
        print(f'{a:30s} - {b:30s}: {v:.4f}')
else:
    print('None found.')

print()
print('Spearman high-correlation pairs (|r| > 0.85):')
if spearman_pairs:
    for a, b, v in spearman_pairs:
        print(f'{a:30s} - {b:30s}: {v:.4f}')
else:
    print('None found.')

plt.figure(figsize=(12, 10))
sns.heatmap(pearson_corr, cmap='viridis', center=0, vmin=-1, vmax=1, square=True, linewidths=0.5)
plt.title('Pearson Correlation Heatmap of Numeric Features')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 10))
sns.heatmap(spearman_corr, cmap='viridis', center=0, vmin=-1, vmax=1, square=True, linewidths=0.5)
plt.title('Spearman Correlation Heatmap of Numeric Features')
plt.tight_layout()
plt.show()

```

## 6. Dimensionality & Clusterability Preview



```python
X_for_pca = numeric_X.fillna(numeric_X.median())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_for_pca)

pca_full = PCA(n_components=min(len(numeric_cols), 20), random_state=42)
pca_full.fit(X_scaled)

cum_var = np.cumsum(pca_full.explained_variance_ratio_)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, len(cum_var)+1), cum_var, marker='o', linestyle='-', color='teal')
ax.axhline(y=0.90, color='red', linestyle='--', label='90% explained variance')
ax.set_xlabel('Number of Principal Components')
ax.set_ylabel('Cumulative Explained Variance Ratio')
ax.set_title('PCA Cumulative Explained Variance')
ax.legend()
plt.tight_layout()
plt.show()

pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], alpha=0.5, s=15, c='steelblue', edgecolors='none')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('2D PCA Scatter Plot - Visual Clusterability Preview')
plt.tight_layout()
plt.show()

print(f'Explained variance by first 2 PCs: {pca_2d.explained_variance_ratio_.sum():.4f}')
print('Cumulative variance for each component:')
for i, v in enumerate(cum_var, 1):
    print(f'PC{i}: {v:.4f}')

```

## 7. Outlier & Extreme Value Inspection



```python
outlier_counts = {}
for col in numeric_cols:
    q1 = numeric_X[col].quantile(0.25)
    q3 = numeric_X[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (numeric_X[col] < lower) | (numeric_X[col] > upper)
    outlier_counts[col] = mask.sum()

outlier_series = pd.Series(outlier_counts).sort_values(ascending=False)
print('Univariate IQR outlier counts per feature (top 10):')
print(outlier_series.head(10).to_string())
print()

X_robust = numeric_X.fillna(numeric_X.median())
robust_scaler = RobustScaler()
X_robust_scaled = robust_scaler.fit_transform(X_robust)

centroid = np.mean(X_robust_scaled, axis=0)
distances = np.linalg.norm(X_robust_scaled - centroid, axis=1)
distance_cutoff = np.quantile(distances, 0.99)
multivariate_outlier_mask = distances > distance_cutoff

print(f'Multivariate outlier count (distance > 99th percentile): {multivariate_outlier_mask.sum()}')
print('Outlier prevalence may distort K-Means centroids; DBSCAN can treat them as noise.')

```

## 8. Clustering Strategy & Executive Summary



```python
summary = {
    'problem': 'Unsupervised player clustering and archetype discovery',
    'dataset': str(DATA_PATH.relative_to(PROJECT_ROOT)) if DATA_PATH.is_relative_to(PROJECT_ROOT) else str(DATA_PATH),
    'samples': X.shape[0],
    'numeric_features_after_cleaning': len(numeric_cols),
    'identifier_columns_removed': [c for c in identifier_cols + high_cardinality_cols],
    'zero_variance_removed': zero_variance,
    'quasi_constant_removed': quasi_constant,
    'heavily_skewed_features_recommended_log_transform': skewed_features,
    'high_pearson_pairs': [(str(a), str(b), float(v)) for a, b, v in pearson_pairs],
    'high_spearman_pairs': [(str(a), str(b), float(v)) for a, b, v in spearman_pairs],
    'pca_cumulative_variance': {f'PC{i}': float(v) for i, v in enumerate(cum_var, 1)},
    'recommended_preprocessing': [
        'Apply log1p transformation to heavily skewed features before scaling',
        'Use RobustScaler for activity/battle counts to reduce outlier influence',
        'Impute missing values with median for numeric features',
        'Drop high-cardinality identifiers before modeling'
    ],
    'dimensionality_reduction_strategy': 'Keep PCA components explaining at least 90% cumulative variance before clustering',
    'clustering_evaluation_metrics': [
        'Silhouette Score',
        'Davies-Bouldin Index',
        'Calinski-Harabasz Index',
        'Elbow Method (inertia sum of squared distances)'
    ],
    'unsupervised_models_to_benchmark': [
        'K-Means',
        'DBSCAN',
        'Agglomerative Hierarchical Clustering'
    ]
}

import json
print('Executive Summary:')
print(json.dumps(summary, indent=2, default=str))

```

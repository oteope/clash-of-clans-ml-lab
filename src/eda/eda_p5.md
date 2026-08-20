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

    Dataset loaded from: c:\Users\Usuario\Desktop\Clash of Clans ML Lab\data\datasets\player_clustering.parquet
    Dimensions: (836830, 31)
    
    Data types:
    player_tag                             str
    town_hall_level                      int64
    builder_hall_level                 float64
    exp_level                            int64
    trophies                             int64
    best_trophies                        int64
    war_stars                            int64
    attack_wins                          int64
    defense_wins                         int64
    donations                            int64
    donations_received                   int64
    clan_capital_contributions           int64
    troop_count                          int64
    troop_mean_level                   float64
    troop_mean_completion_ratio        float64
    hero_count                         float64
    hero_mean_level                    float64
    hero_mean_completion_ratio         float64
    spell_count                        float64
    spell_mean_level                   float64
    spell_mean_completion_ratio        float64
    equipment_count                    float64
    equipment_mean_level               float64
    equipment_mean_completion_ratio    float64
    achievement_count                    int64
    achievement_completion_ratio       float64
    donation_balance                     int64
    donation_ratio                     float64
    combat_activity_total                int64
    progression_ratio_trophies         float64
    builder_progression_ratio          float64
    dtype: object
    
    First 5 rows:
       player_tag  town_hall_level  builder_hall_level  exp_level  trophies  \
    0  #20000290G               11                 7.0        137         0   
    1  #2000092LU               17                10.0        262         0   
    2  #20000C8VR               12                 6.0        159         0   
    3  #20000UJ2L               17                10.0        217       656   
    4  #200022YUC               12                 7.0        160         0   
    
       best_trophies  war_stars  attack_wins  defense_wins  donations  ...  \
    0           2755        313            0             0          0  ...   
    1           5362        502            0             0       4181  ...   
    2           3130        870            0             0          0  ...   
    3           4538       1790            0             0         55  ...   
    4           3217       1074            0             0          1  ...   
    
       equipment_count  equipment_mean_level  equipment_mean_completion_ratio  \
    0             14.0              4.357143                         0.240741   
    1             34.0             12.352941                         0.584423   
    2             15.0              5.066667                         0.281481   
    3             42.0             14.071429                         0.656085   
    4             19.0              6.947368                         0.334308   
    
       achievement_count  achievement_completion_ratio  donation_balance  \
    0                 53                      1.240722                 0   
    1                 53                      6.163203              -276   
    2                 53                      1.743819                 0   
    3                 53                      4.414603                55   
    4                 53                      1.561650                 1   
    
       donation_ratio  combat_activity_total  progression_ratio_trophies  \
    0        0.000000                      0                    0.000000   
    1        0.938075                      0                    0.000000   
    2        0.000000                      0                    0.000000   
    3        0.000000                      0                    0.144557   
    4        0.000000                      0                    0.000000   
    
       builder_progression_ratio  
    0                   0.933227  
    1                   0.973129  
    2                   0.958886  
    3                   0.944381  
    4                   0.957204  
    
    [5 rows x 31 columns]
    

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

    Dimensions: (836830, 31)
    Duplicate rows: 0
    
    Missing values (top 10):
    Series([], dtype: float64)
    
    Identifier columns dropped: ['player_tag']
    High-cardinality columns dropped: ['achievement_completion_ratio']
    Feature matrix shape after dropping identifiers: (836830, 29)
    
    Zero-variance features: ['achievement_count']
    Quasi-constant features (mode frequency >= 95%): ['attack_wins', 'defense_wins', 'achievement_count', 'combat_activity_total']
    X shape after invariant filtering: (836830, 25)
    Numeric features for clustering: 25
    

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

    Statistical summary of numeric features:
                                            mean           std       median          Q1            Q3           IQR          min           max  zero_pct  skewness     kurtosis
    feature                                                                                                                                                                    
    town_hall_level                      13.2108  3.728600e+00      13.0000     11.0000  1.600000e+01  5.000000e+00       1.0000  1.800000e+01    0.0000   -0.5334      -0.3733
    builder_hall_level                    7.1957  2.760500e+00       8.0000      5.0000  1.000000e+01  5.000000e+00       0.0000  1.000000e+01    3.2408   -0.7578      -0.3349
    exp_level                           141.6825  7.199260e+01     139.0000     84.0000  2.030000e+02  1.190000e+02       1.0000  5.000000e+02    0.0000    0.0511      -0.9635
    trophies                            197.0190  5.286795e+02       0.0000      0.0000  1.240000e+02  1.240000e+02       0.0000  5.353000e+03   62.8892    5.3552      38.9850
    best_trophies                      2922.5395  1.760777e+03    2825.0000   1441.0000  4.491000e+03  3.050000e+03       0.0000  7.465000e+03    1.8699    0.0919      -1.1513
    war_stars                           792.3587  1.069586e+03     352.0000     55.0000  1.108000e+03  1.053000e+03       0.0000  1.219000e+04    9.7407    2.0919       5.0310
    donations                           111.5553  5.838789e+03       0.0000      0.0000  1.000000e+01  1.000000e+01       0.0000  3.161641e+06   73.3265  391.3150  176133.9677
    donations_received                  111.1141  1.615284e+03       0.0000      0.0000  4.500000e+01  4.500000e+01       0.0000  3.213720e+05   73.0745  151.3787   26682.7595
    clan_capital_contributions       950114.0583  1.390967e+06  303889.0000  25536.2500  1.312820e+06  1.287284e+06       0.0000  2.739304e+07   11.8339    2.0890       5.0489
    troop_count                          56.9215  1.787260e+01      56.0000     43.0000  7.500000e+01  3.200000e+01      18.0000  8.200000e+01    0.0000   -0.1016      -1.1589
    troop_mean_level                      4.8632  2.117600e+00       4.7455      3.2857  6.303000e+00  3.017300e+00       1.0000  9.811300e+00    0.0000    0.2174      -0.7015
    troop_mean_completion_ratio           0.3904  1.826000e-01       0.3668      0.2490  5.073000e-01  2.583000e-01       0.0847  8.144000e-01    0.0000    0.4827      -0.5466
    hero_count                            5.6097  2.469900e+00       6.0000      4.0000  8.000000e+00  4.000000e+00       0.0000  8.000000e+00    5.3011   -0.8015      -0.4709
    hero_mean_level                      34.1564  1.943790e+01      34.0000     20.0000  4.885710e+01  2.885710e+01       0.0000  7.716670e+01    5.3011   -0.0232      -0.8994
    hero_mean_completion_ratio            0.4421  2.915000e-01       0.4051      0.2101  6.674000e-01  4.573000e-01       0.0000  1.000000e+00    5.3011    0.2834      -0.9804
    spell_count                          12.6152  4.858100e+00      13.0000     11.0000  1.700000e+01  6.000000e+00       0.0000  1.800000e+01    2.6866   -1.0225       0.3305
    spell_mean_level                      4.2094  1.443800e+00       4.3077      3.3333  5.200000e+00  1.866700e+00       0.0000  9.333300e+00    2.6866   -0.5357       0.7250
    spell_mean_completion_ratio           0.5166  2.086000e-01       0.5270      0.3735  6.584000e-01  2.849000e-01       0.0000  1.000000e+00    2.6866   -0.1732      -0.0360
    equipment_count                      20.5586  1.295880e+01      22.0000     10.0000  3.200000e+01  2.200000e+01       0.0000  4.200000e+01   13.5398   -0.1407      -1.1608
    equipment_mean_level                  8.2932  5.769000e+00       8.0000      3.8000  1.216670e+01  8.366700e+00       0.0000  2.185710e+01   13.5398    0.3018      -0.7385
    equipment_mean_completion_ratio       0.4077  2.733000e-01       0.4009      0.2037  5.915000e-01  3.878000e-01       0.0000  1.000000e+00   13.5398    0.2081      -0.7827
    donation_balance                      0.4411  6.045462e+03       0.0000      0.0000  0.000000e+00  0.000000e+00 -321372.0000  3.161641e+06   65.9284  349.6736  153415.6964
    donation_ratio                        0.4550  4.973840e+01       0.0000      0.0000  0.000000e+00  0.000000e+00       0.0000  4.496000e+04   80.7935  883.0692  797797.0393
    progression_ratio_trophies            0.1142  2.766000e-01       0.0000      0.0000  3.980000e-02  3.980000e-02       0.0000  3.800000e+01   62.9323    7.6056     513.3581
    builder_progression_ratio             0.9041  1.831000e-01       0.9569      0.9052  9.844000e-01  7.920000e-02       0.0000  1.400000e+01    3.2633   -3.5476      48.1063
    
    Heavily skewed features (|skewness| > 1.0), consider log1p transformation:
    ['trophies', 'war_stars', 'donations', 'donations_received', 'clan_capital_contributions', 'spell_count', 'donation_balance', 'donation_ratio', 'progression_ratio_trophies', 'builder_progression_ratio']
    
    Scale variance across features:
                    std           IQR
    count  2.500000e+01  2.500000e+01
    mean   5.632089e+04  5.167209e+04
    std    2.780562e+05  2.574199e+05
    min    1.826000e-01  0.000000e+00
    25%    1.443800e+00  3.878000e-01
    50%    5.769000e+00  5.000000e+00
    75%    5.286795e+02  3.200000e+01
    max    1.390967e+06  1.287284e+06
    Justification: Features vary widely in scale, so StandardScaler or RobustScaler should be applied before clustering.
    

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

    Feature group profile:
                   group  n_features  missing_pct  zero_pct  avg_abs_skew
    Progression & Levels           3          0.0      1.08        0.4475
       Activity & Battle           3          0.0     24.83        2.5130
        Social & Economy           2          0.0     73.20      271.3474
    

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

    Pearson high-correlation pairs (|r| > 0.85):
    equipment_mean_level           - equipment_mean_completion_ratio: 0.9970
    troop_mean_level               - troop_mean_completion_ratio   : 0.9877
    hero_mean_level                - hero_mean_completion_ratio    : 0.9843
    spell_mean_level               - spell_mean_completion_ratio   : 0.9836
    donations                      - donation_balance              : 0.9636
    troop_mean_completion_ratio    - hero_mean_completion_ratio    : 0.9526
    town_hall_level                - troop_count                   : 0.9508
    exp_level                      - troop_mean_completion_ratio   : 0.9460
    town_hall_level                - hero_count                    : 0.9455
    exp_level                      - troop_mean_level              : 0.9440
    exp_level                      - hero_mean_completion_ratio    : 0.9436
    exp_level                      - troop_count                   : 0.9414
    troop_mean_level               - hero_mean_completion_ratio    : 0.9371
    hero_count                     - spell_count                   : 0.9364
    exp_level                      - best_trophies                 : 0.9341
    troop_mean_completion_ratio    - hero_mean_level               : 0.9337
    exp_level                      - hero_mean_level               : 0.9324
    troop_count                    - hero_mean_completion_ratio    : 0.9298
    troop_count                    - equipment_count               : 0.9275
    troop_count                    - hero_count                    : 0.9274
    town_hall_level                - spell_count                   : 0.9267
    troop_count                    - spell_count                   : 0.9257
    troop_mean_level               - hero_mean_level               : 0.9237
    equipment_count                - equipment_mean_level          : 0.9164
    hero_count                     - equipment_count               : 0.9140
    troop_count                    - troop_mean_completion_ratio   : 0.9139
    troop_mean_completion_ratio    - spell_mean_completion_ratio   : 0.9118
    town_hall_level                - equipment_count               : 0.9106
    troop_mean_level               - spell_mean_completion_ratio   : 0.9046
    troop_count                    - hero_mean_level               : 0.9043
    troop_mean_completion_ratio    - equipment_mean_level          : 0.9025
    equipment_count                - equipment_mean_completion_ratio: 0.9015
    troop_count                    - troop_mean_level              : 0.8977
    hero_mean_completion_ratio     - equipment_mean_level          : 0.8953
    builder_hall_level             - hero_count                    : 0.8951
    troop_count                    - equipment_mean_level          : 0.8926
    troop_mean_completion_ratio    - equipment_mean_completion_ratio: 0.8925
    town_hall_level                - exp_level                     : 0.8916
    best_trophies                  - hero_mean_completion_ratio    : 0.8910
    best_trophies                  - troop_mean_completion_ratio   : 0.8902
    hero_mean_completion_ratio     - equipment_count               : 0.8889
    hero_mean_level                - spell_mean_completion_ratio   : 0.8882
    builder_hall_level             - troop_count                   : 0.8876
    troop_count                    - equipment_mean_completion_ratio: 0.8855
    spell_count                    - equipment_count               : 0.8842
    hero_mean_completion_ratio     - equipment_mean_completion_ratio: 0.8838
    hero_mean_level                - spell_count                   : 0.8819
    exp_level                      - spell_count                   : 0.8817
    best_trophies                  - troop_mean_level              : 0.8807
    best_trophies                  - troop_count                   : 0.8793
    best_trophies                  - hero_mean_level               : 0.8787
    troop_mean_level               - equipment_mean_level          : 0.8734
    hero_mean_level                - equipment_count               : 0.8713
    hero_mean_level                - equipment_mean_level          : 0.8707
    troop_mean_completion_ratio    - equipment_count               : 0.8707
    hero_mean_completion_ratio     - spell_mean_completion_ratio   : 0.8700
    exp_level                      - spell_mean_completion_ratio   : 0.8699
    troop_mean_level               - equipment_mean_completion_ratio: 0.8682
    town_hall_level                - hero_mean_completion_ratio    : 0.8663
    exp_level                      - hero_count                    : 0.8631
    hero_mean_completion_ratio     - spell_count                   : 0.8624
    hero_mean_level                - equipment_mean_completion_ratio: 0.8623
    exp_level                      - equipment_mean_level          : 0.8605
    exp_level                      - equipment_mean_completion_ratio: 0.8560
    town_hall_level                - equipment_mean_level          : 0.8559
    town_hall_level                - hero_mean_level               : 0.8553
    troop_mean_level               - spell_mean_level              : 0.8544
    troop_mean_completion_ratio    - spell_mean_level              : 0.8543
    exp_level                      - equipment_count               : 0.8542
    town_hall_level                - equipment_mean_completion_ratio: 0.8528
    builder_hall_level             - exp_level                     : 0.8517
    town_hall_level                - troop_mean_completion_ratio   : 0.8516
    
    Spearman high-correlation pairs (|r| > 0.85):
    equipment_mean_level           - equipment_mean_completion_ratio: 0.9972
    troop_mean_level               - troop_mean_completion_ratio   : 0.9894
    hero_mean_level                - hero_mean_completion_ratio    : 0.9892
    trophies                       - progression_ratio_trophies    : 0.9865
    spell_mean_level               - spell_mean_completion_ratio   : 0.9839
    troop_count                    - spell_count                   : 0.9673
    town_hall_level                - troop_count                   : 0.9586
    exp_level                      - troop_mean_completion_ratio   : 0.9578
    troop_mean_completion_ratio    - hero_mean_completion_ratio    : 0.9544
    exp_level                      - hero_mean_completion_ratio    : 0.9496
    troop_count                    - hero_count                    : 0.9483
    exp_level                      - troop_count                   : 0.9471
    exp_level                      - troop_mean_level              : 0.9451
    town_hall_level                - hero_count                    : 0.9430
    troop_count                    - hero_mean_completion_ratio    : 0.9425
    troop_count                    - troop_mean_completion_ratio   : 0.9399
    troop_mean_completion_ratio    - hero_mean_level               : 0.9391
    exp_level                      - best_trophies                 : 0.9391
    spell_count                    - equipment_count               : 0.9367
    hero_count                     - equipment_count               : 0.9355
    troop_mean_level               - hero_mean_completion_ratio    : 0.9353
    troop_count                    - equipment_count               : 0.9352
    exp_level                      - hero_mean_level               : 0.9306
    equipment_count                - equipment_mean_level          : 0.9301
    hero_count                     - spell_count                   : 0.9294
    hero_mean_completion_ratio     - spell_count                   : 0.9288
    town_hall_level                - equipment_count               : 0.9262
    town_hall_level                - spell_count                   : 0.9240
    troop_mean_completion_ratio    - spell_count                   : 0.9230
    troop_mean_completion_ratio    - spell_mean_completion_ratio   : 0.9222
    troop_mean_level               - hero_mean_level               : 0.9172
    exp_level                      - spell_count                   : 0.9165
    equipment_count                - equipment_mean_completion_ratio: 0.9092
    troop_count                    - troop_mean_level              : 0.9081
    troop_count                    - hero_mean_level               : 0.9077
    troop_count                    - equipment_mean_level          : 0.9073
    builder_hall_level             - troop_count                   : 0.9032
    hero_mean_level                - spell_count                   : 0.9031
    troop_mean_level               - spell_mean_completion_ratio   : 0.9012
    builder_hall_level             - hero_count                    : 0.8991
    hero_mean_completion_ratio     - equipment_count               : 0.8987
    best_trophies                  - troop_mean_completion_ratio   : 0.8976
    troop_count                    - equipment_mean_completion_ratio: 0.8976
    spell_count                    - equipment_mean_level          : 0.8974
    town_hall_level                - exp_level                     : 0.8972
    best_trophies                  - hero_mean_completion_ratio    : 0.8925
    troop_mean_completion_ratio    - equipment_mean_level          : 0.8905
    troop_mean_completion_ratio    - equipment_count               : 0.8879
    hero_mean_level                - spell_mean_completion_ratio   : 0.8875
    hero_mean_completion_ratio     - equipment_mean_level          : 0.8868
    town_hall_level                - hero_mean_completion_ratio    : 0.8860
    hero_count                     - equipment_mean_level          : 0.8860
    exp_level                      - hero_count                    : 0.8856
    best_trophies                  - troop_count                   : 0.8855
    spell_count                    - equipment_mean_completion_ratio: 0.8851
    troop_mean_completion_ratio    - equipment_mean_completion_ratio: 0.8841
    troop_mean_completion_ratio    - hero_count                    : 0.8840
    town_hall_level                - equipment_mean_level          : 0.8839
    hero_count                     - hero_mean_completion_ratio    : 0.8832
    best_trophies                  - troop_mean_level              : 0.8828
    troop_mean_level               - spell_count                   : 0.8827
    hero_mean_completion_ratio     - spell_mean_completion_ratio   : 0.8816
    town_hall_level                - troop_mean_completion_ratio   : 0.8794
    exp_level                      - war_stars                     : 0.8791
    best_trophies                  - hero_mean_level               : 0.8790
    hero_mean_completion_ratio     - equipment_mean_completion_ratio: 0.8769
    hero_count                     - equipment_mean_completion_ratio: 0.8741
    town_hall_level                - equipment_mean_completion_ratio: 0.8738
    builder_hall_level             - exp_level                     : 0.8737
    hero_mean_level                - equipment_count               : 0.8731
    troop_mean_completion_ratio    - spell_mean_level              : 0.8722
    exp_level                      - spell_mean_completion_ratio   : 0.8687
    hero_mean_level                - equipment_mean_level          : 0.8646
    builder_hall_level             - hero_mean_completion_ratio    : 0.8626
    exp_level                      - equipment_count               : 0.8595
    clan_capital_contributions     - equipment_count               : 0.8590
    troop_mean_level               - equipment_mean_level          : 0.8576
    troop_mean_level               - spell_mean_level              : 0.8574
    builder_hall_level             - troop_mean_level              : 0.8574
    builder_hall_level             - troop_mean_completion_ratio   : 0.8563
    exp_level                      - equipment_mean_level          : 0.8548
    hero_mean_level                - equipment_mean_completion_ratio: 0.8547
    troop_mean_level               - equipment_mean_completion_ratio: 0.8541
    best_trophies                  - spell_count                   : 0.8528
    troop_mean_level               - hero_count                    : 0.8518
    exp_level                      - equipment_mean_completion_ratio: 0.8505
    town_hall_level                - builder_hall_level            : 0.8505
    


    
![png](eda_p5_files/eda_p5_10_1.png)
    



    
![png](eda_p5_files/eda_p5_10_2.png)
    


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


    
![png](eda_p5_files/eda_p5_12_0.png)
    



    
![png](eda_p5_files/eda_p5_12_1.png)
    


    Explained variance by first 2 PCs: 0.6837
    Cumulative variance for each component:
    PC1: 0.6039
    PC2: 0.6837
    PC3: 0.7509
    PC4: 0.8006
    PC5: 0.8407
    PC6: 0.8806
    PC7: 0.9074
    PC8: 0.9275
    PC9: 0.9431
    PC10: 0.9556
    PC11: 0.9665
    PC12: 0.9749
    PC13: 0.9820
    PC14: 0.9872
    PC15: 0.9907
    PC16: 0.9937
    PC17: 0.9960
    PC18: 0.9976
    PC19: 0.9986
    PC20: 0.9993
    

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

    Univariate IQR outlier counts per feature (top 10):
    donation_balance              285121
    donations                     202148
    donation_ratio                160726
    donations_received            150617
    progression_ratio_trophies    147510
    trophies                      139003
    builder_progression_ratio      73379
    clan_capital_contributions     71117
    war_stars                      59115
    spell_count                    38442
    
    Multivariate outlier count (distance > 99th percentile): 8369
    Outlier prevalence may distort K-Means centroids; DBSCAN can treat them as noise.
    

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

    Executive Summary:
    {
      "problem": "Unsupervised player clustering and archetype discovery",
      "dataset": "data\\datasets\\player_clustering.parquet",
      "samples": 836830,
      "numeric_features_after_cleaning": 25,
      "identifier_columns_removed": [
        "player_tag",
        "player_tag",
        "achievement_completion_ratio"
      ],
      "zero_variance_removed": [
        "achievement_count"
      ],
      "quasi_constant_removed": [
        "attack_wins",
        "defense_wins",
        "achievement_count",
        "combat_activity_total"
      ],
      "heavily_skewed_features_recommended_log_transform": [
        "trophies",
        "war_stars",
        "donations",
        "donations_received",
        "clan_capital_contributions",
        "spell_count",
        "donation_balance",
        "donation_ratio",
        "progression_ratio_trophies",
        "builder_progression_ratio"
      ],
      "high_pearson_pairs": [
        [
          "equipment_mean_level",
          "equipment_mean_completion_ratio",
          0.9970205550570587
        ],
        [
          "troop_mean_level",
          "troop_mean_completion_ratio",
          0.9876575310380487
        ],
        [
          "hero_mean_level",
          "hero_mean_completion_ratio",
          0.9843275309829085
        ],
        [
          "spell_mean_level",
          "spell_mean_completion_ratio",
          0.9835761915244703
        ],
        [
          "donations",
          "donation_balance",
          0.9636464207258181
        ],
        [
          "troop_mean_completion_ratio",
          "hero_mean_completion_ratio",
          0.9525765868794617
        ],
        [
          "town_hall_level",
          "troop_count",
          0.9508323015059112
        ],
        [
          "exp_level",
          "troop_mean_completion_ratio",
          0.9460290089304936
        ],
        [
          "town_hall_level",
          "hero_count",
          0.9454551955636582
        ],
        [
          "exp_level",
          "troop_mean_level",
          0.9439820305375572
        ],
        [
          "exp_level",
          "hero_mean_completion_ratio",
          0.9435929325247647
        ],
        [
          "exp_level",
          "troop_count",
          0.941424672464156
        ],
        [
          "troop_mean_level",
          "hero_mean_completion_ratio",
          0.9371121562253982
        ],
        [
          "hero_count",
          "spell_count",
          0.9364470927376137
        ],
        [
          "exp_level",
          "best_trophies",
          0.9341228652885565
        ],
        [
          "troop_mean_completion_ratio",
          "hero_mean_level",
          0.9337157860004395
        ],
        [
          "exp_level",
          "hero_mean_level",
          0.9323648036967784
        ],
        [
          "troop_count",
          "hero_mean_completion_ratio",
          0.9298007729408236
        ],
        [
          "troop_count",
          "equipment_count",
          0.9275012506754939
        ],
        [
          "troop_count",
          "hero_count",
          0.9274141817000469
        ],
        [
          "town_hall_level",
          "spell_count",
          0.9267427568358542
        ],
        [
          "troop_count",
          "spell_count",
          0.9256985987058725
        ],
        [
          "troop_mean_level",
          "hero_mean_level",
          0.9236875984636393
        ],
        [
          "equipment_count",
          "equipment_mean_level",
          0.9163657118360411
        ],
        [
          "hero_count",
          "equipment_count",
          0.9140307409054056
        ],
        [
          "troop_count",
          "troop_mean_completion_ratio",
          0.9138946024786213
        ],
        [
          "troop_mean_completion_ratio",
          "spell_mean_completion_ratio",
          0.911802164925932
        ],
        [
          "town_hall_level",
          "equipment_count",
          0.9106463091894411
        ],
        [
          "troop_mean_level",
          "spell_mean_completion_ratio",
          0.9046165184080197
        ],
        [
          "troop_count",
          "hero_mean_level",
          0.9042560845942712
        ],
        [
          "troop_mean_completion_ratio",
          "equipment_mean_level",
          0.9025143621046605
        ],
        [
          "equipment_count",
          "equipment_mean_completion_ratio",
          0.9015240104639196
        ],
        [
          "troop_count",
          "troop_mean_level",
          0.8977101675116045
        ],
        [
          "hero_mean_completion_ratio",
          "equipment_mean_level",
          0.8952584299265399
        ],
        [
          "builder_hall_level",
          "hero_count",
          0.8950832699717386
        ],
        [
          "troop_count",
          "equipment_mean_level",
          0.8925894740765673
        ],
        [
          "troop_mean_completion_ratio",
          "equipment_mean_completion_ratio",
          0.8925456865226108
        ],
        [
          "town_hall_level",
          "exp_level",
          0.8916214873597024
        ],
        [
          "best_trophies",
          "hero_mean_completion_ratio",
          0.8909702775640584
        ],
        [
          "best_trophies",
          "troop_mean_completion_ratio",
          0.8902085708209203
        ],
        [
          "hero_mean_completion_ratio",
          "equipment_count",
          0.888936746393464
        ],
        [
          "hero_mean_level",
          "spell_mean_completion_ratio",
          0.8881902805494822
        ],
        [
          "builder_hall_level",
          "troop_count",
          0.8875801444756007
        ],
        [
          "troop_count",
          "equipment_mean_completion_ratio",
          0.8855103043060323
        ],
        [
          "spell_count",
          "equipment_count",
          0.8841813049852403
        ],
        [
          "hero_mean_completion_ratio",
          "equipment_mean_completion_ratio",
          0.8838141306905897
        ],
        [
          "hero_mean_level",
          "spell_count",
          0.88188368374699
        ],
        [
          "exp_level",
          "spell_count",
          0.8817308219994304
        ],
        [
          "best_trophies",
          "troop_mean_level",
          0.8806641715249357
        ],
        [
          "best_trophies",
          "troop_count",
          0.8792806347485193
        ],
        [
          "best_trophies",
          "hero_mean_level",
          0.8787267488219039
        ],
        [
          "troop_mean_level",
          "equipment_mean_level",
          0.8734198281216841
        ],
        [
          "hero_mean_level",
          "equipment_count",
          0.8713347462038958
        ],
        [
          "hero_mean_level",
          "equipment_mean_level",
          0.8707246307958312
        ],
        [
          "troop_mean_completion_ratio",
          "equipment_count",
          0.8706600384671603
        ],
        [
          "hero_mean_completion_ratio",
          "spell_mean_completion_ratio",
          0.869997558677027
        ],
        [
          "exp_level",
          "spell_mean_completion_ratio",
          0.8698929886918175
        ],
        [
          "troop_mean_level",
          "equipment_mean_completion_ratio",
          0.8682172729352925
        ],
        [
          "town_hall_level",
          "hero_mean_completion_ratio",
          0.8662514535059798
        ],
        [
          "exp_level",
          "hero_count",
          0.8631276861184982
        ],
        [
          "hero_mean_completion_ratio",
          "spell_count",
          0.8623552676179685
        ],
        [
          "hero_mean_level",
          "equipment_mean_completion_ratio",
          0.862311980673092
        ],
        [
          "exp_level",
          "equipment_mean_level",
          0.8604581440818542
        ],
        [
          "exp_level",
          "equipment_mean_completion_ratio",
          0.8559949023584552
        ],
        [
          "town_hall_level",
          "equipment_mean_level",
          0.8559192370386024
        ],
        [
          "town_hall_level",
          "hero_mean_level",
          0.8553295869825354
        ],
        [
          "troop_mean_level",
          "spell_mean_level",
          0.8544317301873668
        ],
        [
          "troop_mean_completion_ratio",
          "spell_mean_level",
          0.8543126279068934
        ],
        [
          "exp_level",
          "equipment_count",
          0.8542461097496503
        ],
        [
          "town_hall_level",
          "equipment_mean_completion_ratio",
          0.8528223026195014
        ],
        [
          "builder_hall_level",
          "exp_level",
          0.8516726306052529
        ],
        [
          "town_hall_level",
          "troop_mean_completion_ratio",
          0.8515638267717058
        ]
      ],
      "high_spearman_pairs": [
        [
          "equipment_mean_level",
          "equipment_mean_completion_ratio",
          0.9972105997953977
        ],
        [
          "troop_mean_level",
          "troop_mean_completion_ratio",
          0.9893591043091712
        ],
        [
          "hero_mean_level",
          "hero_mean_completion_ratio",
          0.9891652144364663
        ],
        [
          "trophies",
          "progression_ratio_trophies",
          0.9865269511106622
        ],
        [
          "spell_mean_level",
          "spell_mean_completion_ratio",
          0.9838632063200874
        ],
        [
          "troop_count",
          "spell_count",
          0.9672831520793479
        ],
        [
          "town_hall_level",
          "troop_count",
          0.9585971163351554
        ],
        [
          "exp_level",
          "troop_mean_completion_ratio",
          0.9578024680194912
        ],
        [
          "troop_mean_completion_ratio",
          "hero_mean_completion_ratio",
          0.9544305047518358
        ],
        [
          "exp_level",
          "hero_mean_completion_ratio",
          0.9495809398818071
        ],
        [
          "troop_count",
          "hero_count",
          0.9482952934862265
        ],
        [
          "exp_level",
          "troop_count",
          0.9470510697939925
        ],
        [
          "exp_level",
          "troop_mean_level",
          0.9450682510031122
        ],
        [
          "town_hall_level",
          "hero_count",
          0.9430005540875667
        ],
        [
          "troop_count",
          "hero_mean_completion_ratio",
          0.9425319741439802
        ],
        [
          "troop_count",
          "troop_mean_completion_ratio",
          0.9398936404319556
        ],
        [
          "troop_mean_completion_ratio",
          "hero_mean_level",
          0.9391042696233294
        ],
        [
          "exp_level",
          "best_trophies",
          0.9390695641591144
        ],
        [
          "spell_count",
          "equipment_count",
          0.9366948505242337
        ],
        [
          "hero_count",
          "equipment_count",
          0.9354885856978563
        ],
        [
          "troop_mean_level",
          "hero_mean_completion_ratio",
          0.9352740250356383
        ],
        [
          "troop_count",
          "equipment_count",
          0.935160173535846
        ],
        [
          "exp_level",
          "hero_mean_level",
          0.9305549849872841
        ],
        [
          "equipment_count",
          "equipment_mean_level",
          0.9300846496316316
        ],
        [
          "hero_count",
          "spell_count",
          0.929403364374392
        ],
        [
          "hero_mean_completion_ratio",
          "spell_count",
          0.928817985035059
        ],
        [
          "town_hall_level",
          "equipment_count",
          0.9262094121638835
        ],
        [
          "town_hall_level",
          "spell_count",
          0.9240498551298275
        ],
        [
          "troop_mean_completion_ratio",
          "spell_count",
          0.9229993080144308
        ],
        [
          "troop_mean_completion_ratio",
          "spell_mean_completion_ratio",
          0.9221582395794519
        ],
        [
          "troop_mean_level",
          "hero_mean_level",
          0.9172162150601803
        ],
        [
          "exp_level",
          "spell_count",
          0.9164522200700143
        ],
        [
          "equipment_count",
          "equipment_mean_completion_ratio",
          0.90916487217673
        ],
        [
          "troop_count",
          "troop_mean_level",
          0.9081484215573971
        ],
        [
          "troop_count",
          "hero_mean_level",
          0.907724238724275
        ],
        [
          "troop_count",
          "equipment_mean_level",
          0.9072889579476767
        ],
        [
          "builder_hall_level",
          "troop_count",
          0.9031656585330575
        ],
        [
          "hero_mean_level",
          "spell_count",
          0.903138915086686
        ],
        [
          "troop_mean_level",
          "spell_mean_completion_ratio",
          0.901214053959276
        ],
        [
          "builder_hall_level",
          "hero_count",
          0.8990680520021042
        ],
        [
          "hero_mean_completion_ratio",
          "equipment_count",
          0.898741282318796
        ],
        [
          "best_trophies",
          "troop_mean_completion_ratio",
          0.8976290407252167
        ],
        [
          "troop_count",
          "equipment_mean_completion_ratio",
          0.8976278997232092
        ],
        [
          "spell_count",
          "equipment_mean_level",
          0.8973602057190437
        ],
        [
          "town_hall_level",
          "exp_level",
          0.8972015932212553
        ],
        [
          "best_trophies",
          "hero_mean_completion_ratio",
          0.8925375190958887
        ],
        [
          "troop_mean_completion_ratio",
          "equipment_mean_level",
          0.8905402522818514
        ],
        [
          "troop_mean_completion_ratio",
          "equipment_count",
          0.8878719609300202
        ],
        [
          "hero_mean_level",
          "spell_mean_completion_ratio",
          0.887489447719897
        ],
        [
          "hero_mean_completion_ratio",
          "equipment_mean_level",
          0.8868099455661557
        ],
        [
          "town_hall_level",
          "hero_mean_completion_ratio",
          0.886035279221267
        ],
        [
          "hero_count",
          "equipment_mean_level",
          0.8859864333930398
        ],
        [
          "exp_level",
          "hero_count",
          0.8856189889316435
        ],
        [
          "best_trophies",
          "troop_count",
          0.8855248284460441
        ],
        [
          "spell_count",
          "equipment_mean_completion_ratio",
          0.88514945785217
        ],
        [
          "troop_mean_completion_ratio",
          "equipment_mean_completion_ratio",
          0.884095171145265
        ],
        [
          "troop_mean_completion_ratio",
          "hero_count",
          0.8839843709733027
        ],
        [
          "town_hall_level",
          "equipment_mean_level",
          0.8838768116059399
        ],
        [
          "hero_count",
          "hero_mean_completion_ratio",
          0.8832493226469849
        ],
        [
          "best_trophies",
          "troop_mean_level",
          0.8828132353576089
        ],
        [
          "troop_mean_level",
          "spell_count",
          0.8827418272705023
        ],
        [
          "hero_mean_completion_ratio",
          "spell_mean_completion_ratio",
          0.8815561885781629
        ],
        [
          "town_hall_level",
          "troop_mean_completion_ratio",
          0.8794107622210569
        ],
        [
          "exp_level",
          "war_stars",
          0.8790616112550812
        ],
        [
          "best_trophies",
          "hero_mean_level",
          0.8790322008114863
        ],
        [
          "hero_mean_completion_ratio",
          "equipment_mean_completion_ratio",
          0.8769491007102925
        ],
        [
          "hero_count",
          "equipment_mean_completion_ratio",
          0.8740936240853611
        ],
        [
          "town_hall_level",
          "equipment_mean_completion_ratio",
          0.8738352720138246
        ],
        [
          "builder_hall_level",
          "exp_level",
          0.8737083903546674
        ],
        [
          "hero_mean_level",
          "equipment_count",
          0.8731287826831884
        ],
        [
          "troop_mean_completion_ratio",
          "spell_mean_level",
          0.8722153621867246
        ],
        [
          "exp_level",
          "spell_mean_completion_ratio",
          0.8686808574910994
        ],
        [
          "hero_mean_level",
          "equipment_mean_level",
          0.8645589217636283
        ],
        [
          "builder_hall_level",
          "hero_mean_completion_ratio",
          0.8625824390510557
        ],
        [
          "exp_level",
          "equipment_count",
          0.8595360697854445
        ],
        [
          "clan_capital_contributions",
          "equipment_count",
          0.8589989840334781
        ],
        [
          "troop_mean_level",
          "equipment_mean_level",
          0.8576294720493568
        ],
        [
          "troop_mean_level",
          "spell_mean_level",
          0.8573991578596092
        ],
        [
          "builder_hall_level",
          "troop_mean_level",
          0.8573648929689653
        ],
        [
          "builder_hall_level",
          "troop_mean_completion_ratio",
          0.8563134119486547
        ],
        [
          "exp_level",
          "equipment_mean_level",
          0.8548022793943764
        ],
        [
          "hero_mean_level",
          "equipment_mean_completion_ratio",
          0.8546742506702752
        ],
        [
          "troop_mean_level",
          "equipment_mean_completion_ratio",
          0.8540553241990798
        ],
        [
          "best_trophies",
          "spell_count",
          0.8528382518960808
        ],
        [
          "troop_mean_level",
          "hero_count",
          0.8517675946422655
        ],
        [
          "exp_level",
          "equipment_mean_completion_ratio",
          0.8505111268763734
        ],
        [
          "town_hall_level",
          "builder_hall_level",
          0.8504992128603439
        ]
      ],
      "pca_cumulative_variance": {
        "PC1": 0.603869652769268,
        "PC2": 0.683715113675948,
        "PC3": 0.7508719682702929,
        "PC4": 0.8005582612802525,
        "PC5": 0.840689885332959,
        "PC6": 0.8806166259556786,
        "PC7": 0.9073576334226564,
        "PC8": 0.9275376168286326,
        "PC9": 0.9430968762712317,
        "PC10": 0.9555745712666617,
        "PC11": 0.9664783118234641,
        "PC12": 0.9749472735153872,
        "PC13": 0.981971730660999,
        "PC14": 0.9871707412940754,
        "PC15": 0.9906862490779614,
        "PC16": 0.9936720170551928,
        "PC17": 0.9960342646837925,
        "PC18": 0.997573319444368,
        "PC19": 0.9986437050415908,
        "PC20": 0.9993263113880897
      },
      "recommended_preprocessing": [
        "Apply log1p transformation to heavily skewed features before scaling",
        "Use RobustScaler for activity/battle counts to reduce outlier influence",
        "Impute missing values with median for numeric features",
        "Drop high-cardinality identifiers before modeling"
      ],
      "dimensionality_reduction_strategy": "Keep PCA components explaining at least 90% cumulative variance before clustering",
      "clustering_evaluation_metrics": [
        "Silhouette Score",
        "Davies-Bouldin Index",
        "Calinski-Harabasz Index",
        "Elbow Method (inertia sum of squared distances)"
      ],
      "unsupervised_models_to_benchmark": [
        "K-Means",
        "DBSCAN",
        "Agglomerative Hierarchical Clustering"
      ]
    }
    

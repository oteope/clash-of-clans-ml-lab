from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"

PLAYER_FEATURES_FILE = FEATURES_DIR / "player_features.parquet"
OUTPUT_FILE = DATASETS_DIR / "player_clustering.parquet"

# Features base solicitadas explícitamente.
CANDIDATE_FEATURES: List[str] = [
    "town_hall_level",
    "builder_hall_level",
    "exp_level",
    "trophies",
    "best_trophies",
    "war_stars",
    "attack_wins",
    "defense_wins",
    "donations",
    "donations_received",
    "clan_capital_contributions",
    "troop_count",
    "troop_mean_level",
    "troop_mean_completion_ratio",
    "hero_count",
    "hero_mean_level",
    "hero_mean_completion_ratio",
    "spell_count",
    "spell_mean_level",
    "spell_mean_completion_ratio",
    "equipment_count",
    "equipment_mean_level",
    "equipment_mean_completion_ratio",
    "achievement_count",
    "achievement_completion_ratio",
]

# Features derivadas numéricas citadas como ejemplos en el enunciado.
DERIVED_CANDIDATE_FEATURES: List[str] = [
    "donation_balance",
    "donation_ratio",
    "combat_activity_total",
    "progression_ratio_trophies",
    "builder_progression_ratio",
]

# Columnas que nunca deben usarse como features de clustering.
EXCLUDED_COLUMNS = {
    "player_tag",
    "clan_tag",
    "role",
    "clan_rank",
    "previous_clan_rank",
    "war_success_rate",
    "war_win_rate",
    "performance_class",
}

# Subcadenas que indican targets, labels, predicciones o agrupaciones.
EXCLUDED_SUBSTRINGS = (
    "target",
    "label",
    "prediction",
    "cluster",
    "clan_rank",
    "role",
)


def _load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de features: {path}")
    return pd.read_parquet(path)


def _ensure_player_tag_column(df: pd.DataFrame) -> pd.DataFrame:
    if "player_tag" not in df.columns:
        if df.index.name == "player_tag":
            df = df.reset_index()
        else:
            raise KeyError(
                "El DataFrame no contiene la columna 'player_tag' ni el índice 'player_tag'."
            )
    return df


def _is_excluded_column(name: str) -> bool:
    lowered = name.lower()
    return name in EXCLUDED_COLUMNS or any(
        substring in lowered for substring in EXCLUDED_SUBSTRINGS
    )


def _document_missing(df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, Any]]:
    missing: Dict[str, Dict[str, Any]] = {}
    for col in columns:
        missing_count = int(df[col].isna().sum())
        missing[col] = {
            "missing_count": missing_count,
            "missing_pct": float(missing_count / len(df)),
        }
    return missing


def build_player_clustering_dataset(
    player_features_path: Path = PLAYER_FEATURES_FILE,
    output_path: Path = OUTPUT_FILE,
) -> Dict[str, Any]:
    """Construye el dataset de clustering de jugadores a partir de player_features.parquet.

    Solo se utilizan las columnas incluidas en la whitelist explícita.
    No se aplica escalado, PCA ni selección de clusters.
    """
    df = _load_features(player_features_path)
    df = _ensure_player_tag_column(df)

    if df["player_tag"].duplicated().any():
        duplicate_count = int(df["player_tag"].duplicated().sum())
        raise ValueError(
            f"player_tag debe ser único. Se encontraron {duplicate_count} duplicados."
        )

    player_tag = df["player_tag"].copy()

    # Separar features del identificador.
    feature_df = df.drop(columns=["player_tag"])

    excluded_columns = [col for col in feature_df.columns if _is_excluded_column(col)]
    feature_df = feature_df.drop(columns=excluded_columns)

    # WHITELIST explícita: CANDIDATE_FEATURES + DERIVED_CANDIDATE_FEATURES
    whitelist = list(dict.fromkeys(CANDIDATE_FEATURES + DERIVED_CANDIDATE_FEATURES))
    candidate_features_total = len(whitelist)

    available_whitelist = [col for col in whitelist if col in feature_df.columns]
    missing_whitelist = [col for col in whitelist if col not in feature_df.columns]

    numeric_candidate_cols = [
        col
        for col in available_whitelist
        if pd.api.types.is_numeric_dtype(feature_df[col])
    ]
    non_numeric_candidate_cols = [
        col
        for col in available_whitelist
        if not pd.api.types.is_numeric_dtype(feature_df[col])
    ]

    if not numeric_candidate_cols:
        raise ValueError(
            "No se encontraron features numéricas de la whitelist en player_features.parquet."
        )

    missing_info = _document_missing(feature_df, numeric_candidate_cols)

    imputed_df = feature_df[numeric_candidate_cols].copy()
    dropped_all_missing: List[str] = []
    imputed_medians: Dict[str, float] = {}

    for col in numeric_candidate_cols:
        n_missing = missing_info[col]["missing_count"]

        # Si toda la columna está vacía, no aporta información.
        if n_missing == len(feature_df):
            imputed_df.drop(columns=[col], inplace=True)
            dropped_all_missing.append(col)
            continue

        if n_missing > 0:
            median_value = float(feature_df[col].median(skipna=True))
            if pd.isna(median_value):
                imputed_df.drop(columns=[col], inplace=True)
                dropped_all_missing.append(col)
            else:
                imputed_df[col] = imputed_df[col].fillna(median_value)
                imputed_medians[col] = median_value

    final_feature_cols = imputed_df.columns.tolist()

    # Validaciones finales.
    if imputed_df.isna().any().any():
        raise ValueError("El dataset final aún contiene valores missing.")

    if not all(
        pd.api.types.is_numeric_dtype(imputed_df[col]) for col in final_feature_cols
    ):
        raise ValueError("El dataset final contiene columnas no numéricas.")

    final_df = pd.concat([player_tag, imputed_df], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(output_path, index=False)

    report: Dict[str, Any] = {
        "input_rows": len(df),
        "duplicate_player_tags": 0,
        "whitelist_feature_count": candidate_features_total,
        "available_candidate_features": available_whitelist,
        "missing_candidate_features": missing_whitelist,
        "non_numeric_candidate_features": non_numeric_candidate_cols,
        "numeric_candidate_features": numeric_candidate_cols,
        "excluded_columns": excluded_columns,
        "numeric_feature_count_before_imputation": len(numeric_candidate_cols),
        "missing_columns": missing_info,
        "imputed_medians": imputed_medians,
        "dropped_all_missing_columns": dropped_all_missing,
        "final_feature_count": len(final_feature_cols),
        "final_features": final_feature_cols,
        "output_path": str(output_path),
    }

    print(f"Dataset de clustering guardado en: {output_path}")
    print(f"Features finales: {len(final_feature_cols)}")
    print(f"Features: {final_feature_cols}")
    return report


if __name__ == "__main__":
    build_player_clustering_dataset()

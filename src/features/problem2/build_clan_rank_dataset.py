from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.features.problem2.clan_rank_analysis import audit_clan_rank_proxies

PROCESSED_DIR = Path("data/processed")
FEATURES_DIR = Path("data/features")
DATASETS_DIR = Path("data/datasets")

# Columnas que NO pueden utilizarse como features del Problema 2
FORBIDDEN_FEATURES = {"clan_rank", "previous_clan_rank", "role"}


def load_inputs(
    processed_dir: Path = PROCESSED_DIR,
    features_dir: Path = FEATURES_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga las tablas compactas necesarias y las player features ya calculadas."""
    clan_members = pd.read_parquet(processed_dir / "clan_members.parquet")
    clans = pd.read_parquet(processed_dir / "clans.parquet")
    player_features = pd.read_parquet(features_dir / "player_features.parquet")
    return clan_members, clans, player_features


def select_clan_context_features(clans_df: pd.DataFrame) -> pd.DataFrame:
    """Selecciona características estructurales razonables del clan."""
    desired_cols = [
        "clan_tag",
        "clan_level",
        "clan_points",
        "clan_capital_points",
        "members",
        "required_trophies",
        "war_frequency",
        "war_league",
        "capital_league",
        "type",
        "is_family_friendly",
    ]
    cols = [c for c in desired_cols if c in clans_df.columns]
    return clans_df[cols].copy()


def _compute_relative_features(
    df: pd.DataFrame,
    relative_cols: List[str],
) -> pd.DataFrame:
    """
    Calcula características relativas dentro del clan:
      - diferencia respecto a la media
      - ratio respecto a la media
      - percentil dentro del clan
    """
    result = pd.DataFrame(index=df.index)
    result["player_tag"] = df["player_tag"]
    result["clan_tag"] = df["clan_tag"]

    for col in relative_cols:
        if col not in df.columns:
            continue
        mean_col = f"clan_mean_{col}"
        df[mean_col] = df.groupby("clan_tag")[col].transform("mean")
        result[f"{col}_diff_from_clan_mean"] = df[col] - df[mean_col]
        result[f"{col}_ratio_to_clan_mean"] = (
            df[col] / df[mean_col].replace(0, np.nan)
        )
        result[f"{col}_clan_pct"] = df.groupby("clan_tag")[col].rank(
            method="average", pct=True
        )

    # Limpieza de NaN e Inf
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.fillna(0)
    return result


def _merge_preserving_clan_values(
    clan_members: pd.DataFrame,
    player_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Une clan_members con player_features usando player_tag,
    manteniendo los valores originales de clan_members cuando exista colisión.
    """
    merged = clan_members.merge(
        player_features,
        on="player_tag",
        how="inner",
        suffixes=("", "_player"),
    )

    duplicate_cols = [
        c for c in merged.columns
        if c.endswith("_player") and c[:-7] in merged.columns
    ]
    if duplicate_cols:
        merged = merged.drop(columns=duplicate_cols)

    return merged


def _is_variable_banned(feature_name: str, banned_vars: set) -> bool:
    """
    Devuelve True si la columna debe excluirse.
    Se excluye la columna exacta o cualquier derivada que empiece por `<var>_`.
    """
    if feature_name in banned_vars:
        return True
    for var in banned_vars:
        if feature_name.startswith(f"{var}_"):
            return True
    return False


def build_clan_rank_features(
    clan_members_df: pd.DataFrame,
    player_features_df: pd.DataFrame,
    clans_df: pd.DataFrame,
    include_trophies: bool = True,
) -> pd.DataFrame:
    """
    Construye el dataset para el Problema 2: target = clan_rank.

    Una fila por (clan_tag, player_tag).
    Las relaciones sin player_features se excluyen mediante inner join.
    Las features excluidas por leakage/proxy se eliminan antes de generar el dataset.

    Parámetro:
    ----------
    include_trophies : bool
        Si es True, se conservan las features de trophies.
        Si es False, se excluyen trophies y todas sus variables relacionadas:
            - trophies
            - trophies_diff_from_clan_mean
            - trophies_ratio_to_clan_mean
            - trophies_clan_pct
            - best_trophies
            - progression_ratio_trophies
            - clan_mean_trophies
            - required_trophies
    """
    # 1) Auditoría de posibles proxies
    analysis_df = audit_clan_rank_proxies(clan_members_df, player_features_df)

    banned_vars = set(FORBIDDEN_FEATURES)
    for _, row in analysis_df.iterrows():
        if row["classification"] in {"EXCLUDE", "TOO_DIRECT"}:
            banned_vars.add(row["variable"])

    # 2) Ajuste específico para trophies según la variante solicitada
    if include_trophies:
        banned_vars.discard("trophies")
    else:
        # Excluir todas las columnas relacionadas con trophies para 2B
        trophy_related_banned = {
            "trophies",
            "trophies_diff_from_clan_mean",
            "trophies_ratio_to_clan_mean",
            "trophies_clan_pct",
            "best_trophies",
            "progression_ratio_trophies",
            "clan_mean_trophies",
            "required_trophies",
        }
        banned_vars.update(trophy_related_banned)

    # 3) Join principal
    merged = _merge_preserving_clan_values(clan_members_df, player_features_df)

    # 4) Características relativas al clan
    relative_cols = [
        "trophies",
        "exp_level",
        "town_hall_level",
        "donations",
        "donations_received",
        "capital_contributions",
        "war_stars",
        "attack_wins",
        "defense_wins",
    ]
    rel_features = _compute_relative_features(merged, relative_cols)

    # 5) Contexto estructural del clan
    clan_ctx = select_clan_context_features(clans_df)
    output = merged.merge(clan_ctx, on="clan_tag", how="left")

    # 6) Añadir features relativas
    output = output.merge(
        rel_features,
        on=["player_tag", "clan_tag"],
        how="left",
    )

    # 7) Seleccionar columnas finales aplicando política anti-leakage
    target_col = "clan_rank"
    id_cols = ["player_tag", "clan_tag"]

    feature_cols = [
        c for c in output.columns
        if c not in id_cols
        and c != target_col
        and not _is_variable_banned(c, banned_vars)
    ]

    final = output[id_cols + feature_cols + [target_col]].copy()

    # 8) Verificaciones de seguridad
    assert target_col not in feature_cols, "clan_rank no debe ser feature"
    assert "previous_clan_rank" not in final.columns, "previous_clan_rank debe excluirse"
    assert "role" not in final.columns, "role no debe aparecer como feature"

    return final


def build_clan_rank_dataset(
    processed_dir: Path = PROCESSED_DIR,
    features_dir: Path = FEATURES_DIR,
    datasets_dir: Path = DATASETS_DIR,
    include_trophies: bool = True,
) -> pd.DataFrame:
    """
    Función principal que carga datos, construye una variante del dataset y la guarda.

    La variante se decide mediante include_trophies:
      - True  -> data/datasets/clan_rank_regression_with_trophies.parquet
      - False -> data/datasets/clan_rank_regression_without_trophies.parquet
    """
    clan_members_df, clans_df, player_features_df = load_inputs(
        processed_dir, features_dir
    )

    final = build_clan_rank_features(
        clan_members_df,
        player_features_df,
        clans_df,
        include_trophies=include_trophies,
    )

    datasets_dir.mkdir(parents=True, exist_ok=True)

    output_filename = (
        "clan_rank_regression_with_trophies.parquet"
        if include_trophies
        else "clan_rank_regression_without_trophies.parquet"
    )

    final.to_parquet(
        datasets_dir / output_filename,
        index=False,
    )
    return final


def build_clan_rank_dataset_variants(
    processed_dir: Path = PROCESSED_DIR,
    features_dir: Path = FEATURES_DIR,
    datasets_dir: Path = DATASETS_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Genera las dos variantes solicitadas:
      1. clan_rank_regression_with_trophies.parquet
      2. clan_rank_regression_without_trophies.parquet
    """
    clan_members_df, clans_df, player_features_df = load_inputs(
        processed_dir, features_dir
    )

    with_trophies = build_clan_rank_features(
        clan_members_df,
        player_features_df,
        clans_df,
        include_trophies=True,
    )
    without_trophies = build_clan_rank_features(
        clan_members_df,
        player_features_df,
        clans_df,
        include_trophies=False,
    )

    datasets_dir.mkdir(parents=True, exist_ok=True)

    with_trophies.to_parquet(
        datasets_dir / "clan_rank_regression_with_trophies.parquet",
        index=False,
    )
    without_trophies.to_parquet(
        datasets_dir / "clan_rank_regression_without_trophies.parquet",
        index=False,
    )

    return with_trophies, without_trophies


if __name__ == "__main__":
    build_clan_rank_dataset_variants()

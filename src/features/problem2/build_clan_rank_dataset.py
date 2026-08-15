from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")
FEATURES_DIR = Path("data/features")
DATASETS_DIR = Path("data/datasets")

# Columnas que NO pueden utilizarse como features del Problema 2
FORBIDDEN_FEATURES = {"clan_rank", "previous_clan_rank"}


def load_inputs(
    processed_dir: Path = PROCESSED_DIR,
    features_dir: Path = FEATURES_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga las tablas compactas necesarias y las player features ya calculadas.
    """
    clan_members = pd.read_parquet(processed_dir / "clan_members.parquet")
    clans = pd.read_parquet(processed_dir / "clans.parquet")
    player_features = pd.read_parquet(features_dir / "player_features.parquet")
    return clan_members, clans, player_features


def select_clan_context_features(clans_df: pd.DataFrame) -> pd.DataFrame:
    """
    Selecciona únicamente características estructurales razonables del clan.
    """
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
    Calcula caracteristicas relativas dentro del clan:
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
        # percentil dentro del clan
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

    # Eliminar columnas duplicadas provenientes de player_features
    # que ya existen en clan_members (nos quedamos con la versión de clan_members)
    duplicate_cols = [
        c for c in merged.columns
        if c.endswith("_player") and c[:-7] in merged.columns
    ]
    if duplicate_cols:
        merged = merged.drop(columns=duplicate_cols)

    return merged


def build_clan_rank_features(
    clan_members_df: pd.DataFrame,
    player_features_df: pd.DataFrame,
    clans_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye el dataset para el Problema 2: target = clan_rank.

    Una fila por (clan_tag, player_tag).
    Las filas cuyos jugadores no tienen player_features son excluidas.
    """
    # 1) Joins
    merged = _merge_preserving_clan_values(clan_members_df, player_features_df)

    # 2) Características relativas al clan
    #    Usamos columnas de clan_members y de player_features que estén en merged
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

    # 3) Unir con características estructurales del clan
    clan_ctx = select_clan_context_features(clans_df)
    output = merged.merge(clan_ctx, on="clan_tag", how="left")

    # 4) Añadir las features relativas
    output = output.merge(
        rel_features,
        on=["player_tag", "clan_tag"],
        how="left",
    )

    # 5) Eliminar columnas prohibidas
    for col in FORBIDDEN_FEATURES:
        if col in output.columns and col != "clan_rank":
            output = output.drop(columns=col)

    # 6) Reordenar: identificadores, features, target
    # Obtenemos las columnas de features excluyendo target e ids
    target_col = "clan_rank"
    id_cols = ["player_tag", "clan_tag"]
    feature_cols = [
        c for c in output.columns
        if c not in id_cols and c != target_col
    ]

    final = output[id_cols + feature_cols + [target_col]].copy()

    # 7) Verificar que no haya clan_rank en features
    assert target_col not in feature_cols, "clan_rank no debe ser feature"
    assert "previous_clan_rank" not in final.columns, "previous_clan_rank debe excluirse"

    return final


def build_clan_rank_dataset(
    processed_dir: Path = PROCESSED_DIR,
    features_dir: Path = FEATURES_DIR,
    datasets_dir: Path = DATASETS_DIR,
) -> pd.DataFrame:
    """
    Función principal que carga datos, construye el dataset y lo guarda.
    """
    clan_members_df, clans_df, player_features_df = load_inputs(
        processed_dir, features_dir
    )

    final = build_clan_rank_features(
        clan_members_df,
        player_features_df,
        clans_df,
    )

    datasets_dir.mkdir(parents=True, exist_ok=True)
    final.to_parquet(
        datasets_dir / "clan_rank_regression.parquet",
        index=False,
    )
    return final


if __name__ == "__main__":
    build_clan_rank_dataset()

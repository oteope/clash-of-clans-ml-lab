from pathlib import Path
from typing import Dict

import pandas as pd

from src.features.player_features import (
    build_player_features,
    build_player_features_from_files,
)
from src.features.player_clan_features import compute_clan_relative_features

PROCESSED_DIR = Path("data/processed")
FEATURES_DIR = Path("data/features")
DATASETS_DIR = Path("data/datasets")


def load_small_tables() -> Dict[str, pd.DataFrame]:
    """Carga las tablas compactas que caben en memoria."""
    return {
        "players": pd.read_parquet(PROCESSED_DIR / "players.parquet"),
        "clans": pd.read_parquet(PROCESSED_DIR / "clans.parquet"),
        "clan_members": pd.read_parquet(PROCESSED_DIR / "clan_members.parquet"),
    }


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
    # Mantener solo las columnas que existen
    cols = [c for c in desired_cols if c in clans_df.columns]
    return clans_df[cols].copy()


def build_all_features(
    clan_members_df: pd.DataFrame,
    players_df: pd.DataFrame,
    troops_df: pd.DataFrame,
    heroes_df: pd.DataFrame,
    spells_df: pd.DataFrame,
    equipment_df: pd.DataFrame,
    achievements_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construye player_features y player_clan_features.

    Esta versión recibe DataFrames completos y se mantiene para tests.
    """
    pf = build_player_features(
        players_df, troops_df, heroes_df, spells_df, equipment_df, achievements_df
    )
    pcf = compute_clan_relative_features(clan_members_df, pf)
    return pf, pcf


def assemble_role_dataset(
    pf: pd.DataFrame,
    pcf: pd.DataFrame,
    clan_members_df: pd.DataFrame,
    clans_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Une player_features, player_clan_features y contexto de clan,
    y añade el target `role` preservando la relación player-clan.

    El dataset resultante tiene una fila por (clan_tag, player_tag).
    """
    # Contexto de clan
    clan_ctx = select_clan_context_features(clans_df)

    # Unión de player_clan_features con contexto de clan
    merged = pcf.merge(clan_ctx, on="clan_tag", how="left")

    # Añadir todas las player_features
    merged = merged.merge(pf, on="player_tag", how="left", suffixes=("", "_player"))

    # Target role desde clan_members (sin duplicados por seguridad)
    role_df = clan_members_df[["clan_tag", "player_tag", "role"]].drop_duplicates()
    final = merged.merge(role_df, on=["clan_tag", "player_tag"], how="left")

    return final


def main() -> None:
    """Pipeline completo para generación de features del Problema 1."""
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    # Cargar solo tablas compactas
    small_tables = load_small_tables()
    players_df = small_tables["players"]
    clans_df = small_tables["clans"]
    clan_members_df = small_tables["clan_members"]

    # Calcular player features mediante procesamiento por batches
    pf = build_player_features_from_files(
        PROCESSED_DIR, batch_size=100_000
    )

    # Calcular player-clan features usando clan_members y player_features
    pcf = compute_clan_relative_features(clan_members_df, pf)

    # Guardar features intermedias
    pf.to_parquet(FEATURES_DIR / "player_features.parquet", index=False)
    pcf.to_parquet(FEATURES_DIR / "player_clan_features.parquet", index=False)

    # Dataset final
    role_dataset = assemble_role_dataset(
        pf=pf,
        pcf=pcf,
        clan_members_df=clan_members_df,
        clans_df=clans_df,
    )
    role_dataset.to_parquet(
        DATASETS_DIR / "role_classification.parquet",
        index=False,
    )


if __name__ == "__main__":
    main()

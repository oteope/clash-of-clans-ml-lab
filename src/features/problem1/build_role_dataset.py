from pathlib import Path
from typing import Dict

import pandas as pd

from src.features.problem1.player_features import (
    build_player_features,
    build_player_features_from_files,
)
from src.features.problem1.player_clan_features import compute_clan_relative_features

PROCESSED_DIR = Path("data/processed")
FEATURES_DIR = Path("data/features")
DATASETS_DIR = Path("data/datasets")


def load_small_tables() -> Dict[str, pd.DataFrame]:
    """Load the compact tables that fit in memory."""
    return {
        "players": pd.read_parquet(PROCESSED_DIR / "players.parquet"),
        "clans": pd.read_parquet(PROCESSED_DIR / "clans.parquet"),
        "clan_members": pd.read_parquet(PROCESSED_DIR / "clan_members.parquet"),
    }


def select_clan_context_features(clans_df: pd.DataFrame) -> pd.DataFrame:
    """
    Select only reasonable clan structural features.
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
    # Keep only existing columns
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
    Build player_features and player_clan_features.

    This version accepts complete DataFrames and is retained for tests.
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
    Join player_features, player_clan_features, and clan context,
    and add the `role` target while preserving the player-clan relationship.

    The resulting dataset has one row per (clan_tag, player_tag).
    """
    # Clan context
    clan_ctx = select_clan_context_features(clans_df)

    # Join player_clan_features with clan context
    merged = pcf.merge(clan_ctx, on="clan_tag", how="left")

    # Add all player_features
    merged = merged.merge(pf, on="player_tag", how="left", suffixes=("", "_player"))

    # Role target from clan_members (without duplicates for safety)
    role_df = clan_members_df[["clan_tag", "player_tag", "role"]].drop_duplicates()
    final = merged.merge(role_df, on=["clan_tag", "player_tag"], how="left")

    return final


def main() -> None:
    """Complete pipeline for Problem 1 feature generation."""
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    # Load only compact tables
    small_tables = load_small_tables()
    players_df = small_tables["players"]
    clans_df = small_tables["clans"]
    clan_members_df = small_tables["clan_members"]

    # Calculate player features through batch processing
    pf = build_player_features_from_files(
        PROCESSED_DIR, batch_size=100_000
    )

    # Calculate player-clan features using clan_members and player_features
    pcf = compute_clan_relative_features(clan_members_df, pf)

    # Save intermediate features
    pf.to_parquet(FEATURES_DIR / "player_features.parquet", index=False)
    pcf.to_parquet(FEATURES_DIR / "player_clan_features.parquet", index=False)

    # Final dataset
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

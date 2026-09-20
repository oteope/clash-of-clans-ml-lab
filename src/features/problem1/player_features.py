from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

FEATURE_BATCH_SIZE = 100_000


def aggregate_progression_df(
    prog_df: pd.DataFrame,
    level_col: str = "level",
    max_col: str = "max_level",
    prefix: str = "",
) -> pd.DataFrame:
    """
    Aggregate a progression table (troops, heroes, spells, equipment)
    by player.

    Return a DataFrame indexed by player_tag with:
      - {prefix}_count
      - {prefix}_mean_level
      - {prefix}_mean_completion_ratio
    """
    if prog_df.empty:
        return pd.DataFrame(
            columns=[
                f"{prefix}_count",
                f"{prefix}_mean_level",
                f"{prefix}_mean_completion_ratio",
            ],
            index=pd.Index([], name="player_tag"),
        )

    df = prog_df.copy()
    df["_completion_ratio"] = df[level_col] / df[max_col].replace(0, np.nan)

    grouped = df.groupby("player_tag")
    agg = grouped.agg(
        **{
            f"{prefix}_count": (level_col, "size"),
            f"{prefix}_mean_level": (level_col, "mean"),
            f"{prefix}_mean_completion_ratio": ("_completion_ratio", "mean"),
        }
    )
    return agg


def aggregate_achievements(ach_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the achievements table by player.

    Return a DataFrame indexed by player_tag with:
      - achievement_count
      - achievement_completion_ratio
    """
    if ach_df.empty:
        return pd.DataFrame(
            columns=["achievement_count", "achievement_completion_ratio"],
            index=pd.Index([], name="player_tag"),
        )

    df = ach_df.copy()
    df["_completion_ratio"] = df["value"] / df["target"].replace(0, np.nan)

    grouped = df.groupby("player_tag")
    agg = grouped.agg(
        achievement_count=("value", "size"),
        achievement_completion_ratio=("_completion_ratio", "mean"),
    )
    return agg


def _chunk_progression_agg(
    df: pd.DataFrame, level_col: str, max_col: str
) -> pd.DataFrame:
    """Return a partial aggregation for a progression-table chunk."""
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["_completion_ratio"] = df[level_col] / df[max_col].replace(0, np.nan)

    agg = df.groupby("player_tag").agg(
        _count=("player_tag", "size"),
        _sum_level=(level_col, "sum"),
        _sum_completion_ratio=("_completion_ratio", "sum"),
        _count_completion_ratio=("_completion_ratio", "count"),
    )
    return agg


def _chunk_achievements_agg(df: pd.DataFrame) -> pd.DataFrame:
    """Return a partial aggregation for an achievements-table chunk."""
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["_completion_ratio"] = df["value"] / df["target"].replace(0, np.nan)

    agg = df.groupby("player_tag").agg(
        _count=("value", "size"),
        _sum_completion_ratio=("_completion_ratio", "sum"),
        _count_completion_ratio=("_completion_ratio", "count"),
    )
    return agg


def _stream_groupby(
    parquet_path: Path, batch_size: int, chunk_agg_fn
) -> pd.DataFrame:
    """
    Read a Parquet file in batches and accumulate aggregations by player_tag.
    """
    parquet_file = pq.ParquetFile(parquet_path)
    acc = pd.DataFrame()

    for batch in parquet_file.iter_batches(batch_size=batch_size):
        df = batch.to_pandas()
        chunk_agg = chunk_agg_fn(df)
        if chunk_agg.empty:
            continue
        acc = pd.concat([acc, chunk_agg])
        acc = acc.groupby("player_tag").sum()

    return acc


def _finalize_progression(acc: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Convert a progression accumulator into final columns."""
    if acc.empty:
        return pd.DataFrame(
            columns=[
                f"{prefix}_count",
                f"{prefix}_mean_level",
                f"{prefix}_mean_completion_ratio",
            ],
            index=pd.Index([], name="player_tag"),
        )

    final = pd.DataFrame(index=acc.index)
    final[f"{prefix}_count"] = acc["_count"]
    final[f"{prefix}_mean_level"] = acc["_sum_level"] / acc["_count"].replace(0, np.nan)
    final[f"{prefix}_mean_completion_ratio"] = (
        acc["_sum_completion_ratio"] / acc["_count_completion_ratio"].replace(0, np.nan)
    )
    return final


def _finalize_achievements(acc: pd.DataFrame) -> pd.DataFrame:
    """Convert an achievements accumulator into final columns."""
    if acc.empty:
        return pd.DataFrame(
            columns=["achievement_count", "achievement_completion_ratio"],
            index=pd.Index([], name="player_tag"),
        )

    final = pd.DataFrame(index=acc.index)
    final["achievement_count"] = acc["_count"]
    final["achievement_completion_ratio"] = (
        acc["_sum_completion_ratio"] / acc["_count_completion_ratio"].replace(0, np.nan)
    )
    return final


def build_player_features(
    players_df: pd.DataFrame,
    troops_df: pd.DataFrame,
    heroes_df: pd.DataFrame,
    spells_df: pd.DataFrame,
    equipment_df: pd.DataFrame,
    achievements_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build player-level features.

    One row per player_tag.
    """
    # Base columns directly from players.parquet
    base_cols = [
        "player_tag",
        "town_hall_level",
        "exp_level",
        "trophies",
        "best_trophies",
        "war_stars",
        "attack_wins",
        "defense_wins",
        "builder_hall_level",
        "builder_base_trophies",
        "best_builder_base_trophies",
        "donations",
        "donations_received",
        "clan_capital_contributions",
    ]
    base = players_df[base_cols].copy()

    # Features derived from the base profile
    base["donation_balance"] = base["donations"] - base["donations_received"]
    base["donation_ratio"] = (
        base["donations"] / base["donations_received"].replace(0, np.nan)
    )
    base["combat_activity_total"] = base["attack_wins"] + base["defense_wins"]
    base["progression_ratio_trophies"] = (
        base["trophies"] / base["best_trophies"].replace(0, np.nan)
    )
    base["builder_progression_ratio"] = (
        base["builder_base_trophies"]
        / base["best_builder_base_trophies"].replace(0, np.nan)
    )

    # Progression-table aggregations
    troops_agg = aggregate_progression_df(troops_df, prefix="troop")
    heroes_agg = aggregate_progression_df(heroes_df, prefix="hero")
    spells_agg = aggregate_progression_df(spells_df, prefix="spell")
    equipment_agg = aggregate_progression_df(equipment_df, prefix="equipment")
    achievements_agg = aggregate_achievements(achievements_df)

    # Join all sources
    result = base.merge(troops_agg, left_on="player_tag", right_index=True, how="left")
    result = result.merge(heroes_agg, left_on="player_tag", right_index=True, how="left")
    result = result.merge(spells_agg, left_on="player_tag", right_index=True, how="left")
    result = result.merge(
        equipment_agg, left_on="player_tag", right_index=True, how="left"
    )
    result = result.merge(
        achievements_agg, left_on="player_tag", right_index=True, how="left"
    )

    # Fill null numeric values with 0
    for col in result.columns:
        if col == "player_tag":
            continue
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0)

    return result


def build_player_features_from_files(
    processed_dir: Path, batch_size: int = FEATURE_BATCH_SIZE
) -> pd.DataFrame:
    """
    Build player features without loading entire large tables.

    Read players.parquet and progression tables in batches.
    """
    # Load compact player table
    players_df = pd.read_parquet(processed_dir / "players.parquet")

    # Batch aggregations for each large table
    troops_acc = _stream_groupby(
        processed_dir / "player_troops.parquet",
        batch_size,
        lambda df: _chunk_progression_agg(df, "level", "max_level"),
    )
    heroes_acc = _stream_groupby(
        processed_dir / "player_heroes.parquet",
        batch_size,
        lambda df: _chunk_progression_agg(df, "level", "max_level"),
    )
    spells_acc = _stream_groupby(
        processed_dir / "player_spells.parquet",
        batch_size,
        lambda df: _chunk_progression_agg(df, "level", "max_level"),
    )
    equipment_acc = _stream_groupby(
        processed_dir / "player_hero_equipment.parquet",
        batch_size,
        lambda df: _chunk_progression_agg(df, "level", "max_level"),
    )
    achievements_acc = _stream_groupby(
        processed_dir / "player_achievements.parquet",
        batch_size,
        lambda df: _chunk_achievements_agg(df),
    )

    troops_agg = _finalize_progression(troops_acc, "troop")
    heroes_agg = _finalize_progression(heroes_acc, "hero")
    spells_agg = _finalize_progression(spells_acc, "spell")
    equipment_agg = _finalize_progression(equipment_acc, "equipment")
    achievements_agg = _finalize_achievements(achievements_acc)

    # Build base features
    base_cols = [
        "player_tag",
        "town_hall_level",
        "exp_level",
        "trophies",
        "best_trophies",
        "war_stars",
        "attack_wins",
        "defense_wins",
        "builder_hall_level",
        "builder_base_trophies",
        "best_builder_base_trophies",
        "donations",
        "donations_received",
        "clan_capital_contributions",
    ]
    base = players_df[base_cols].copy()

    base["donation_balance"] = base["donations"] - base["donations_received"]
    base["donation_ratio"] = (
        base["donations"] / base["donations_received"].replace(0, np.nan)
    )
    base["combat_activity_total"] = base["attack_wins"] + base["defense_wins"]
    base["progression_ratio_trophies"] = (
        base["trophies"] / base["best_trophies"].replace(0, np.nan)
    )
    base["builder_progression_ratio"] = (
        base["builder_base_trophies"]
        / base["best_builder_base_trophies"].replace(0, np.nan)
    )

    # Join aggregations
    result = base.merge(troops_agg, left_on="player_tag", right_index=True, how="left")
    result = result.merge(heroes_agg, left_on="player_tag", right_index=True, how="left")
    result = result.merge(spells_agg, left_on="player_tag", right_index=True, how="left")
    result = result.merge(
        equipment_agg, left_on="player_tag", right_index=True, how="left"
    )
    result = result.merge(
        achievements_agg, left_on="player_tag", right_index=True, how="left"
    )

    # Fill null numeric values with 0
    for col in result.columns:
        if col == "player_tag":
            continue
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0)

    return result

from typing import Any

import numpy as np
import pandas as pd


def aggregate_progression_df(
    prog_df: pd.DataFrame,
    level_col: str = "level",
    max_col: str = "max_level",
    prefix: str = "",
) -> pd.DataFrame:
    """
    Agrega una tabla de progresión (tropas, héroes, hechizos, equipo)
    por jugador.

    Retorna un DataFrame indexado por player_tag con:
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
    Agrega la tabla de logros por jugador.

    Retorna un DataFrame indexado por player_tag con:
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


def build_player_features(
    players_df: pd.DataFrame,
    troops_df: pd.DataFrame,
    heroes_df: pd.DataFrame,
    spells_df: pd.DataFrame,
    equipment_df: pd.DataFrame,
    achievements_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye las features a nivel de jugador.

    Una fila por player_tag.
    """
    # Columnas base directamente desde players.parquet
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

    # Features derivadas del perfil base
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

    # Agregaciones de tablas de progresión
    troops_agg = aggregate_progression_df(troops_df, prefix="troop")
    heroes_agg = aggregate_progression_df(heroes_df, prefix="hero")
    spells_agg = aggregate_progression_df(spells_df, prefix="spell")
    equipment_agg = aggregate_progression_df(equipment_df, prefix="equipment")
    achievements_agg = aggregate_achievements(achievements_df)

    # Unir todas las fuentes
    result = base.merge(troops_agg, left_on="player_tag", right_index=True, how="left")
    result = result.merge(heroes_agg, left_on="player_tag", right_index=True, how="left")
    result = result.merge(spells_agg, left_on="player_tag", right_index=True, how="left")
    result = result.merge(
        equipment_agg, left_on="player_tag", right_index=True, how="left"
    )
    result = result.merge(
        achievements_agg, left_on="player_tag", right_index=True, how="left"
    )

    # Rellenar valores nulos numéricos con 0
    for col in result.columns:
        if col != "player_tag" and np.issubdtype(result[col].dtype, np.number):
            result[col] = result[col].fillna(0)

    return result

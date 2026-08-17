import numpy as np
import pandas as pd


def compute_clan_relative_features(
    clan_members_df: pd.DataFrame,
    player_features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula features relativas jugador-clan.

    No utiliza clan_rank ni previous_clan_rank.
    Retorna un DataFrame con una fila por relación (clan_tag, player_tag)
    y columnas de diferencias/ratios/percentiles dentro del clan.
    """
    # Columnas provenientes de clan_members
    member_base_cols = [
        "clan_tag",
        "player_tag",
        "trophies",
        "exp_level",
        "town_hall_level",
        "donations",
        "donations_received",
        "capital_contributions",
    ]
    base = clan_members_df[member_base_cols].copy()

    # Columnas adicionales desde player_features
    extra_cols = ["player_tag", "war_stars", "attack_wins", "defense_wins"]
    extra = player_features_df[extra_cols].copy()

    merged = base.merge(extra, on="player_tag", how="left")

    # Rellenar con 0 columnas numéricas que no existían en player_features
    for col in ["war_stars", "attack_wins", "defense_wins"]:
        merged[col] = merged[col].fillna(0)

    # Features para las que calcularemos estadísticas relativas
    relative_cols = [
        "trophies",
        "exp_level",
        "town_hall_level",
        "war_stars",
        "capital_contributions",
        "donations",
        "donations_received",
        "attack_wins",
        "defense_wins",
    ]

    # Medias por clan
    clan_means = merged.groupby("clan_tag")[relative_cols].transform("mean")
    clan_means = clan_means.rename(columns=lambda c: f"clan_mean_{c}")

    # Inicializar resultado con identificadores
    result = merged[["player_tag", "clan_tag"]].copy()

    # Diferencias y ratios
    for col in relative_cols:
        mean_col = f"clan_mean_{col}"
        result[f"{col}_diff_from_clan_mean"] = merged[col] - clan_means[mean_col]
        result[f"{col}_ratio_to_clan_mean"] = (
            merged[col] / clan_means[mean_col].replace(0, np.nan)
        )

    # Percentiles dentro del clan para algunas features relevantes
    percentile_cols = ["trophies", "exp_level", "war_stars"]
    for col in percentile_cols:
        merged[f"{col}_clan_pct"] = merged.groupby("clan_tag")[col].rank(
            method="average", pct=True
        )

    percentile_output = merged[
        [f"{col}_clan_pct" for col in percentile_cols]
    ]
    result = pd.concat([result, percentile_output], axis=1)

    # Limpieza de NaN/Inf
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.fillna(0)

    return result

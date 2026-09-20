import numpy as np
import pandas as pd


def compute_clan_relative_features(
    clan_members_df: pd.DataFrame,
    player_features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute player-clan relative features.

    Does not use clan_rank or previous_clan_rank.
    Return a DataFrame with one row per (clan_tag, player_tag) relationship
    and columns for within-clan differences, ratios, and percentiles.
    """
    # Columns from clan_members
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

    # Additional columns from player_features
    extra_cols = ["player_tag", "war_stars", "attack_wins", "defense_wins"]
    extra = player_features_df[extra_cols].copy()

    merged = base.merge(extra, on="player_tag", how="left")

    # Fill with 0 numeric columns that did not exist in player_features
    for col in ["war_stars", "attack_wins", "defense_wins"]:
        merged[col] = merged[col].fillna(0)

    # Features for which relative statistics will be calculated
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

    # Means by clan
    clan_means = merged.groupby("clan_tag")[relative_cols].transform("mean")
    clan_means = clan_means.rename(columns=lambda c: f"clan_mean_{c}")

    # Initialize result with identifiers
    result = merged[["player_tag", "clan_tag"]].copy()

    # Differences and ratios
    for col in relative_cols:
        mean_col = f"clan_mean_{col}"
        result[f"{col}_diff_from_clan_mean"] = merged[col] - clan_means[mean_col]
        result[f"{col}_ratio_to_clan_mean"] = (
            merged[col] / clan_means[mean_col].replace(0, np.nan)
        )

    # Within-clan percentiles for some relevant features
    percentile_cols = ["trophies", "exp_level", "war_stars"]
    for col in percentile_cols:
        merged[f"{col}_clan_pct"] = merged.groupby("clan_tag")[col].rank(
            method="average", pct=True
        )

    percentile_output = merged[
        [f"{col}_clan_pct" for col in percentile_cols]
    ]
    result = pd.concat([result, percentile_output], axis=1)

    # Clean NaN/Inf values
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.fillna(0)

    return result

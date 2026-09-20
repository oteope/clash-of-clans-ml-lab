from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
PROCESSED_DIR_DEFAULT = Path("data/processed")
FEATURES_DIR_DEFAULT = Path("data/features")
DATASETS_DIR_DEFAULT = Path("data/datasets")

# ---------------------------------------------------------------------------
# Minimum war-history threshold
#
# This value is set after inspecting the distribution of:
#   war_total = war_wins + war_losses + war_ties
#
# To avoid making this decision blindly, the
# ``_analyze_war_total_distribution`` function is provided. It prints the
# distribution by ranges and makes it possible to justify the threshold with real data.
#
# With the available data, the minimum of 5 wars discards anecdotal clans
# (0–1 wars) and reduces the risk of unstable rates due to insufficient
# history, without removing an excessive fraction of clans.
# ---------------------------------------------------------------------------
MIN_WAR_HISTORY_DEFAULT = 5

# ---------------------------------------------------------------------------
# War-related variables that must NOT be used as features.
# ---------------------------------------------------------------------------
EXCLUDED_WAR_FEATURES = {
    "war_wins",
    "war_losses",
    "war_ties",
    "war_win_streak",
    "war_points",
}

# Direct derivatives of the preceding variables
DERIVED_WAR_FEATURES = {
    "total_wars",
    "win_rate",
    "loss_rate",
    "tie_rate",
    "war_success_rate",
}

# ---------------------------------------------------------------------------
# Explicit whitelist of clan structural features.
# Any clans.parquet column not in this list will not be used as a feature,
# even if it is not a war variable.
# ---------------------------------------------------------------------------
CLAN_STRUCTURAL_FEATURES = [
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
    "location_id",
    "location_name",
]


def _analyze_war_total_distribution(clans_df: pd.DataFrame) -> pd.DataFrame:
    """
    Print and return the distribution of ``war_total`` by ranges.

    This function makes it possible to inspect the population of clans with
    history 0, 1, 2, ... and choose a reasonable minimum war-history threshold.
    """
    if "war_total" not in clans_df.columns:
        df = clans_df.copy()
        df["war_total"] = (
            df["war_wins"].astype(float)
            + df["war_losses"].astype(float)
            + df["war_ties"].astype(float)
        )
    else:
        df = clans_df

    bins = [0, 1, 2, 3, 4, 5, 10, 20, 50, 100, np.inf]
    labels = ["0", "1", "2", "3", "4", "5-9", "10-19", "20-49", "50-99", "100+"]
    war_total = df["war_total"]
    hist = pd.cut(war_total, bins=bins, labels=labels, right=False)
    dist = hist.value_counts().sort_index()

    summary = pd.DataFrame(
        {
            "war_total_bin": dist.index,
            "count": dist.values,
            "percentage": dist.values / len(df) * 100,
        }
    )

    print("Distribución de war_total (wins + losses + ties):")
    print(summary.to_string(index=False))
    print(
        f"Clanes con >= {MIN_WAR_HISTORY_DEFAULT} guerras: "
        f"{war_total[war_total >= MIN_WAR_HISTORY_DEFAULT].count()} "
        f"({war_total[war_total >= MIN_WAR_HISTORY_DEFAULT].count() / len(df) * 100:.1f}%)"
    )

    return summary


def _read_parquet(path: Path) -> pd.DataFrame:
    """Read a Parquet file and raise a clear error if it does not exist."""
    if not path.exists():
        raise FileNotFoundError(f"Parquet no encontrado: {path}")
    return pd.read_parquet(path)


def _load_clan_members(processed_dir: Path) -> pd.DataFrame:
    """
    Load clan_members.parquet and retain only player_tag and clan_tag.

    Deduplicate by the (clan_tag, player_tag) pair, not by player_tag.
    A player can appear in multiple clans; all valid relationships must be
    retained.
    """
    members = _read_parquet(processed_dir / "clan_members.parquet")
    required = {"player_tag", "clan_tag"}
    if not required.issubset(members.columns):
        raise ValueError(
            "clan_members.parquet debe contener player_tag y clan_tag"
        )

    members = members[["player_tag", "clan_tag"]].copy()

    # Remove only exact duplicates of the clan-player relationship.
    members = members.drop_duplicates(
        subset=["clan_tag", "player_tag"], keep="first"
    )

    # Defensive validation: there can be no more than one row per (clan_tag, player_tag)
    if members.duplicated(subset=["clan_tag", "player_tag"]).any():
        raise ValueError(
            "clan_members.parquet contiene relaciones duplicadas "
            "(clan_tag, player_tag)"
        )

    return members


def _load_player_features(features_dir: Path) -> pd.DataFrame:
    """
    Load the precomputed player_features.parquet.

    Do not process the large tables of troops, heroes, spells, equipment, or
    achievements again.
    """
    path = features_dir / "player_features.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"player_features.parquet no encontrado en {features_dir}. "
            "Debe generarse previamente con src.features.player_features"
        )

    pf = pd.read_parquet(path)
    if "player_tag" not in pf.columns:
        raise ValueError("player_features.parquet no contiene player_tag")

    # Required check: a single record per player_tag.
    if not pf["player_tag"].is_unique:
        raise ValueError(
            "player_features.parquet contiene player_tag duplicados. "
            "Debe existir exactamente una fila por jugador antes del merge."
        )

    return pf


def _aggregate_clan_player_features(members_features: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate player-level features at the clan level.

    Return a DataFrame with one row per clan_tag.

    IMPORTANT:
    Historical player war statistics, such as war_stars, are not aggregated.
    Only the clan's internal composition / progression / economy is described.
    """
    if members_features.empty:
        return pd.DataFrame()

    df = members_features.copy()
    grouped = df.groupby("clan_tag")

    agg_specs: dict = {}

    def add_mean_median_std(col: str, prefix: str) -> None:
        if col in df.columns:
            agg_specs[f"mean_{prefix}"] = (col, "mean")
            agg_specs[f"median_{prefix}"] = (col, "median")
            agg_specs[f"std_{prefix}"] = (col, "std")

    # --------------------------------------------------------------
    # Basic numeric-column aggregations
    # --------------------------------------------------------------
    add_mean_median_std("town_hall_level", "town_hall_level")
    add_mean_median_std("exp_level", "exp_level")
    add_mean_median_std("trophies", "trophies")
    add_mean_median_std("donations", "donations")
    add_mean_median_std("donations_received", "donations_received")
    add_mean_median_std(
        "clan_capital_contributions", "clan_capital_contributions"
    )

    # --------------------------------------------------------------
    # Means of progression features already calculated at the player level
    # --------------------------------------------------------------
    progression_mean_cols = [
        "troop_mean_level",
        "troop_mean_completion_ratio",
        "hero_mean_level",
        "hero_mean_completion_ratio",
        "spell_mean_level",
        "spell_mean_completion_ratio",
        "equipment_mean_level",
        "equipment_mean_completion_ratio",
        "achievement_completion_ratio",
    ]
    for col in progression_mean_cols:
        if col in df.columns:
            agg_specs[f"mean_{col}"] = (col, "mean")

    result = grouped.agg(**agg_specs)

    # --------------------------------------------------------------
    # Percentages of high TH levels
    # --------------------------------------------------------------
    if "town_hall_level" in df.columns:
        th_extra = grouped.agg(
            th18_percentage=("town_hall_level", lambda s: (s >= 18).mean()),
            th17_plus_percentage=("town_hall_level", lambda s: (s >= 17).mean()),
        )
        result = result.merge(
            th_extra, left_index=True, right_index=True, how="left"
        )

    # --------------------------------------------------------------
    # Sums required for ratios and balances
    # --------------------------------------------------------------
    sum_specs: dict = {"member_count": ("player_tag", "size")}
    if "donations" in df.columns:
        sum_specs["sum_donations"] = ("donations", "sum")
    if "donations_received" in df.columns:
        sum_specs["sum_donations_received"] = ("donations_received", "sum")
    if "clan_capital_contributions" in df.columns:
        sum_specs["sum_clan_capital_contributions"] = (
            "clan_capital_contributions",
            "sum",
        )

    sums = grouped.agg(**sum_specs)

    # Merge sums (including member_count) so they form part of the final
    # aggregated-feature contract.
    result = result.merge(sums, left_index=True, right_index=True, how="left")

    # --------------------------------------------------------------
    # Additional ratios and balances
    # --------------------------------------------------------------
    extra = pd.DataFrame(index=sums.index)

    if "sum_donations" in sums.columns and "sum_donations_received" in sums.columns:
        extra["donation_balance"] = (
            sums["sum_donations"] - sums["sum_donations_received"]
        )
        denom_don = sums["sum_donations_received"].replace(0, np.nan)
        extra["donation_ratio"] = sums["sum_donations"] / denom_don
        extra["donation_rate"] = (
            sums["sum_donations"] / sums["member_count"].replace(0, np.nan)
        )

    if "sum_clan_capital_contributions" in sums.columns:
        extra["capital_contribution_rate"] = (
            sums["sum_clan_capital_contributions"]
            / sums["member_count"].replace(0, np.nan)
        )

    result = result.merge(extra, left_index=True, right_index=True, how="left")
    return result


def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Missing-value policy:
    - numeric → 0
    - boolean → False
    - categorical/object → "unknown"

    Exclude ``clan_tag`` to avoid imputing the identifier.
    """
    def _is_boolean_series(series: pd.Series) -> bool:
        """Detect boolean series, including object-dtype series with True/False/None."""
        if pd.api.types.is_bool_dtype(series):
            return True
        non_null = series.dropna()
        if len(non_null) == 0:
            return False
        return all(isinstance(v, (bool, np.bool_)) for v in non_null)

    out = df.copy()
    for col in out.columns:
        if col == "clan_tag":
            continue
        if _is_boolean_series(out[col]):
            out[col] = out[col].fillna(False).astype(bool)
        elif pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].fillna(0)
        else:
            out[col] = out[col].fillna("unknown")
    return out


def build_war_performance_dataset(
    processed_dir: Path,
    features_dir: Path,
    min_war_total: int = MIN_WAR_HISTORY_DEFAULT,
) -> pd.DataFrame:
    """
    Build the Problem 3 dataset.

    - 1 row = 1 clan.
    - Target: war_success_rate = war_wins / (war_wins + war_losses + war_ties).
    - Only clans with total war history >= ``min_war_total``.
    - Cumulative war features and direct derivatives are excluded.
    - player_features.parquet is reused without iterating over large tables.
    - Historical player war performance (for example, war_stars) is not included
      in composition features.
    """
    clans = _read_parquet(processed_dir / "clans.parquet")

    required_clan_cols = {"clan_tag", "war_wins", "war_losses", "war_ties"}
    missing = required_clan_cols.difference(clans.columns)
    if missing:
        raise ValueError(
            f"clans.parquet no contiene las columnas requeridas: {sorted(missing)}"
        )

    clans = clans.copy()
    clans["war_total"] = (
        clans["war_wins"].astype(float)
        + clans["war_losses"].astype(float)
        + clans["war_ties"].astype(float)
    )

    # ------------------------------------------------------------------
    # Optional: inspect distribution to justify min_war_total.
    # Uncomment to view the summary before setting the threshold.
    # _analyze_war_total_distribution(clans)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Filter clans with insufficient history
    # ------------------------------------------------------------------
    valid_clans = clans[clans["war_total"] >= min_war_total].copy()
    if valid_clans.empty:
        return pd.DataFrame(columns=["clan_tag", "war_success_rate"])

    # ------------------------------------------------------------------
    # Calculate target
    # ------------------------------------------------------------------
    valid_clans["war_success_rate"] = (
        valid_clans["war_wins"].astype(float)
        / valid_clans["war_total"].replace(0, np.nan)
    )
    valid_clans = valid_clans[valid_clans["war_success_rate"].notna()].copy()

    # ------------------------------------------------------------------
    # Separate target and structural features using the whitelist
    # ------------------------------------------------------------------
    target = valid_clans[["clan_tag", "war_success_rate"]].copy()

    available_clan_cols = [
        c for c in CLAN_STRUCTURAL_FEATURES if c in valid_clans.columns
    ]
    clan_features = valid_clans[["clan_tag"] + available_clan_cols].copy()

    # ------------------------------------------------------------------
    # Join clan composition
    # ------------------------------------------------------------------
    members = _load_clan_members(processed_dir)
    player_features = _load_player_features(features_dir)

    members_features = members.merge(player_features, on="player_tag", how="left")
    clan_composition = _aggregate_clan_player_features(members_features)

    result = clan_features.merge(
        clan_composition, left_on="clan_tag", right_index=True, how="left"
    )

    # Join the target
    result = result.merge(target, on="clan_tag", how="left")

    # Ensure one row per clan
    result = result.drop_duplicates(subset="clan_tag", keep="first")

    # Apply missing-value policy
    result = _fill_missing_values(result)

    return result.reset_index(drop=True)


def main() -> None:
    """Generate the final dataset. Use with caution; review the threshold first."""
    processed_dir = PROCESSED_DIR_DEFAULT
    features_dir = FEATURES_DIR_DEFAULT
    datasets_dir = DATASETS_DIR_DEFAULT
    datasets_dir.mkdir(parents=True, exist_ok=True)

    df = build_war_performance_dataset(
        processed_dir,
        features_dir,
        min_war_total=MIN_WAR_HISTORY_DEFAULT,
    )

    out_path = datasets_dir / "clan_war_performance_regression.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Dataset guardado en {out_path} | clanes: {len(df)}")


if __name__ == "__main__":
    main()

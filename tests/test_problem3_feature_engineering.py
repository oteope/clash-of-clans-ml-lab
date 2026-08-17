from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.problem3.build_war_performance_dataset import (
    MIN_WAR_HISTORY_DEFAULT,
    CLAN_STRUCTURAL_FEATURES,
    EXCLUDED_WAR_FEATURES,
    DERIVED_WAR_FEATURES,
    _load_clan_members,
    _load_player_features,
    _aggregate_clan_player_features,
    _fill_missing_values,
    build_war_performance_dataset,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def write_data(tmp_path, clans, clan_members, player_features):
    """Escribe los parquet de entrada y devuelve los directorios."""
    processed_dir = tmp_path / "processed"
    features_dir = tmp_path / "features"
    processed_dir.mkdir(exist_ok=True)
    features_dir.mkdir(exist_ok=True)

    clans.to_parquet(processed_dir / "clans.parquet", index=False)
    clan_members.to_parquet(processed_dir / "clan_members.parquet", index=False)
    player_features.to_parquet(
        features_dir / "player_features.parquet", index=False
    )
    return processed_dir, features_dir


def make_sample_dfs():
    """
    Construye DataFrames de ejemplo con:
    - Clanes con columnas estructurales (y una extra no permitida).
    - clan_members donde el jugador 'P1' pertenece a dos clanes.
    - player_features con columnas numéricas y de progresión,
      incluyendo columnas que NO deben filtrarse como features (war_stars, etc.)
    """
    clans = pd.DataFrame(
        {
            "clan_tag": ["C1", "C2", "C3", "C4", "C5", "C6"],
            "war_wins": [4, 5, 1, 10, 0, 3],
            "war_losses": [1, 1, 1, 2, 0, 1],
            "war_ties": [0, 0, 0, 0, 0, 0],
            "war_win_streak": [1, 2, 0, 3, 0, 1],
            "war_points": [100, 200, 50, 300, 0, 150],
            "clan_level": [5, 6, 3, 10, 2, 4],
            "clan_points": [1000, 2000, 500, 5000, 200, 800],
            "clan_capital_points": [50, 60, 20, 100, 10, 30],
            "members": [45, 48, 30, 50, 10, 25],
            "required_trophies": [0, 0, 0, 0, 0, 0],
            "war_frequency": [1, 2, 1, 2, 1, 2],
            "war_league": [0, 1, 0, 1, 0, 1],
            "capital_league": [0, 1, 0, 1, 0, 1],
            "type": ["open", "invite_only", "open", "closed", "open", "invite_only"],
            "is_family_friendly": [True, False, True, False, True, False],
            "location_id": [32000000] * 6,
            "location_name": ["Spain"] * 6,
            "extra_col": [1, 2, 3, 4, 5, 6],  # columna no permitida
        }
    )

    clan_members = pd.DataFrame(
        {
            "player_tag": ["P1", "P2", "P3", "P4", "P5", "P6", "P1"],
            "clan_tag": ["C1", "C2", "C3", "C4", "C5", "C6", "C2"],
        }
    )

    player_features = pd.DataFrame(
        {
            "player_tag": ["P1", "P2", "P3", "P4", "P5", "P6"],
            "town_hall_level": [16, 15, 14, 13, 12, 11],
            "exp_level": [200, 180, 150, 120, 100, 80],
            "trophies": [5000, 4500, 4000, 3500, 3000, 2500],
            "donations": [100, 200, 300, 400, 500, 600],
            "donations_received": [80, 150, 250, 350, 450, 550],
            "clan_capital_contributions": [1000, 2000, 3000, 4000, 5000, 6000],
            "war_stars": [10, 20, 30, 40, 50, 60],
            "attack_wins": [5, 6, 7, 8, 9, 10],
            "defense_wins": [4, 5, 6, 7, 8, 9],
            "troop_mean_level": [12, 11, 10, 9, 8, 7],
            "troop_mean_completion_ratio": [0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
            "hero_mean_level": [10, 9, 8, 7, 6, 5],
            "hero_mean_completion_ratio": [0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            "spell_mean_level": [8, 7, 6, 5, 4, 3],
            "spell_mean_completion_ratio": [0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
            "equipment_mean_level": [6, 5, 4, 3, 2, 1],
            "equipment_mean_completion_ratio": [0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            "achievement_completion_ratio": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        }
    )

    return clans, clan_members, player_features


def get_feature_columns(df):
    """Devuelve las columnas que no son identificador ni target."""
    return [c for c in df.columns if c not in ("clan_tag", "war_success_rate")]


# ---------------------------------------------------------------------------
# 1. Granularidad y target
# ---------------------------------------------------------------------------
def test_granularity_and_target_calculation(tmp_path):
    clans, clan_members, player_features = make_sample_dfs()
    processed_dir, features_dir = write_data(tmp_path, clans, clan_members, player_features)

    df = build_war_performance_dataset(
        processed_dir, features_dir, min_war_total=MIN_WAR_HISTORY_DEFAULT
    )

    # 1 fila = 1 clan
    assert df["clan_tag"].is_unique
    assert len(df) == df["clan_tag"].nunique()

    # rango del target
    assert (df["war_success_rate"] >= 0).all()
    assert (df["war_success_rate"] <= 1).all()

    # target exacto
    for _, row in df.iterrows():
        tag = row["clan_tag"]
        original = clans.loc[clans["clan_tag"] == tag].iloc[0]
        expected = original["war_wins"] / (
            original["war_wins"] + original["war_losses"] + original["war_ties"]
        )
        assert np.isclose(row["war_success_rate"], expected)


# ---------------------------------------------------------------------------
# 2. Target: no división por cero y rango
# ---------------------------------------------------------------------------
def test_target_no_division_by_zero(tmp_path):
    clans = pd.DataFrame(
        {
            "clan_tag": ["CZERO", "CPOS"],
            "war_wins": [0, 3],
            "war_losses": [0, 1],
            "war_ties": [0, 0],
            # columnas estructurales mínimas
            "clan_level": [1, 2],
            "clan_points": [1, 2],
            "clan_capital_points": [1, 2],
            "members": [1, 2],
            "required_trophies": [0, 0],
            "war_frequency": [0, 0],
            "war_league": [0, 0],
            "capital_league": [0, 0],
            "type": ["open", "open"],
            "is_family_friendly": [True, True],
            "location_id": [1, 1],
            "location_name": ["x", "y"],
        }
    )
    clan_members = pd.DataFrame({"player_tag": ["PZ"], "clan_tag": ["CZERO"]})
    player_features = pd.DataFrame(
        {
            "player_tag": ["PZ"],
            "town_hall_level": [10],
            "exp_level": [100],
            "trophies": [1000],
        }
    )
    processed_dir, features_dir = write_data(tmp_path, clans, clan_members, player_features)

    df = build_war_performance_dataset(processed_dir, features_dir, min_war_total=0)

    # CZERO debe quedar fuera por NaN en target
    assert "CZERO" not in df["clan_tag"].values
    assert "CPOS" in df["clan_tag"].values
    assert (df["war_success_rate"] >= 0).all()
    assert (df["war_success_rate"] <= 1).all()


# ---------------------------------------------------------------------------
# 3. Leakage del clan
# ---------------------------------------------------------------------------
def test_clan_leakage_excluded(tmp_path):
    clans, clan_members, player_features = make_sample_dfs()
    processed_dir, features_dir = write_data(tmp_path, clans, clan_members, player_features)

    df = build_war_performance_dataset(
        processed_dir, features_dir, min_war_total=MIN_WAR_HISTORY_DEFAULT
    )

    feature_cols = set(get_feature_columns(df))
    assert not (EXCLUDED_WAR_FEATURES & feature_cols)
    assert not (DERIVED_WAR_FEATURES & feature_cols)


# ---------------------------------------------------------------------------
# 4. Leakage del jugador
# ---------------------------------------------------------------------------
def test_player_leakage_excluded(tmp_path):
    clans, clan_members, player_features = make_sample_dfs()
    processed_dir, features_dir = write_data(tmp_path, clans, clan_members, player_features)

    df = build_war_performance_dataset(
        processed_dir, features_dir, min_war_total=MIN_WAR_HISTORY_DEFAULT
    )
    feature_cols = get_feature_columns(df)

    for col in ["war_stars", "attack_wins", "defense_wins"]:
        assert col not in feature_cols
        # no debe haber derivados como mean_war_stars, std_attack_wins, etc.
        assert not any(col in c for c in feature_cols)


# ---------------------------------------------------------------------------
# 5. Join player → clan
# ---------------------------------------------------------------------------
def test_player_in_multiple_clans_preserved(tmp_path):
    clans = pd.DataFrame(
        {
            "clan_tag": ["C1", "C2"],
            "war_wins": [1, 1],
            "war_losses": [0, 0],
            "war_ties": [0, 0],
            "clan_level": [5, 6],
            "clan_points": [1000, 2000],
            "clan_capital_points": [50, 60],
            "members": [1, 1],
            "required_trophies": [0, 0],
            "war_frequency": [1, 1],
            "war_league": [0, 0],
            "capital_league": [0, 0],
            "type": ["open", "open"],
            "is_family_friendly": [True, True],
            "location_id": [1, 1],
            "location_name": ["a", "b"],
        }
    )
    clan_members = pd.DataFrame(
        {"player_tag": ["P1", "P1"], "clan_tag": ["C1", "C2"]}
    )
    player_features = pd.DataFrame(
        {
            "player_tag": ["P1"],
            "town_hall_level": [15],
            "exp_level": [100],
            "trophies": [5000],
        }
    )
    processed_dir, features_dir = write_data(tmp_path, clans, clan_members, player_features)

    df = build_war_performance_dataset(processed_dir, features_dir, min_war_total=0)

    assert set(df["clan_tag"]) == {"C1", "C2"}
    # El mismo jugador contribuye a ambos clanes (member_count = 1 para cada uno)
    assert df.loc[df["clan_tag"] == "C1", "member_count"].iloc[0] == 1
    assert df.loc[df["clan_tag"] == "C2", "member_count"].iloc[0] == 1
    # La feature del jugador se asigna a ambos clanes
    assert df.loc[df["clan_tag"] == "C1", "mean_town_hall_level"].iloc[0] == 15
    assert df.loc[df["clan_tag"] == "C2", "mean_town_hall_level"].iloc[0] == 15


# ---------------------------------------------------------------------------
# 6. Player features: unicidad y agregación
# ---------------------------------------------------------------------------
def test_player_features_duplicate_raise_error(tmp_path):
    clans = pd.DataFrame(
        {
            "clan_tag": ["C1"],
            "war_wins": [1],
            "war_losses": [0],
            "war_ties": [0],
            "clan_level": [5],
            "clan_points": [1000],
            "clan_capital_points": [50],
            "members": [1],
            "required_trophies": [0],
            "war_frequency": [1],
            "war_league": [0],
            "capital_league": [0],
            "type": ["open"],
            "is_family_friendly": [True],
            "location_id": [1],
            "location_name": ["a"],
        }
    )
    clan_members = pd.DataFrame({"player_tag": ["P1"], "clan_tag": ["C1"]})
    player_features = pd.DataFrame(
        {
            "player_tag": ["P1", "P1"],  # duplicado
            "town_hall_level": [15, 16],
            "exp_level": [100, 110],
        }
    )
    processed_dir, features_dir = write_data(tmp_path, clans, clan_members, player_features)

    with pytest.raises(ValueError):
        build_war_performance_dataset(processed_dir, features_dir, min_war_total=0)


def test_player_features_aggregation_uses_clan_tag(tmp_path):
    # El mismo jugador en un solo clan, con varias columnas fuente
    clans = pd.DataFrame(
        {
            "clan_tag": ["C1"],
            "war_wins": [1],
            "war_losses": [0],
            "war_ties": [0],
            "clan_level": [5],
            "clan_points": [1000],
            "clan_capital_points": [50],
            "members": [1],
            "required_trophies": [0],
            "war_frequency": [1],
            "war_league": [0],
            "capital_league": [0],
            "type": ["open"],
            "is_family_friendly": [True],
            "location_id": [1],
            "location_name": ["a"],
        }
    )
    clan_members = pd.DataFrame({"player_tag": ["P1"], "clan_tag": ["C1"]})
    player_features = pd.DataFrame(
        {
            "player_tag": ["P1"],
            "town_hall_level": [15],
            "exp_level": [100],
            "trophies": [5000],
            "troop_mean_level": [12],
            "troop_mean_completion_ratio": [0.8],
            "hero_mean_level": [10],
            "hero_mean_completion_ratio": [0.6],
            "spell_mean_level": [8],
            "spell_mean_completion_ratio": [0.7],
            "equipment_mean_level": [6],
            "equipment_mean_completion_ratio": [0.5],
            "achievement_completion_ratio": [0.9],
        }
    )
    processed_dir, features_dir = write_data(tmp_path, clans, clan_members, player_features)

    df = build_war_performance_dataset(processed_dir, features_dir, min_war_total=0)

    # Las agregaciones de progresión deben aparecer
    expected_cols = [
        "mean_town_hall_level",
        "mean_exp_level",
        "mean_trophies",
        "mean_troop_mean_level",
        "mean_troop_mean_completion_ratio",
        "mean_hero_mean_level",
        "mean_hero_mean_completion_ratio",
        "mean_spell_mean_level",
        "mean_spell_mean_completion_ratio",
        "mean_equipment_mean_level",
        "mean_equipment_mean_completion_ratio",
        "mean_achievement_completion_ratio",
    ]
    for col in expected_cols:
        assert col in df.columns


# ---------------------------------------------------------------------------
# 7. Features estructurales
# ---------------------------------------------------------------------------
def test_only_whitelisted_structural_features(tmp_path):
    clans, clan_members, player_features = make_sample_dfs()
    processed_dir, features_dir = write_data(tmp_path, clans, clan_members, player_features)

    df = build_war_performance_dataset(
        processed_dir, features_dir, min_war_total=MIN_WAR_HISTORY_DEFAULT
    )
    feature_cols = get_feature_columns(df)

    # No debe aparecer la columna extra no permitida
    assert "extra_col" not in feature_cols

    # Cualquier columna de origen clan debe estar en la whitelist o ser aggregated de player
    allowed_clan_cols = set(CLAN_STRUCTURAL_FEATURES)
    for col in feature_cols:
        # Las agregaciones de player tendrán prefijos conocidos o serán columnas creadas;
        # en este test, comprobamos que ninguna columna de clan fuera de whitelist
        # aparezca en el resultado.
        if col in clans.columns:
            assert col in allowed_clan_cols, f"Columna de clan no permitida: {col}"


# ---------------------------------------------------------------------------
# 8. Missing data
# ---------------------------------------------------------------------------
def test_fill_missing_values_policy():
    df = pd.DataFrame(
        {
            "clan_tag": ["C1", "C2", "C3"],
            "num_col": [1, None, np.nan],
            "bool_col": [True, None, False],
            "cat_col": ["a", None, "c"],
        }
    )
    filled = _fill_missing_values(df)

    assert filled["num_col"].tolist() == [1, 0, 0]
    assert filled["bool_col"].tolist() == [True, False, False]
    assert filled["cat_col"].tolist() == ["a", "unknown", "c"]
    # clan_tag no se modifica
    assert filled["clan_tag"].tolist() == ["C1", "C2", "C3"]


# ---------------------------------------------------------------------------
# 9. Inmutabilidad
# ---------------------------------------------------------------------------
def test_fill_missing_values_does_not_modify_input():
    df = pd.DataFrame(
        {
            "a": [1, None],
            "b": [True, None],
        }
    )
    original = df.copy()
    _fill_missing_values(df)
    pd.testing.assert_frame_equal(df, original)


def test_aggregate_clan_player_features_does_not_modify_input():
    members_features = pd.DataFrame(
        {
            "player_tag": ["P1", "P2"],
            "clan_tag": ["C1", "C1"],
            "town_hall_level": [15, 16],
            "exp_level": [100, 110],
        }
    )
    original = members_features.copy()
    _aggregate_clan_player_features(members_features)
    pd.testing.assert_frame_equal(members_features, original)


# ---------------------------------------------------------------------------
# 10. Threshold
# ---------------------------------------------------------------------------
def test_min_war_total_threshold(tmp_path):
    clans = pd.DataFrame(
        {
            "clan_tag": ["C0", "C1", "C4", "C5"],
            "war_wins": [0, 1, 2, 3],
            "war_losses": [0, 0, 0, 1],
            "war_ties": [0, 0, 0, 0],
            "clan_level": [1, 1, 1, 1],
            "clan_points": [1, 1, 1, 1],
            "clan_capital_points": [1, 1, 1, 1],
            "members": [1, 1, 1, 1],
            "required_trophies": [0, 0, 0, 0],
            "war_frequency": [1, 1, 1, 1],
            "war_league": [0, 0, 0, 0],
            "capital_league": [0, 0, 0, 0],
            "type": ["open"] * 4,
            "is_family_friendly": [True] * 4,
            "location_id": [1] * 4,
            "location_name": ["x"] * 4,
        }
    )
    clan_members = pd.DataFrame({"player_tag": ["P1"], "clan_tag": ["C5"]})
    player_features = pd.DataFrame(
        {
            "player_tag": ["P1"],
            "town_hall_level": [10],
            "exp_level": [100],
            "trophies": [1000],
        }
    )

    # war_total: C0=0, C1=1, C4=2, C5=4
    processed_dir, features_dir = write_data(tmp_path, clans, clan_members, player_features)

    df = build_war_performance_dataset(processed_dir, features_dir, min_war_total=2)

    expected_tags = ["C4", "C5"]
    assert set(df["clan_tag"]) == set(expected_tags)


# ---------------------------------------------------------------------------
# 11. Reproducibilidad
# ---------------------------------------------------------------------------
def test_reproducibility(tmp_path):
    clans, clan_members, player_features = make_sample_dfs()
    processed_dir, features_dir = write_data(tmp_path, clans, clan_members, player_features)

    df1 = build_war_performance_dataset(
        processed_dir, features_dir, min_war_total=MIN_WAR_HISTORY_DEFAULT
    )
    df2 = build_war_performance_dataset(
        processed_dir, features_dir, min_war_total=MIN_WAR_HISTORY_DEFAULT
    )

    pd.testing.assert_frame_equal(df1, df2)

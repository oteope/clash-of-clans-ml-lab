import numpy as np
import pandas as pd
import pytest

from src.features.problem5.build_player_clustering_dataset import (
    CANDIDATE_FEATURES,
    DERIVED_CANDIDATE_FEATURES,
    build_player_clustering_dataset,
)


ALL_CANDIDATES = list(dict.fromkeys(CANDIDATE_FEATURES + DERIVED_CANDIDATE_FEATURES))


def make_input_df(n_rows: int = 5, include_missing: bool = False, include_duplicates: bool = False) -> pd.DataFrame:
    rows = []
    for i in range(n_rows):
        rows.append(
            {
                "player_tag": f"#PLAYER{i:02d}",
                "clan_tag": f"#CLAN{i:02d}",
                "role": "member",
                "clan_rank": i,
                "previous_clan_rank": i - 1,
                "war_success_rate": 0.5 + 0.1 * i,
                "performance_class": "mid",
                "some_id": f"ID{i}",
                "extra_numeric_feature": 1.0 + i,
                "town_hall_level": 10 + i % 3,
                "builder_hall_level": 5 + i % 2,
                "exp_level": 100 + i * 10,
                "trophies": 1000 + i * 50,
                "best_trophies": 1200 + i * 50,
                "war_stars": 20 + i,
                "attack_wins": 10 + i,
                "defense_wins": 5 + i,
                "donations": 100 + i * 10,
                "donations_received": 80 + i * 10,
                "clan_capital_contributions": 200 + i * 20,
                "troop_count": 10 + i,
                "troop_mean_level": 5.0 + i * 0.1,
                "troop_mean_completion_ratio": 0.6 + i * 0.01,
                "hero_count": 3 + i % 2,
                "hero_mean_level": 20.0 + i * 0.5,
                "hero_mean_completion_ratio": 0.7 + i * 0.01,
                "spell_count": 5 + i % 3,
                "spell_mean_level": 3.0 + i * 0.1,
                "spell_mean_completion_ratio": 0.5 + i * 0.01,
                "equipment_count": 8 + i % 4,
                "equipment_mean_level": 4.0 + i * 0.2,
                "equipment_mean_completion_ratio": 0.4 + i * 0.01,
                "achievement_count": 25 + i,
                "achievement_completion_ratio": 0.5 + i * 0.01,
                "donation_balance": 20 + i,
                "donation_ratio": 1.2 + i * 0.1,
                "combat_activity_total": 30 + i * 5,
                "progression_ratio_trophies": 0.8 + i * 0.01,
                "builder_progression_ratio": 0.6 + i * 0.01,
                "cluster": i,
                "cluster_id": i,
                "profile": "x",
                "class": "A",
                "label": "B",
                "prediction": 1,
            }
        )

    df = pd.DataFrame(rows)

    if include_missing:
        df.loc[0, "donations"] = np.nan
        df.loc[1, "war_stars"] = np.nan
        df.loc[2, "hero_mean_completion_ratio"] = np.nan

    if include_duplicates:
        duplicate = df.iloc[[0]].copy()
        df = pd.concat([df, duplicate], ignore_index=True)

    return df


def build_with_df(tmp_path, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    input_path = tmp_path / "player_features.parquet"
    output_path = tmp_path / "player_clustering.parquet"
    df.to_parquet(input_path, index=False)
    report = build_player_clustering_dataset(input_path, output_path)
    out_df = pd.read_parquet(output_path)
    return out_df, report


def test_one_row_per_player_and_unique(tmp_path):
    df = make_input_df(n_rows=5)
    out_df, _ = build_with_df(tmp_path, df)

    assert len(out_df) == len(df)
    assert out_df["player_tag"].is_unique
    assert not out_df["player_tag"].duplicated().any()


def test_no_target_columns_present(tmp_path):
    df = make_input_df()
    out_df, _ = build_with_df(tmp_path, df)

    forbidden = {
        "role",
        "clan_rank",
        "previous_clan_rank",
        "war_success_rate",
        "performance_class",
    }
    assert set(out_df.columns).isdisjoint(forbidden)


def test_player_tag_identifier_and_clan_tag_excluded(tmp_path):
    df = make_input_df()
    out_df, _ = build_with_df(tmp_path, df)

    assert "player_tag" in out_df.columns
    assert "clan_tag" not in out_df.columns

    X_cols = [col for col in out_df.columns if col != "player_tag"]
    assert "player_tag" not in X_cols
    assert "clan_tag" not in X_cols
    assert "some_id" not in X_cols


def test_expected_features_present_when_available(tmp_path):
    df = make_input_df()
    out_df, _ = build_with_df(tmp_path, df)

    available_expected = [
        col for col in ALL_CANDIDATES
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]
    for col in available_expected:
        assert col in out_df.columns


def test_all_features_numeric(tmp_path):
    df = make_input_df()
    out_df, _ = build_with_df(tmp_path, df)

    X = out_df.drop(columns=["player_tag"])
    for col in X.columns:
        assert pd.api.types.is_numeric_dtype(X[col])


def test_duplicate_player_tag_raises(tmp_path):
    df = make_input_df(include_duplicates=True)
    input_path = tmp_path / "player_features.parquet"
    output_path = tmp_path / "player_clustering.parquet"
    df.to_parquet(input_path, index=False)

    with pytest.raises(ValueError):
        build_player_clustering_dataset(input_path, output_path)


def test_missing_policy_median_imputation(tmp_path):
    df = make_input_df(n_rows=5, include_missing=True)
    out_df, _ = build_with_df(tmp_path, df)

    X = out_df.drop(columns=["player_tag"])
    assert not X.isna().any().any()

    donations_median = df["donations"].median(skipna=True)
    assert out_df.loc[0, "donations"] == pytest.approx(donations_median)

    war_stars_median = df["war_stars"].median(skipna=True)
    assert out_df.loc[1, "war_stars"] == pytest.approx(war_stars_median)

    hero_median = df["hero_mean_completion_ratio"].median(skipna=True)
    assert out_df.loc[2, "hero_mean_completion_ratio"] == pytest.approx(hero_median)


def test_immutability_input_file_unchanged(tmp_path):
    input_path = tmp_path / "player_features.parquet"
    output_path = tmp_path / "out.parquet"
    df = make_input_df()
    df.to_parquet(input_path, index=False)
    df_before = pd.read_parquet(input_path)

    build_player_clustering_dataset(input_path, output_path)
    df_after = pd.read_parquet(input_path)

    pd.testing.assert_frame_equal(df_before, df_after)


def test_reproducibility_same_input_same_output(tmp_path):
    input_path = tmp_path / "player_features.parquet"
    out1 = tmp_path / "out1.parquet"
    out2 = tmp_path / "out2.parquet"
    df = make_input_df()
    df.to_parquet(input_path, index=False)

    build_player_clustering_dataset(input_path, out1)
    build_player_clustering_dataset(input_path, out2)

    df1 = pd.read_parquet(out1)
    df2 = pd.read_parquet(out2)
    pd.testing.assert_frame_equal(df1, df2)


def test_no_predefined_clusters(tmp_path):
    df = make_input_df()
    out_df, _ = build_with_df(tmp_path, df)

    forbidden_substrings = ("cluster", "cluster_id", "profile", "class", "label")
    for col in out_df.columns:
        lower = col.lower()
        assert not any(sub in lower for sub in forbidden_substrings)


def test_reuses_player_features_parquet(tmp_path):
    custom_df = make_input_df(n_rows=3)
    custom_df["trophies"] = 999
    input_path = tmp_path / "custom_player_features.parquet"
    output_path = tmp_path / "out_custom.parquet"
    custom_df.to_parquet(input_path, index=False)

    build_player_clustering_dataset(input_path, output_path)
    out_df = pd.read_parquet(output_path)

    assert (out_df["trophies"] == 999).all()


def test_excluded_identifiers_targets_not_in_final_features(tmp_path):
    df = make_input_df()
    out_df, _ = build_with_df(tmp_path, df)

    excluded = {
        "clan_tag",
        "role",
        "clan_rank",
        "previous_clan_rank",
        "war_success_rate",
        "performance_class",
        "some_id",
    }
    assert set(out_df.columns).isdisjoint(excluded)


def test_only_whitelist_features_in_X(tmp_path):
    df = make_input_df()
    out_df, _ = build_with_df(tmp_path, df)

    whitelist = set(ALL_CANDIDATES)
    X_cols = [col for col in out_df.columns if col != "player_tag"]

    assert all(col in whitelist for col in X_cols)
    assert "extra_numeric_feature" not in X_cols

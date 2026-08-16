import unittest
from contextlib import redirect_stdout
import io

import numpy as np
import pandas as pd

from src.features.problem2.build_clan_rank_dataset import (
    _merge_preserving_clan_values,
    build_clan_rank_features,
)
from src.features.problem2.clan_rank_analysis import (
    CANDIDATE_VARIABLES,
    audit_clan_rank_proxies,
    main,
    _merge_for_analysis,
)


def make_clan_members():
    return pd.DataFrame(
        {
            "clan_tag": ["#C1", "#C1", "#C2", "#C2"],
            "player_tag": ["#P1", "#P2", "#P3", "#P4"],
            "role": ["member", "admin", "coLeader", "leader"],
            "town_hall_level": [10, 11, 12, 10],
            "exp_level": [100, 150, 200, 120],
            "trophies": [2500, 3000, 3500, 2800],
            "builder_base_trophies": [2000, 2200, 2500, 2100],
            "donations": [500, 700, 1000, 800],
            "donations_received": [300, 450, 600, 400],
            "capital_contributions": [100, 150, 200, 120],
            "clan_rank": [2, 1, 2, 1],
            "previous_clan_rank": [3, 1, 3, 2],
        }
    )


def make_player_features():
    return pd.DataFrame(
        {
            "player_tag": ["#P1", "#P2", "#P3", "#P4"],
            "town_hall_level": [10, 11, 12, 10],
            "exp_level": [100, 150, 200, 120],
            "trophies": [2500, 3000, 3500, 2800],
            "best_trophies": [2600, 3100, 3600, 2900],
            "war_stars": [100, 200, 300, 150],
            "attack_wins": [50, 80, 120, 60],
            "defense_wins": [30, 40, 60, 35],
            "builder_hall_level": [5, 6, 7, 5],
            "builder_base_trophies": [2000, 2200, 2500, 2100],
            "best_builder_base_trophies": [2100, 2300, 2600, 2200],
            "donations": [500, 700, 1000, 800],
            "donations_received": [300, 450, 600, 400],
            "clan_capital_contributions": [100, 150, 200, 120],
            "donation_balance": [200, 250, 400, 400],
            "donation_ratio": [1.6667, 1.5556, 1.6667, 2.0],
            "combat_activity_total": [80, 120, 180, 95],
            "progression_ratio_trophies": [0.9615, 0.9677, 0.9722, 0.9655],
            "builder_progression_ratio": [0.9524, 0.9565, 0.9615, 0.9545],
            "troop_count": [2, 1, 2, 1],
            "troop_mean_level": [5.5, 4.0, 4.0, 0.0],
            "troop_mean_completion_ratio": [0.55, 0.4, 0.7, 0.0],
            "hero_count": [1, 1, 0, 0],
            "hero_mean_level": [20.0, 30.0, 0.0, 0.0],
            "hero_mean_completion_ratio": [0.4, 0.6, 0.0, 0.0],
            "spell_count": [1, 0, 1, 0],
            "spell_mean_level": [3.0, 0.0, 7.0, 0.0],
            "spell_mean_completion_ratio": [0.3, 0.0, 0.7, 0.0],
            "equipment_count": [1, 0, 0, 1],
            "equipment_mean_level": [8.0, 0.0, 0.0, 5.0],
            "equipment_mean_completion_ratio": [0.5333, 0.0, 0.0, 0.3333],
            "achievement_count": [1, 1, 0, 0],
            "achievement_completion_ratio": [0.1, 0.5, 0.0, 0.0],
        }
    )


def make_clans():
    return pd.DataFrame(
        {
            "clan_tag": ["#C1", "#C2"],
            "clan_level": [10, 12],
            "clan_points": [30000, 45000],
            "clan_capital_points": [1000, 1500],
            "members": [30, 25],
            "required_trophies": [2000, 2500],
            "war_frequency": ["always", "always"],
            "war_league": ["Crystal", "Master"],
            "capital_league": ["Gold", "Crystal"],
            "type": ["open", "invite_only"],
            "is_family_friendly": [True, False],
        }
    )


class TestProblem2Dataset(unittest.TestCase):
    def setUp(self):
        self.clan_members = make_clan_members()
        self.player_features = make_player_features()
        self.clans = make_clans()
        self.analysis = audit_clan_rank_proxies(
            self.clan_members, self.player_features
        )
        self.final = build_clan_rank_features(
            self.clan_members,
            self.player_features,
            self.clans,
        )

    def test_target_present(self):
        self.assertIn("clan_rank", self.final.columns)
        self.assertFalse(self.final["clan_rank"].isna().any())

    def test_no_missing_target_unexpected(self):
        self.assertEqual(self.final["clan_rank"].isna().sum(), 0)

    def test_granularity_one_row_per_player_clan(self):
        self.assertEqual(len(self.final), len(self.clan_members))
        self.assertEqual(
            self.final.duplicated(subset=["player_tag", "clan_tag"]).sum(), 0
        )

    def test_player_tag_preserved(self):
        expected_players = set(self.clan_members["player_tag"])
        self.assertEqual(set(self.final["player_tag"]), expected_players)

    def test_clan_tag_preserved(self):
        expected_clans = set(self.clan_members["clan_tag"])
        self.assertEqual(set(self.final["clan_tag"]), expected_clans)

    def test_previous_clan_rank_excluded(self):
        self.assertNotIn("previous_clan_rank", self.final.columns)

    def test_role_not_in_features(self):
        feature_cols = set(self.final.columns) - {"player_tag", "clan_tag", "clan_rank"}
        self.assertNotIn("role", feature_cols)
        self.assertNotIn("role", self.final.columns)

    def test_clan_rank_not_in_features(self):
        feature_cols = set(self.final.columns) - {"player_tag", "clan_tag", "clan_rank"}
        self.assertTrue(feature_cols.isdisjoint({"clan_rank", "previous_clan_rank", "role"}))

    def test_relative_features_exist_for_safe_vars(self):
        self.assertIn("exp_level_diff_from_clan_mean", self.final.columns)
        self.assertIn("war_stars_ratio_to_clan_mean", self.final.columns)

    def test_no_row_multiplication(self):
        self.assertEqual(len(self.final), len(self.clan_members))

    def test_missing_player_profiles_are_dropped(self):
        cm = self.clan_members.copy()
        cm = cm[cm["player_tag"] != "#P3"]
        new_row = {
            "clan_tag": "#C2",
            "player_tag": "#PX",
            "role": "member",
            "town_hall_level": 9,
            "exp_level": 80,
            "trophies": 2000,
            "builder_base_trophies": 1500,
            "donations": 300,
            "donations_received": 200,
            "capital_contributions": 50,
            "clan_rank": 3,
            "previous_clan_rank": 4,
        }
        cm = pd.concat([cm, pd.DataFrame([new_row])], ignore_index=True)
        pf = self.player_features.copy()
        final = build_clan_rank_features(cm, pf, self.clans)
        self.assertNotIn("#PX", final["player_tag"].tolist())
        self.assertEqual(len(final), len(cm) - 1)

    def test_trophies_are_audited(self):
        trophies_row = self.analysis[self.analysis["variable"] == "trophies"].iloc[0]
        self.assertIn(
            trophies_row["classification"],
            ["SAFE", "POTENTIAL PROXY", "TOO_DIRECT", "EXCLUDE"],
        )

    def test_trophies_excluded_if_too_direct(self):
        trophies_class = self.analysis[self.analysis["variable"] == "trophies"]["classification"].iloc[0]
        feature_cols = set(self.final.columns) - {"player_tag", "clan_tag", "clan_rank"}
        if trophies_class in {"TOO_DIRECT", "EXCLUDE"}:
            self.assertNotIn("trophies", feature_cols)
            self.assertNotIn("trophies_diff_from_clan_mean", feature_cols)
            self.assertNotIn("trophies_clan_pct", feature_cols)

    def test_merge_preserving_clan_values(self):
        merged = _merge_preserving_clan_values(self.clan_members, self.player_features)
        for _, row in merged.iterrows():
            self.assertIn(row["trophies"], [2500, 3000, 3500, 2800])

    # --- Nuevos tests para la tarea ---

    def test_audit_returns_results(self):
        self.assertGreater(len(self.analysis), 0)

    def test_trophies_in_analysis(self):
        self.assertIn("trophies", self.analysis["variable"].tolist())

    def test_candidate_variables_appear(self):
        for var in CANDIDATE_VARIABLES:
            if var in self.clan_members.columns or var in self.player_features.columns:
                self.assertIn(var, self.analysis["variable"].tolist())

    def test_each_variable_has_classification(self):
        self.assertFalse(self.analysis["classification"].isna().any())
        valid = {"SAFE", "POTENTIAL PROXY", "TOO_DIRECT", "EXCLUDE"}
        for c in self.analysis["classification"]:
            self.assertIn(c, valid)

    def test_main_runs_and_prints_report(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            main(self.clan_members, self.player_features)
        output = buf.getvalue()
        self.assertIn("Número de observaciones analizadas", output)
        self.assertIn("Conclusión sobre trophies", output)
        self.assertIn("trophies", output)

    def test_merge_for_analysis_with_player_tag_column(self):
        merged = _merge_for_analysis(self.clan_members, self.player_features)
        self.assertIn("player_tag", merged.columns)
        self.assertEqual(len(merged), 4)

    def test_merge_for_analysis_with_player_tag_index(self):
        # Caso con índice sin nombre que contiene player_tag
        pf_indexed = self.player_features.set_index("player_tag").rename_axis(None)
        merged = _merge_for_analysis(self.clan_members, pf_indexed)
        self.assertIn("player_tag", merged.columns)
        self.assertEqual(len(merged), 4)


if __name__ == "__main__":
    unittest.main()

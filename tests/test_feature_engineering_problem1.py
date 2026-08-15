import unittest

import numpy as np
import pandas as pd

from src.features.player_features import (
    aggregate_progression_df,
    aggregate_achievements,
    build_player_features,
)
from src.features.player_clan_features import compute_clan_relative_features
from src.features.build_role_dataset import (
    select_clan_context_features,
    assemble_role_dataset,
    build_all_features,
)


def make_players_df():
    """Crea un DataFrame de jugadores de ejemplo."""
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
        }
    )


def make_troops_df():
    return pd.DataFrame(
        {
            "player_tag": ["#P1", "#P1", "#P2"],
            "level": [5, 6, 4],
            "max_level": [10, 10, 10],
        }
    )


def make_heroes_df():
    return pd.DataFrame(
        {
            "player_tag": ["#P1", "#P2"],
            "level": [20, 30],
            "max_level": [50, 50],
        }
    )


def make_spells_df():
    return pd.DataFrame(
        {
            "player_tag": ["#P1", "#P3"],
            "level": [3, 7],
            "max_level": [10, 10],
        }
    )


def make_equipment_df():
    return pd.DataFrame(
        {
            "player_tag": ["#P1", "#P4"],
            "level": [8, 5],
            "max_level": [15, 15],
        }
    )


def make_achievements_df():
    return pd.DataFrame(
        {
            "player_tag": ["#P1", "#P2"],
            "value": [1, 5],
            "target": [10, 10],
        }
    )


def make_clan_members_df():
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


def make_clans_df():
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


class TestPlayerFeatures(unittest.TestCase):
    def test_aggregate_progression_df_columns_and_values(self):
        troops = make_troops_df()
        agg = aggregate_progression_df(troops, prefix="troop")
        self.assertIn("troop_count", agg.columns)
        self.assertIn("troop_mean_level", agg.columns)
        self.assertIn("troop_mean_completion_ratio", agg.columns)
        # #P1 tiene 2 tropas, nivel medio (5+6)/2 = 5.5
        self.assertAlmostEqual(agg.loc["#P1", "troop_count"], 2)
        self.assertAlmostEqual(agg.loc["#P1", "troop_mean_level"], 5.5)

    def test_aggregate_achievements_columns_and_values(self):
        ach = make_achievements_df()
        agg = aggregate_achievements(ach)
        self.assertIn("achievement_count", agg.columns)
        self.assertIn("achievement_completion_ratio", agg.columns)
        # #P1: value=1, target=10 -> ratio=0.1
        self.assertAlmostEqual(
            agg.loc["#P1", "achievement_completion_ratio"], 0.1
        )

    def test_build_player_features_creates_expected_columns(self):
        players = make_players_df()
        troops = make_troops_df()
        heroes = make_heroes_df()
        spells = make_spells_df()
        equipment = make_equipment_df()
        achievements = make_achievements_df()

        pf = build_player_features(
            players, troops, heroes, spells, equipment, achievements
        )
        expected_cols = [
            "player_tag",
            "donation_balance",
            "donation_ratio",
            "combat_activity_total",
            "progression_ratio_trophies",
            "builder_progression_ratio",
            "troop_count",
            "hero_count",
            "spell_count",
            "equipment_count",
            "achievement_count",
        ]
        for col in expected_cols:
            self.assertIn(col, pf.columns)
        # Los jugadores sin tropas/heroes deben tener 0
        self.assertEqual(pf.loc[pf.player_tag == "#P4", "hero_count"].iloc[0], 0)

    def test_missing_progression_rows_filled_with_zero(self):
        players = make_players_df()
        # Solo tropas para #P1, sin filas de heroes
        troops = make_troops_df()
        heroes = pd.DataFrame(columns=["player_tag", "level", "max_level"])
        spells = make_spells_df()
        equipment = make_equipment_df()
        achievements = make_achievements_df()

        pf = build_player_features(
            players, troops, heroes, spells, equipment, achievements
        )
        # #P3 no tiene heroes, debe ser 0
        self.assertEqual(pf.loc[pf.player_tag == "#P3", "hero_count"].iloc[0], 0)
        self.assertEqual(
            pf.loc[pf.player_tag == "#P3", "hero_mean_level"].iloc[0], 0
        )


class TestPlayerClanFeatures(unittest.TestCase):
    def test_compute_clan_relative_features_columns(self):
        clan_members = make_clan_members_df()
        players = make_players_df()
        pf = build_player_features(
            players,
            make_troops_df(),
            make_heroes_df(),
            make_spells_df(),
            make_equipment_df(),
            make_achievements_df(),
        )
        pcf = compute_clan_relative_features(clan_members, pf)
        # Verificar que existen columnas de diff, ratio y percentil
        self.assertIn("trophies_diff_from_clan_mean", pcf.columns)
        self.assertIn("exp_level_ratio_to_clan_mean", pcf.columns)
        self.assertIn("trophies_clan_pct", pcf.columns)
        self.assertIn("war_stars_clan_pct", pcf.columns)

    def test_clan_relative_values_are_reasonable(self):
        clan_members = make_clan_members_df()
        players = make_players_df()
        pf = build_player_features(
            players,
            make_troops_df(),
            make_heroes_df(),
            make_spells_df(),
            make_equipment_df(),
            make_achievements_df(),
        )
        pcf = compute_clan_relative_features(clan_members, pf)

        # Para #C1: tropies = 2500,3000 -> media 2750
        # jugador #P1 (2500) diff = -250
        row_p1 = pcf[(pcf.clan_tag == "#C1") & (pcf.player_tag == "#P1")]
        self.assertAlmostEqual(
            row_p1.iloc[0]["trophies_diff_from_clan_mean"], -250.0
        )
        # Percentil de #P1: 1 de 2 -> 0.5
        self.assertAlmostEqual(row_p1.iloc[0]["trophies_clan_pct"], 0.5)


class TestAssembleRoleDataset(unittest.TestCase):
    def test_final_dataset_granularity_and_role(self):
        clan_members = make_clan_members_df()
        players = make_players_df()
        clans = make_clans_df()

        pf, pcf = build_all_features(
            clan_members,
            players,
            make_troops_df(),
            make_heroes_df(),
            make_spells_df(),
            make_equipment_df(),
            make_achievements_df(),
        )
        final = assemble_role_dataset(pf, pcf, clan_members, clans)

        # 1 fila por relación player-clan
        self.assertEqual(len(final), len(clan_members))
        # sin pares duplicados
        self.assertEqual(
            final.duplicated(subset=["clan_tag", "player_tag"]).sum(), 0
        )
        # roles preservados
        self.assertEqual(set(final["role"]), set(clan_members["role"]))

    def test_leakage_prevention_feature_columns(self):
        clan_members = make_clan_members_df()
        players = make_players_df()
        clans = make_clans_df()

        pf, pcf = build_all_features(
            clan_members,
            players,
            make_troops_df(),
            make_heroes_df(),
            make_spells_df(),
            make_equipment_df(),
            make_achievements_df(),
        )
        final = assemble_role_dataset(pf, pcf, clan_members, clans)

        feature_cols = set(final.columns) - {"player_tag", "clan_tag", "role"}
        forbidden = {"role", "clan_rank", "previous_clan_rank"}
        self.assertTrue(feature_cols.isdisjoint(forbidden))

    def test_clan_tag_not_overwritten(self):
        clan_members = make_clan_members_df()
        players = make_players_df()
        # Forzar un clan_tag diferente en players para un jugador
        players.loc[players.player_tag == "#P1", "clan_tag"] = "#OTHER"
        clans = make_clans_df()

        pf, pcf = build_all_features(
            clan_members,
            players,
            make_troops_df(),
            make_heroes_df(),
            make_spells_df(),
            make_equipment_df(),
            make_achievements_df(),
        )
        final = assemble_role_dataset(pf, pcf, clan_members, clans)

        # El clan_tag del dataset debe venir de clan_members, no de players
        row_p1 = final[final.player_tag == "#P1"].iloc[0]
        self.assertEqual(row_p1["clan_tag"], "#C1")

    def test_player_tag_preserved(self):
        clan_members = make_clan_members_df()
        players = make_players_df()
        clans = make_clans_df()

        pf, pcf = build_all_features(
            clan_members,
            players,
            make_troops_df(),
            make_heroes_df(),
            make_spells_df(),
            make_equipment_df(),
            make_achievements_df(),
        )
        final = assemble_role_dataset(pf, pcf, clan_members, clans)
        self.assertEqual(set(final.player_tag), set(clan_members.player_tag))


if __name__ == "__main__":
    unittest.main()

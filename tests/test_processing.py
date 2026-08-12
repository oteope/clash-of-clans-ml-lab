import unittest
from unittest.mock import patch

import pandas as pd

from src.processing.build_normalized_tables import (
    normalize_clan,
    normalize_member,
    normalize_player,
    normalize_troops,
    normalize_heroes,
    normalize_hero_equipment,
    normalize_spells,
    normalize_achievements,
    _deduplicate_dataframe,
)


class TestNormalizationFunctions(unittest.TestCase):
    """Unit tests for the individual normalisation helpers."""

    # ------------------------------------------------------------------
    # Clan
    # ------------------------------------------------------------------
    def test_normalize_clan_basic(self):
        raw = {
            "tag": "#CLAN123",
            "name": "BestClan",
            "clanLevel": 12,
            "clanPoints": 50000,
            "location": {"id": 32000006, "name": "Spain"},
        }
        row = normalize_clan(raw)
        self.assertEqual(row["clan_tag"], "#CLAN123")
        self.assertEqual(row["name"], "BestClan")
        self.assertEqual(row["clan_level"], 12)
        self.assertEqual(row["location_id"], 32000006)
        self.assertEqual(row["location_name"], "Spain")
        self.assertIsNone(row["clan_builder_base_points"])  # missing field

    def test_normalize_clan_missing_location(self):
        raw = {"tag": "#CLANX", "clanLevel": 5}
        row = normalize_clan(raw)
        self.assertIsNone(row["location_id"])
        self.assertIsNone(row["location_name"])

    # ------------------------------------------------------------------
    # Member
    # ------------------------------------------------------------------
    def test_normalize_member(self):
        member = {
            "tag": "#PLAY1",
            "name": "John",
            "townHallLevel": 14,
            "trophies": 3200,
            "league": {"id": 1, "name": "Bronze"},
        }
        row = normalize_member(member, "#CLAN123")
        self.assertEqual(row["clan_tag"], "#CLAN123")
        self.assertEqual(row["player_tag"], "#PLAY1")
        self.assertEqual(row["town_hall_level"], 14)
        self.assertEqual(row["league_id"], 1)
        self.assertEqual(row["league_name"], "Bronze")

    def test_normalize_member_missing_league(self):
        member = {"tag": "#P", "trophies": 1000}
        row = normalize_member(member, "#C")
        self.assertIsNone(row["league_id"])

    # ------------------------------------------------------------------
    # Player
    # ------------------------------------------------------------------
    def test_normalize_player(self):
        raw = {
            "tag": "#PLAY1",
            "name": "Alice",
            "townHallLevel": 11,
            "clan": {"tag": "#CLAN1", "name": "Giants", "clanLevel": 10},
        }
        row = normalize_player(raw)
        self.assertEqual(row["player_tag"], "#PLAY1")
        self.assertEqual(row["clan_tag"], "#CLAN1")
        self.assertEqual(row["clan_name"], "Giants")
        self.assertEqual(row["clan_level"], 10)
        self.assertIsNone(row["trophies"])   # not present

    # ------------------------------------------------------------------
    # Troops
    # ------------------------------------------------------------------
    def test_normalize_troops(self):
        troops = [
            {"name": "Barbarian", "level": 5, "maxLevel": 8, "village": "home"},
            {"name": "Archer", "level": 6, "maxLevel": 8, "village": "home"},
        ]
        rows = normalize_troops("#P1", troops)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["player_tag"], "#P1")
        self.assertEqual(rows[0]["troop_name"], "Barbarian")

    def test_normalize_troops_empty(self):
        self.assertEqual(normalize_troops("#P1", None), [])
        self.assertEqual(normalize_troops("#P1", []), [])

    # ------------------------------------------------------------------
    # Heroes
    # ------------------------------------------------------------------
    def test_normalize_heroes(self):
        heroes = [{"name": "Barbarian King", "level": 30, "maxLevel": 80, "village": "home"}]
        rows = normalize_heroes("#P1", heroes)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["hero_name"], "Barbarian King")

    # ------------------------------------------------------------------
    # Equipment
    # ------------------------------------------------------------------
    def test_normalize_hero_equipment(self):
        eq = [{"name": "Vampstache", "level": 9, "maxLevel": 15, "village": "home"}]
        rows = normalize_hero_equipment("#P1", eq)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["equipment_name"], "Vampstache")

    # ------------------------------------------------------------------
    # Spells
    # ------------------------------------------------------------------
    def test_normalize_spells(self):
        spells = [{"name": "Lightning", "level": 8, "maxLevel": 10, "village": "home"}]
        rows = normalize_spells("#P1", spells)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["spell_name"], "Lightning")

    # ------------------------------------------------------------------
    # Achievements
    # ------------------------------------------------------------------
    def test_normalize_achievements(self):
        ach = [{"name": "Nice and Tidy", "stars": 2, "value": 100, "target": 100, "village": "home"}]
        rows = normalize_achievements("#P1", ach)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["achievement_name"], "Nice and Tidy")
        self.assertEqual(rows[0]["stars"], 2)

    # ------------------------------------------------------------------
    # Edge cases – missing / malformed input
    # ------------------------------------------------------------------
    def test_player_missing_clan(self):
        raw = {"tag": "#P2", "name": "Bob"}
        row = normalize_player(raw)
        self.assertIsNone(row["clan_tag"])

    def test_troop_missing_village(self):
        troops = [{"name": "Wizard", "level": 6}]
        rows = normalize_troops("#P3", troops)
        self.assertIsNone(rows[0]["village"])


class TestDeduplication(unittest.TestCase):
    """Tests for the DataFrame-based deduplication helper."""

    def test_dedup_keeps_first(self):
        df = pd.DataFrame([
            {"tag": "A", "value": 1},
            {"tag": "A", "value": 2},
            {"tag": "B", "value": 3},
        ])
        deduped = _deduplicate_dataframe(df, ["tag"])
        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped.loc[deduped["tag"] == "A", "value"].iloc[0], 1)


class TestPipelineIsolation(unittest.TestCase):
    """Verify that the pipeline functions do NOT modify the input data."""

    def test_normalize_clan_does_not_mutate_raw(self):
        raw = {"tag": "#CL", "clanLevel": 3, "location": {"id": 1}}
        original = raw.copy()
        _ = normalize_clan(raw)
        self.assertEqual(raw, original)

    def test_normalize_troops_does_not_mutate_list(self):
        troops = [{"name": "Healer", "level": 4}]
        original = troops[:]
        _ = normalize_troops("#P", troops)
        self.assertEqual(troops, original)


if __name__ == "__main__":
    unittest.main()

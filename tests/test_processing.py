import json
import pathlib
import tempfile
import unittest

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

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
    _ParquetBatchWriter,
    build_all_tables,
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
        self.assertIsNone(row["clan_builder_base_points"])
        self.assertIsNone(row["war_league"])

    def test_normalize_clan_serializes_league_objects(self):
        raw = {
            "tag": "#CLANX",
            "warLeague": {"id": 1, "name": "Gold"},
            "capitalLeague": {"id": 2, "name": "Bronze"},
        }
        row = normalize_clan(raw)
        self.assertEqual(row["war_league"], '{"id": 1, "name": "Gold"}')
        self.assertEqual(row["capital_league"], '{"id": 2, "name": "Bronze"}')

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
            "leagueTier": {"id": 8, "name": "Crystal"},
        }
        row = normalize_member(member, "#CLAN123")
        self.assertEqual(row["clan_tag"], "#CLAN123")
        self.assertEqual(row["player_tag"], "#PLAY1")
        self.assertEqual(row["town_hall_level"], 14)
        self.assertEqual(row["league_id"], 1)
        self.assertEqual(row["league_name"], "Bronze")
        self.assertEqual(row["league_tier_id"], 8)
        self.assertEqual(row["league_tier_name"], "Crystal")
        self.assertIsNone(row["capital_contributions"])

    def test_normalize_member_capital_contributions(self):
        member = {"tag": "#P", "clanCapitalContributions": 1234}
        row = normalize_member(member, "#C")
        self.assertEqual(row["capital_contributions"], 1234)

    def test_normalize_member_missing_league(self):
        member = {"tag": "#P", "trophies": 1000}
        row = normalize_member(member, "#C")
        self.assertIsNone(row["league_id"])
        self.assertIsNone(row["league_tier_id"])

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
        self.assertIsNone(row["trophies"])

    def test_player_missing_clan(self):
        raw = {"tag": "#P2", "name": "Bob"}
        row = normalize_player(raw)
        self.assertIsNone(row["clan_tag"])

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

    def test_troop_missing_village(self):
        troops = [{"name": "Wizard", "level": 6}]
        rows = normalize_troops("#P3", troops)
        self.assertIsNone(rows[0]["village"])

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
        self.assertNotIn("hero_name", rows[0])

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


class TestParquetBatchWriter(unittest.TestCase):
    """Tests for the batch writer with explicit schema alignment."""

    def test_schema_alignment_across_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "test.parquet"
            schema = pa.schema([
                ("a", pa.string()),
                ("b", pa.int64()),
                ("c", pa.string()),
            ])
            writer = _ParquetBatchWriter(path, schema=schema)
            writer.write_batch([{"a": "x", "b": 1}])
            writer.write_batch([{"a": "y", "b": 2, "c": "z", "d": 999}])
            writer.close()

            table = pq.read_table(path)
            self.assertEqual(table.schema.names, ["a", "b", "c"])
            df = table.to_pandas()
            self.assertEqual(len(df), 2)
            self.assertTrue(pd.isna(df.loc[0, "c"]))
            self.assertEqual(df.loc[1, "c"], "z")
            self.assertNotIn("d", df.columns)


class TestBuildAllTablesIntegration(unittest.TestCase):
    """Integration tests for the full processing pipeline using small controlled fixtures."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base = pathlib.Path(self.temp_dir.name)
        self.raw_base = self.base / "raw"
        self.processed_base = self.base / "processed"
        self.raw_base.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_json(self, subdir: str, filename: str, data):
        path = self.raw_base / subdir
        path.mkdir(parents=True, exist_ok=True)
        with (path / filename).open("w", encoding="utf-8") as f:
            json.dump(data, f)

    def _read_parquet(self, filename: str) -> pd.DataFrame:
        return pq.read_table(self.processed_base / filename).to_pandas()

    def test_build_creates_expected_parquet_files(self):
        # Clan
        self._write_json("clans", "#CLAN1.json", {
            "tag": "#CLAN1",
            "name": "Clan A",
            "clanLevel": 10,
            "clanPoints": 1000,
            "location": {"id": 1, "name": "Spain"},
            "warLeague": {"id": 1, "name": "Gold"},
            "capitalLeague": {"id": 2, "name": "Bronze"},
        })

        # Members
        self._write_json("members", "#CLAN1.json", {
            "items": [
                {
                    "tag": "#PLAY1",
                    "name": "John",
                    "townHallLevel": 12,
                    "trophies": 1500,
                    "league": {"id": 1, "name": "Gold"},
                    "leagueTier": {"id": 2, "name": "Silver"},
                    "clanCapitalContributions": 500,
                }
            ]
        })

        # Player
        self._write_json("players", "#PLAY1.json", {
            "tag": "#PLAY1",
            "name": "John",
            "townHallLevel": 12,
            "expLevel": 100,
            "trophies": 1500,
            "bestTrophies": 1700,
            "warStars": 200,
            "attackWins": 100,
            "defenseWins": 50,
            "builderHallLevel": 5,
            "builderBaseTrophies": 1000,
            "bestBuilderBaseTrophies": 1200,
            "donations": 30,
            "donationsReceived": 20,
            "clanCapitalContributions": 500,
            "clan": {"tag": "#CLAN1", "name": "Clan A", "clanLevel": 10},
            "troops": [
                {"name": "Barbarian", "level": 8, "maxLevel": 10, "village": "home"},
                {"name": "Archer", "level": 8, "maxLevel": 10, "village": "home"},
            ],
            "heroes": [
                {"name": "Barbarian King", "level": 50, "maxLevel": 80, "village": "home"},
            ],
            "heroEquipment": [
                {"name": "Vampstache", "level": 9, "maxLevel": 15, "village": "home"},
            ],
            "spells": [
                {"name": "Lightning", "level": 8, "maxLevel": 10, "village": "home"},
            ],
            "achievements": [
                {"name": "Nice and Tidy", "stars": 2, "value": 100, "target": 100, "village": "home"},
            ],
        })

        build_all_tables(raw_base=self.raw_base, processed_base=self.processed_base)

        expected_files = [
            "clans.parquet",
            "clan_members.parquet",
            "players.parquet",
            "player_troops.parquet",
            "player_heroes.parquet",
            "player_hero_equipment.parquet",
            "player_spells.parquet",
            "player_achievements.parquet",
        ]
        for fname in expected_files:
            self.assertTrue((self.processed_base / fname).exists(), f"Falta {fname}")

        clans = self._read_parquet("clans.parquet")
        self.assertEqual(len(clans), 1)
        self.assertEqual(clans.loc[0, "clan_tag"], "#CLAN1")
        self.assertEqual(clans.loc[0, "war_league"], '{"id": 1, "name": "Gold"}')

        members = self._read_parquet("clan_members.parquet")
        self.assertEqual(len(members), 1)
        self.assertEqual(members.loc[0, "capital_contributions"], 500)

        players = self._read_parquet("players.parquet")
        self.assertEqual(len(players), 1)
        self.assertEqual(players.loc[0, "player_tag"], "#PLAY1")

        troops = self._read_parquet("player_troops.parquet")
        self.assertEqual(len(troops), 2)

        heroes = self._read_parquet("player_heroes.parquet")
        self.assertEqual(len(heroes), 1)

        equipment = self._read_parquet("player_hero_equipment.parquet")
        self.assertEqual(len(equipment), 1)

        spells = self._read_parquet("player_spells.parquet")
        self.assertEqual(len(spells), 1)

        achievements = self._read_parquet("player_achievements.parquet")
        self.assertEqual(len(achievements), 1)

    def test_build_reproducible_no_duplicates(self):
        self._write_json("clans", "#CLAN1.json", {
            "tag": "#CLAN1",
            "name": "Clan A",
            "clanLevel": 5,
        })
        self._write_json("members", "#CLAN1.json", {
            "items": [{"tag": "#PLAY1", "name": "John"}]
        })
        self._write_json("players", "#PLAY1.json", {
            "tag": "#PLAY1",
            "name": "John",
            "townHallLevel": 10,
            "clan": {"tag": "#CLAN1", "name": "Clan A", "clanLevel": 5},
        })

        # Primera ejecución
        build_all_tables(raw_base=self.raw_base, processed_base=self.processed_base)
        players_first = self._read_parquet("players.parquet")
        count_first = len(players_first)
        self.assertEqual(count_first, 1)

        # Segunda ejecución (debe limpiar y reconstruir)
        build_all_tables(raw_base=self.raw_base, processed_base=self.processed_base)
        players_second = self._read_parquet("players.parquet")
        count_second = len(players_second)

        self.assertEqual(count_first, count_second)
        self.assertEqual(count_second, 1)

        # Los archivos raw no deben haber sido modificados
        raw_file = self.raw_base / "players" / "#PLAY1.json"
        self.assertTrue(raw_file.exists())
        with raw_file.open("r", encoding="utf-8") as f:
            raw_content = json.load(f)
        self.assertEqual(raw_content["tag"], "#PLAY1")

    def test_corrupt_json_is_skipped(self):
        # Archivo corrupto + archivo válido
        corrupt_path = self.raw_base / "players" / "corrupt.json"
        corrupt_path.parent.mkdir(parents=True, exist_ok=True)
        corrupt_path.write_text("{invalid json", encoding="utf-8")

        self._write_json("players", "good.json", {
            "tag": "#GOOD",
            "name": "Good Player",
            "townHallLevel": 9,
        })

        with self.assertLogs("src.processing.build_normalized_tables", level="WARNING") as log:
            build_all_tables(raw_base=self.raw_base, processed_base=self.processed_base)

        self.assertTrue(any("Failed to load" in msg for msg in log.output))

        players = self._read_parquet("players.parquet")
        self.assertEqual(len(players), 1)
        self.assertEqual(players.loc[0, "player_tag"], "#GOOD")

    def test_dedup_nested_entities(self):
        self._write_json("players", "#P1.json", {
            "tag": "#P1",
            "name": "Player",
            "townHallLevel": 11,
            "troops": [
                {"name": "Barbarian", "level": 8, "maxLevel": 10, "village": "home"},
                {"name": "Barbarian", "level": 9, "maxLevel": 10, "village": "home"},
                {"name": "Archer", "level": 7, "maxLevel": 10, "village": "home"},
            ],
            "heroes": [
                {"name": "Barbarian King", "level": 30, "maxLevel": 80, "village": "home"},
                {"name": "Barbarian King", "level": 31, "maxLevel": 80, "village": "home"},
            ],
        })

        build_all_tables(raw_base=self.raw_base, processed_base=self.processed_base)

        troops = self._read_parquet("player_troops.parquet")
        self.assertEqual(len(troops), 2)
        barbarian = troops[troops["troop_name"] == "Barbarian"]
        self.assertEqual(len(barbarian), 1)
        self.assertEqual(barbarian.iloc[0]["level"], 8)

        heroes = self._read_parquet("player_heroes.parquet")
        self.assertEqual(len(heroes), 1)
        self.assertEqual(heroes.loc[0, "hero_name"], "Barbarian King")

    def test_no_relations_discarded(self):
        self._write_json("clans", "#CLAN1.json", {
            "tag": "#CLAN1",
            "name": "Clan A",
            "clanLevel": 3,
        })
        self._write_json("members", "#CLAN1.json", {
            "items": [{"tag": "#GHOST", "name": "Ghost Player"}]
        })
        # No creamos players/#GHOST.json

        build_all_tables(raw_base=self.raw_base, processed_base=self.processed_base)

        members = self._read_parquet("clan_members.parquet")
        self.assertEqual(len(members), 1)
        self.assertEqual(members.loc[0, "player_tag"], "#GHOST")
        self.assertEqual(members.loc[0, "clan_tag"], "#CLAN1")


if __name__ == "__main__":
    unittest.main()

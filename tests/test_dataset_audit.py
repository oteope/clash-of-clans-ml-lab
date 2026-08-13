import json
import sys
import tempfile
import unittest
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audit.dataset_audit import run_audit, generate_markdown_report, generate_json_report


class TestDatasetAudit(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base = Path(self.temp_dir.name)
        self.raw = self.base / "data" / "raw"
        self.raw.mkdir(parents=True, exist_ok=True)
        (self.raw / "players").mkdir(exist_ok=True)
        (self.raw / "clans").mkdir(exist_ok=True)
        (self.raw / "members").mkdir(exist_ok=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_json(self, subdir: str, filename: str, data):
        path = self.raw / subdir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return path

    def test_player_count_and_unique_tags(self):
        self._write_json("players", "p1.json", {"tag": "#P1", "townHallLevel": 5})
        self._write_json("players", "p2.json", {"tag": "#P2", "townHallLevel": 7})
        report = run_audit(self.raw)
        self.assertEqual(report["players"]["total"], 2)
        self.assertEqual(report["players"]["unique_tags"], 2)

    def test_clan_count(self):
        self._write_json("clans", "c1.json", {"tag": "#C1", "clanLevel": 10})
        self._write_json("clans", "c2.json", {"tag": "#C2", "clanLevel": 12})
        report = run_audit(self.raw)
        self.assertEqual(report["clans"]["total"], 2)
        self.assertEqual(report["clans"]["unique_tags"], 2)

    def test_townhall_distribution(self):
        self._write_json("players", "p1.json", {"tag": "#P1", "townHallLevel": 5})
        self._write_json("players", "p2.json", {"tag": "#P2", "townHallLevel": 5})
        self._write_json("players", "p3.json", {"tag": "#P3", "townHallLevel": 10})
        report = run_audit(self.raw)
        th_dist = report["players"]["distributions"]["townHallLevel"]
        self.assertEqual(th_dist["observed_categories"], 2)
        self.assertEqual(th_dist["distribution"][0]["value"], "5")
        self.assertEqual(th_dist["distribution"][0]["count"], 2)

    def test_location_distribution(self):
        self._write_json("clans", "c1.json", {
            "tag": "#C1",
            "location": {"id": 32000000, "name": "Europe"}
        })
        self._write_json("clans", "c2.json", {
            "tag": "#C2",
            "location": {"id": 32000001, "name": "North America"}
        })
        self._write_json("clans", "c3.json", {
            "tag": "#C3",
            "location": {"id": 32000000, "name": "Europe"}
        })
        report = run_audit(self.raw)
        loc_dist = report["clans"]["distributions"]["location"]
        self.assertEqual(loc_dist["observed_categories"], 2)
        self.assertEqual(loc_dist["dominant"], "32000000|Europe")
        self.assertAlmostEqual(loc_dist["dominant_percentage"], 66.666, places=2)

    def test_players_per_clan_and_concentration(self):
        # Estructura real de data/raw/members/<CLAN_TAG>.json
        self._write_json("members", "#C1.json", {
            "items": [
                {"tag": "#P1"},
                {"tag": "#P2"}
            ]
        })
        self._write_json("members", "#C2.json", {
            "items": [
                {"tag": "#P3"},
                {"tag": "#P4"},
                {"tag": "#P5"}
            ]
        })
        self._write_json("members", "#C3.json", {
            "items": [
                {"tag": "#P6"}
            ]
        })

        report = run_audit(self.raw)
        member_stats = report["members"]["stats"]
        # Total relationships = 6
        self.assertEqual(report["members"]["total_relationships"], 6)
        # Unique clans = 3
        self.assertEqual(report["members"]["unique_clans_represented"], 3)
        # Unique players = 6
        self.assertEqual(report["members"]["unique_players_represented"], 6)
        # Average per clan = 6/3 = 2
        self.assertAlmostEqual(member_stats["average_players_per_clan"], 2.0)
        # Median per clan: clan sizes [2,3,1] -> sorted [1,2,3] median = 2
        self.assertEqual(member_stats["median_players_per_clan"], 2.0)
        # Min = 1, max = 3
        self.assertEqual(member_stats["min_players_per_clan"], 1)
        self.assertEqual(member_stats["max_players_per_clan"], 3)
        # Top 1 concentration: top clan size 3 / 6 = 50%
        self.assertAlmostEqual(member_stats["concentration_percentages"]["top_1"], 50.0)

    def test_numeric_stats(self):
        self._write_json("players", "p1.json", {"tag": "#P1", "trophies": 1000, "warStars": 50})
        self._write_json("players", "p2.json", {"tag": "#P2", "trophies": 2000, "warStars": 60})
        self._write_json("players", "p3.json", {"tag": "#P3", "trophies": 3000, "warStars": 70})
        report = run_audit(self.raw)
        trophies = report["players"]["numeric_stats"]["trophies"]
        self.assertEqual(trophies["count"], 3)
        self.assertEqual(trophies["min"], 1000.0)
        self.assertEqual(trophies["max"], 3000.0)
        self.assertAlmostEqual(trophies["mean"], 2000.0)
        self.assertAlmostEqual(trophies["std"], 1000.0, places=2)

    def test_missing_fields(self):
        self._write_json("players", "p1.json", {"tag": "#P1", "townHallLevel": 5})
        self._write_json("players", "p2.json", {"tag": "#P2"})
        report = run_audit(self.raw)
        missing = report["players"]["missing_fields"]
        self.assertEqual(missing["townHallLevel"], 1)
        self.assertEqual(report["players"]["total"], 2)

    def test_concentration_warning(self):
        for i in range(9):
            self._write_json("players", f"p{i}.json", {"tag": f"#P{i}", "townHallLevel": 10})
        self._write_json("players", "p9.json", {"tag": "#P9", "townHallLevel": 5})
        report = run_audit(self.raw)
        warnings = report["warnings"]
        self.assertTrue(any(w["type"] == "POTENTIAL CONCENTRATION" for w in warnings))

    def test_markdown_generation(self):
        self._write_json("players", "p1.json", {"tag": "#P1", "townHallLevel": 7})
        report = run_audit(self.raw)
        md = generate_markdown_report(report)
        self.assertIn("# Dataset Audit", md)
        self.assertIn("## Overview", md)
        self.assertIn("## Player distributions", md)

    def test_json_generation(self):
        self._write_json("players", "p1.json", {"tag": "#P1"})
        report = run_audit(self.raw)
        json_str = generate_json_report(report)
        data = json.loads(json_str)
        self.assertEqual(data["players"]["total"], 1)

    def test_empty_dataset(self):
        report = run_audit(self.raw)
        self.assertEqual(report["players"]["total"], 0)
        self.assertEqual(report["clans"]["total"], 0)
        self.assertEqual(report["members"]["total_relationships"], 0)
        md = generate_markdown_report(report)
        self.assertIn("Limited coverage", md)

    def test_corrupt_json(self):
        path = self.raw / "players" / "bad.json"
        with open(path, "w", encoding="utf-8") as f:
            f.write("{not valid json")
        report = run_audit(self.raw)
        self.assertGreaterEqual(report["files"]["players_corrupt"], 1)
        self.assertEqual(report["players"]["total"], 0)


if __name__ == "__main__":
    unittest.main()

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from src.extraction.main_extraction import _perform_search, _resolve_location_ids
from src.extraction.api_client import CoCClient


class TestMainExtraction(unittest.TestCase):
    def setUp(self):
        self.client = MagicMock(spec=CoCClient)
        self.client.search_clans = AsyncMock()
        self.client.get_locations = AsyncMock(return_value=[])

    def test_successful_search_updates_history(self):
        """A successful search must add an entry to the history list."""
        returned_clans = [
            {"tag": "#ABC123", "name": "Clan A"},
            {"tag": "#XYZ789", "name": "Clan B"},
        ]
        self.client.search_clans.return_value = returned_clans

        history = []
        tags, results, new = asyncio.run(
            _perform_search(
                client=self.client,
                loc_id="locA",
                cfg={"min_members": 1},
                search_history=history,
            )
        )
        # basic checks on returned values
        self.assertEqual(tags, {"#ABC123", "#XYZ789"})
        self.assertEqual(len(history), 1)
        entry = history[0]
        self.assertEqual(entry["location_id"], "locA")
        self.assertEqual(entry["filters"], {"min_members": 1})
        self.assertIn("used_at", entry)
        self.assertEqual(entry["results"], 2)
        # new_clans depends on _raw_file_exists; we can't control disk here,
        # but we just confirm the field exists.
        self.assertIn("new_clans", entry)

    def test_failed_search_does_not_update_history(self):
        """If the API call raises, the history list must remain unchanged."""
        self.client.search_clans.side_effect = Exception("API down")

        history = []
        tags, results, new = asyncio.run(
            _perform_search(
                client=self.client,
                loc_id="locB",
                cfg={"min_members": 2},
                search_history=history,
            )
        )
        self.assertEqual(tags, set())
        self.assertEqual(len(history), 0)

    # ---------- tests for location resolution ----------
    def test_resolve_location_ids_converts_global_to_none(self):
        """'global' must become None to signal a worldwide search (no locationId)."""
        locations = [{"id": 32000000, "name": "Global"}]
        self.client.get_locations.return_value = locations
        raw = ["global", "32000006"]
        resolved = asyncio.run(_resolve_location_ids(self.client, raw))
        self.assertEqual(resolved, [None, "32000006"])

    def test_resolve_location_ids_preserves_numeric(self):
        """Already numeric location IDs must stay unchanged."""
        locations = [{"id": 32000001, "name": "Test"}]
        self.client.get_locations.return_value = locations
        raw = ["32000006", "32000001"]
        resolved = asyncio.run(_resolve_location_ids(self.client, raw))
        self.assertEqual(resolved, ["32000006", "32000001"])

    def test_resolve_location_ids_fallback_if_no_api(self):
        """If the API call fails, fallback must handle 'global' (return None)."""
        self.client.get_locations.side_effect = Exception("network error")
        raw = ["global", "32000006"]
        resolved = asyncio.run(_resolve_location_ids(self.client, raw))
        # 'global' -> None, numeric stays numeric
        self.assertIn(None, resolved)
        self.assertIn("32000006", resolved)

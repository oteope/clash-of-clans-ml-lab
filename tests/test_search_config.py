import unittest
import datetime
from src.extraction.search_config import (
    _config_fingerprint,
    generate_search_configurations,
    select_unused_configs,
    _normalize_filters,
)


class TestSearchConfig(unittest.TestCase):
    def test_fingerprint_ordering_ignored(self):
        cfg1 = {"min_members": 2, "max_members": 10}
        cfg2 = {"max_members": 10, "min_members": 2}
        self.assertEqual(_config_fingerprint(cfg1), _config_fingerprint(cfg2))

    def test_generate_configs_uses_allowed_keys(self):
        configs = generate_search_configurations()
        self.assertGreater(len(configs), 0)
        allowed_keys = {
            "min_members",
            "max_members",
            "min_clan_level",
            "max_clan_level",
            "min_clan_points",
            "max_clan_points",
        }
        for cfg in configs:
            self.assertTrue(set(cfg.keys()).issubset(allowed_keys))
            # Ensure no camelCase keys appear
            self.assertNotIn("minMembers", cfg)
            self.assertNotIn("maxMembers", cfg)
            self.assertNotIn("minClanLevel", cfg)
            self.assertNotIn("maxClanLevel", cfg)
            self.assertNotIn("minClanPoints", cfg)
            self.assertNotIn("maxClanPoints", cfg)

    def test_select_dedups_identical_filters(self):
        pool = [{"min_members": 3}]
        history = [
            {
                "location_id": "loc",
                "filters": {"min_members": 3},
                "used_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        ]
        selected = select_unused_configs(
            "loc", history, pool, max_searches=5, cooldown_minutes=100000
        )
        self.assertEqual(len(selected), 0)

    def test_select_returns_only_unused(self):
        pool = [
            {"min_members": 2},
            {"min_members": 3},
        ]
        history = [
            {
                "location_id": "X",
                "filters": {"min_members": 2},
                "used_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        ]
        selected = select_unused_configs(
            "X", history, pool, max_searches=3, cooldown_minutes=100000
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0], {"min_members": 3})

    def test_select_respects_cooldown(self):
        pool = [{"min_members": 2}]
        now = datetime.datetime.now(datetime.timezone.utc)
        # entry used a long time ago (more than cooldown)
        old_time = now - datetime.timedelta(minutes=120)
        history = [
            {
                "location_id": "loc",
                "filters": {"min_members": 2},
                "used_at": old_time.isoformat(),
            }
        ]
        selected = select_unused_configs(
            "loc", history, pool, max_searches=5, cooldown_minutes=60
        )
        self.assertEqual(len(selected), 1)  # eligible again after cooldown

    def test_select_empty_if_exhausted(self):
        pool = [{"min_members": 2}]
        history = [
            {
                "location_id": "loc",
                "filters": {"min_members": 2},
                "used_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        ]
        selected = select_unused_configs(
            "loc", history, pool, max_searches=10, cooldown_minutes=99999
        )
        self.assertEqual(len(selected), 0)

    # --- New tests for snake_case and backward compatibility ---
    def test_config_keys_snake_case(self):
        """All generated configurations must use snake_case, no camelCase."""
        configs = generate_search_configurations()
        for cfg in configs:
            for key in cfg:
                self.assertFalse(
                    any(c.isupper() for c in key),
                    f"Unexpected uppercase in key '{key}'",
                )

    def test_normalize_old_camel_case_filter(self):
        old_filters = {"minMembers": 5, "maxClanLevel": 12}
        normalized = _normalize_filters(old_filters)
        self.assertEqual(normalized, {"min_members": 5, "max_clan_level": 12})

    def test_normalize_preserves_unknown_keys(self):
        filters = {"unknownKey": 42, "minMembers": 3}
        normalized = _normalize_filters(filters)
        self.assertIn("unknownKey", normalized)
        self.assertIn("min_members", normalized)
        self.assertNotIn("minMembers", normalized)

    def test_select_ignores_camel_case_history(self):
        """History entries with camelCase filters should be treated as if
        they were the corresponding snake_case entry."""
        pool = [{"min_members": 2, "max_members": 10}]
        history = [
            {
                "location_id": "loc",
                "filters": {"minMembers": 2, "maxMembers": 10},
                "used_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        ]
        selected = select_unused_configs(
            "loc", history, pool, max_searches=5, cooldown_minutes=100000
        )
        self.assertEqual(len(selected), 0)

    # --- New tests for API minimum values ---
    def test_generated_level_range_starts_at_two(self):
        configs = generate_search_configurations()
        for cfg in configs:
            if "min_clan_level" in cfg:
                self.assertGreaterEqual(
                    cfg["min_clan_level"],
                    2,
                    f"Config with min_clan_level below minimum: {cfg}",
                )

    def test_generated_points_range_starts_at_one(self):
        configs = generate_search_configurations()
        for cfg in configs:
            if "min_clan_points" in cfg:
                self.assertGreaterEqual(
                    cfg["min_clan_points"],
                    1,
                    f"Config with min_clan_points below minimum: {cfg}",
                )

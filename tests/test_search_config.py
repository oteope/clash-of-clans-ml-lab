import unittest
import datetime
from src.extraction.search_config import (
    _config_fingerprint,
    generate_search_configurations,
    select_unused_configs,
)


class TestSearchConfig(unittest.TestCase):
    def test_fingerprint_ordering_ignored(self):
        cfg1 = {"minMembers": 1, "maxMembers": 10}
        cfg2 = {"maxMembers": 10, "minMembers": 1}
        self.assertEqual(_config_fingerprint(cfg1), _config_fingerprint(cfg2))

    def test_generate_configs_uses_allowed_keys(self):
        configs = generate_search_configurations()
        self.assertGreater(len(configs), 0)
        allowed_keys = {
            "minMembers",
            "maxMembers",
            "minClanLevel",
            "maxClanLevel",
            "minClanPoints",
            "maxClanPoints",
        }
        for cfg in configs:
            self.assertTrue(set(cfg.keys()).issubset(allowed_keys))

    def test_select_dedups_identical_filters(self):
        pool = [{"minMembers": 3}]
        history = [
            {
                "location_id": "loc",
                "filters": {"minMembers": 3},
                "used_at": datetime.datetime.utcnow().isoformat(),
            }
        ]
        selected = select_unused_configs(
            "loc", history, pool, max_searches=5, cooldown_minutes=100000
        )
        self.assertEqual(len(selected), 0)

    def test_select_returns_only_unused(self):
        pool = [
            {"minMembers": 1},
            {"minMembers": 2},
        ]
        history = [
            {
                "location_id": "X",
                "filters": {"minMembers": 1},
                "used_at": datetime.datetime.utcnow().isoformat(),
            }
        ]
        selected = select_unused_configs(
            "X", history, pool, max_searches=3, cooldown_minutes=100000
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0], {"minMembers": 2})

    def test_select_respects_cooldown(self):
        pool = [{"minMembers": 1}]
        now = datetime.datetime.utcnow()
        # entry used a long time ago (more than cooldown)
        old_time = now - datetime.timedelta(minutes=120)
        history = [
            {
                "location_id": "loc",
                "filters": {"minMembers": 1},
                "used_at": old_time.isoformat(),
            }
        ]
        selected = select_unused_configs(
            "loc", history, pool, max_searches=5, cooldown_minutes=60
        )
        self.assertEqual(len(selected), 1)  # eligible again after cooldown

    def test_select_empty_if_exhausted(self):
        pool = [{"minMembers": 1}]
        history = [
            {
                "location_id": "loc",
                "filters": {"minMembers": 1},
                "used_at": datetime.datetime.utcnow().isoformat(),
            }
        ]
        selected = select_unused_configs(
            "loc", history, pool, max_searches=10, cooldown_minutes=99999
        )
        self.assertEqual(len(selected), 0)

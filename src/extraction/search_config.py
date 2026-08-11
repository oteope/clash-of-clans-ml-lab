"""
Search configuration management for diversified clan discovery.

Generates a pool of filter combinations, tracks which have been used per
location, and selects unused ones in each run.
"""
import datetime
import json
import logging
import pathlib

from src.extraction.config import SEARCH_HISTORY_PATH, SEARCHES_PER_REGION_PER_RUN, SEARCH_COOLDOWN_MINUTES

logger = logging.getLogger(__name__)


def _config_fingerprint(filters: dict) -> frozenset:
    """Return an order‑independent fingerprint for a filter dictionary."""
    return frozenset((k, v) for k, v in filters.items() if v is not None)


# ---------------------------------------------------------------------------
# Configuration pool
# ---------------------------------------------------------------------------
def generate_search_configurations() -> list[dict]:
    """
    Build a list of different filter combinations (member ranges, clan level
    ranges, clan points ranges) that will be used for varied clan searches.

    Only pairwise combinations are generated to keep the pool manageable.
    """
    member_ranges = [
        {"minMembers": 1, "maxMembers": 10},
        {"minMembers": 11, "maxMembers": 20},
        {"minMembers": 21, "maxMembers": 30},
        {"minMembers": 31, "maxMembers": 40},
        {"minMembers": 41, "maxMembers": 50},
    ]
    level_ranges = [
        {"minClanLevel": 1, "maxClanLevel": 5},
        {"minClanLevel": 6, "maxClanLevel": 10},
        {"minClanLevel": 11, "maxClanLevel": 15},
        {"minClanLevel": 16, "maxClanLevel": 20},
    ]
    points_ranges = [
        {"minClanPoints": 0, "maxClanPoints": 1000},
        {"minClanPoints": 1001, "maxClanPoints": 3000},
        {"minClanPoints": 3001, "maxClanPoints": 5000},
        {"minClanPoints": 5001, "maxClanPoints": 10000},
        {"minClanPoints": 10001, "maxClanPoints": 40000},
        {"minClanPoints": 40001, "maxClanPoints": 999999},
    ]

    configs: list[dict] = []

    # Single‑dimension filters
    configs.extend(member_ranges)
    configs.extend(level_ranges)
    configs.extend(points_ranges)

    # Pairwise combinations
    for mb in member_ranges:
        for lv in level_ranges:
            configs.append({**mb, **lv})
    for mb in member_ranges:
        for pt in points_ranges:
            configs.append({**mb, **pt})
    for lv in level_ranges:
        for pt in points_ranges:
            configs.append({**lv, **pt})

    return configs


# ---------------------------------------------------------------------------
# Persistence of search history
# ---------------------------------------------------------------------------
def _history_path() -> pathlib.Path:
    return SEARCH_HISTORY_PATH


def load_search_history() -> list[dict]:
    """Load the list of previously executed searches from disk."""
    path = _history_path()
    if not path.exists():
        logger.debug("No search history file found at %s, starting fresh.", path)
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_search_history(history: list[dict]) -> None:
    """Persist the search history list to disk."""
    path = _history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    logger.info("Search history saved to %s", path)


# ---------------------------------------------------------------------------
# Selection logic
# ---------------------------------------------------------------------------
def select_unused_configs(
    location_id: str,
    history: list[dict],
    pool: list[dict],
    max_searches: int = SEARCHES_PER_REGION_PER_RUN,
    cooldown_minutes: int = SEARCH_COOLDOWN_MINUTES,
) -> list[dict]:
    """
    Return up to *max_searches* configurations from *pool* that have not been
    used for *location_id* in the recent *cooldown_minutes*.

    A configuration is considered used when a history entry exists for the
    same *location_id* and the same fingerprint that was recorded within the
    cooldown window.
    """
    now = datetime.datetime.utcnow()
    cooldown = datetime.timedelta(minutes=cooldown_minutes)

    used_fingerprints: set[frozenset] = set()
    for entry in history:
        if entry.get("location_id") != location_id:
            continue
        used_at_str = entry.get("used_at")
        if not used_at_str:
            continue
        try:
            used_time = datetime.datetime.fromisoformat(used_at_str)
        except (ValueError, TypeError):
            continue
        if now - used_time < cooldown:
            fp = _config_fingerprint(entry.get("filters", {}))
            used_fingerprints.add(fp)

    selected: list[dict] = []
    for cfg in pool:
        if _config_fingerprint(cfg) not in used_fingerprints:
            selected.append(cfg)
            if len(selected) >= max_searches:
                break
    return selected

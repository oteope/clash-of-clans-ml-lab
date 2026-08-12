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


# ---------------------------------------------------------------------------
# Helpers for normalising filter keys (camelCase -> snake_case)
# ---------------------------------------------------------------------------
_CAMEL_TO_SNAKE = {
    "minMembers": "min_members",
    "maxMembers": "max_members",
    "minClanLevel": "min_clan_level",
    "maxClanLevel": "max_clan_level",
    "minClanPoints": "min_clan_points",
    "maxClanPoints": "max_clan_points",
}


def _normalize_filters(filters: dict) -> dict:
    """Convert camelCase filter keys to snake_case, keeping unknown keys unchanged."""
    return {_CAMEL_TO_SNAKE.get(k, k): v for k, v in filters.items()}


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
    # The Clash of Clans API enforces the following minima:
    #   minMembers >= 2
    #   minClanLevel >= 2
    #   minClanPoints >= 1
    member_ranges = [
        {"min_members": 2, "max_members": 10},
        {"min_members": 11, "max_members": 20},
        {"min_members": 21, "max_members": 30},
        {"min_members": 31, "max_members": 40},
        {"min_members": 41, "max_members": 50},
    ]
    level_ranges = [
        {"min_clan_level": 2, "max_clan_level": 5},
        {"min_clan_level": 6, "max_clan_level": 10},
        {"min_clan_level": 11, "max_clan_level": 15},
        {"min_clan_level": 16, "max_clan_level": 20},
    ]
    points_ranges = [
        {"min_clan_points": 1, "max_clan_points": 1000},
        {"min_clan_points": 1001, "max_clan_points": 3000},
        {"min_clan_points": 3001, "max_clan_points": 5000},
        {"min_clan_points": 5001, "max_clan_points": 10000},
        {"min_clan_points": 10001, "max_clan_points": 40000},
        {"min_clan_points": 40001, "max_clan_points": 999999},
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
    """Load the list of previously executed searches from disk,
    normalising any old camelCase filter keys."""
    path = _history_path()
    if not path.exists():
        logger.debug("No search history file found at %s, starting fresh.", path)
        return []
    with path.open("r", encoding="utf-8") as f:
        history: list[dict] = json.load(f)

    # Normalise filter keys in every entry
    for entry in history:
        entry["filters"] = _normalize_filters(entry.get("filters", {}))
    return history


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
    now = datetime.datetime.now(datetime.timezone.utc)
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
            if used_time.tzinfo is None:
                used_time = used_time.replace(tzinfo=datetime.timezone.utc)
            else:
                used_time = used_time.astimezone(datetime.timezone.utc)
        except (ValueError, TypeError):
            continue

        if now - used_time < cooldown:
            # Normalize filter keys before computing fingerprint to handle
            # old history entries that may still contain camelCase keys.
            filters_raw = entry.get("filters", {})
            if filters_raw:
                filters_raw = _normalize_filters(filters_raw)
            fp = _config_fingerprint(filters_raw)
            used_fingerprints.add(fp)

    selected: list[dict] = []
    for cfg in pool:
        if _config_fingerprint(cfg) not in used_fingerprints:
            selected.append(cfg)
            if len(selected) >= max_searches:
                break
    return selected

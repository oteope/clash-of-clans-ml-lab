import json
import logging
import pathlib
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

RAW_BASE = pathlib.Path("data/raw")
PROCESSED_BASE = pathlib.Path("data/processed")


def load_json(filepath: pathlib.Path) -> Optional[dict]:
    """Safely load a JSON file, returning None on failure."""
    try:
        with filepath.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to load %s: %s", filepath, exc)
        return None


# ----------------------------------------------------------------------
# Normalisation helpers – each returns a dict (or list of dicts) with
# snake_case keys and None for missing fields.
# ----------------------------------------------------------------------

def normalize_clan(raw: dict) -> dict:
    """Convert a raw clan overview JSON into a flat clan row."""
    location = raw.get("location") or {}
    return {
        "clan_tag": raw.get("tag"),
        "name": raw.get("name"),
        "description": raw.get("description"),
        "clan_level": raw.get("clanLevel"),
        "clan_points": raw.get("clanPoints"),
        "clan_builder_base_points": raw.get("clanBuilderBasePoints"),
        "clan_capital_points": raw.get("clanCapitalPoints"),
        "members": raw.get("members"),
        "required_trophies": raw.get("requiredTrophies"),
        "war_frequency": raw.get("warFrequency"),
        "war_win_streak": raw.get("warWinStreak"),
        "war_wins": raw.get("warWins"),
        "war_ties": raw.get("warTies"),
        "war_losses": raw.get("warLosses"),
        "war_log_public": raw.get("warLogPublic"),
        "war_league": raw.get("warLeague"),
        "capital_league": raw.get("capitalLeague"),
        "location_id": location.get("id"),
        "location_name": location.get("name"),
        "type": raw.get("type"),
        "is_family_friendly": raw.get("isFamilyFriendly"),
    }


def normalize_member(member: dict, clan_tag: str) -> dict:
    """Convert a single member entry (from /clans/{tag}/members) into a row."""
    league = member.get("league") or {}
    return {
        "clan_tag": clan_tag,
        "player_tag": member.get("tag"),
        "name": member.get("name"),
        "role": member.get("role"),
        "town_hall_level": member.get("townHallLevel"),
        "exp_level": member.get("expLevel"),
        "league_id": league.get("id"),
        "league_name": league.get("name"),
        "league_tier_id": league.get("tierId"),
        "league_tier_name": league.get("tierName"),
        "trophies": member.get("trophies"),
        "builder_base_trophies": member.get("builderBaseTrophies"),
        "clan_rank": member.get("clanRank"),
        "previous_clan_rank": member.get("previousClanRank"),
        "donations": member.get("donations"),
        "donations_received": member.get("donationsReceived"),
    }


def normalize_player(raw: dict) -> dict:
    """Convert a full player JSON into one row."""
    clan = raw.get("clan") or {}
    return {
        "player_tag": raw.get("tag"),
        "name": raw.get("name"),
        "town_hall_level": raw.get("townHallLevel"),
        "exp_level": raw.get("expLevel"),
        "trophies": raw.get("trophies"),
        "best_trophies": raw.get("bestTrophies"),
        "war_stars": raw.get("warStars"),
        "attack_wins": raw.get("attackWins"),
        "defense_wins": raw.get("defenseWins"),
        "builder_hall_level": raw.get("builderHallLevel"),
        "builder_base_trophies": raw.get("builderBaseTrophies"),
        "best_builder_base_trophies": raw.get("bestBuilderBaseTrophies"),
        "donations": raw.get("donations"),
        "donations_received": raw.get("donationsReceived"),
        "clan_capital_contributions": raw.get("clanCapitalContributions"),
        "clan_tag": clan.get("tag"),
        "clan_name": clan.get("name"),
        "clan_level": clan.get("clanLevel"),
    }


# ----------------------------------------------------------------------
# List normalisers – each returns a list of flat dicts.
# ----------------------------------------------------------------------

def normalize_troops(player_tag: str, troops: Optional[list]) -> list:
    rows = []
    if not troops:
        return rows
    for t in troops:
        rows.append({
            "player_tag": player_tag,
            "troop_name": t.get("name"),
            "level": t.get("level"),
            "max_level": t.get("maxLevel"),
            "village": t.get("village"),
        })
    return rows


def normalize_heroes(player_tag: str, heroes: Optional[list]) -> list:
    rows = []
    if not heroes:
        return rows
    for h in heroes:
        rows.append({
            "player_tag": player_tag,
            "hero_name": h.get("name"),
            "level": h.get("level"),
            "max_level": h.get("maxLevel"),
            "village": h.get("village"),
        })
    return rows


def normalize_hero_equipment(player_tag: str, equipment_list: Optional[list]) -> list:
    rows = []
    if not equipment_list:
        return rows
    for eq in equipment_list:
        rows.append({
            "player_tag": player_tag,
            "equipment_name": eq.get("name"),
            "level": eq.get("level"),
            "max_level": eq.get("maxLevel"),
            "village": eq.get("village"),
        })
    return rows


def normalize_spells(player_tag: str, spells: Optional[list]) -> list:
    rows = []
    if not spells:
        return rows
    for s in spells:
        rows.append({
            "player_tag": player_tag,
            "spell_name": s.get("name"),
            "level": s.get("level"),
            "max_level": s.get("maxLevel"),
            "village": s.get("village"),
        })
    return rows


def normalize_achievements(player_tag: str, achievements: Optional[list]) -> list:
    rows = []
    if not achievements:
        return rows
    for a in achievements:
        rows.append({
            "player_tag": player_tag,
            "achievement_name": a.get("name"),
            "stars": a.get("stars"),
            "value": a.get("value"),
            "target": a.get("target"),
            "village": a.get("village"),
        })
    return rows


# ----------------------------------------------------------------------
# Deduplication helper
# ----------------------------------------------------------------------

def _deduplicate_dataframe(df: pd.DataFrame, keys: list) -> pd.DataFrame:
    """Drop duplicate rows based on the given key columns, keeping the first occurrence."""
    return df.drop_duplicates(subset=keys, keep="first")


# ----------------------------------------------------------------------
# Main pipeline entry-point
# ----------------------------------------------------------------------

def build_all_tables() -> None:
    """Process all raw JSON files and write normalized Parquet tables."""
    logger.info("Starting normalized table build...")
    PROCESSED_BASE.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Clans
    # ------------------------------------------------------------------
    clans_dir = RAW_BASE / "clans"
    clan_rows: list = []
    clan_tags_seen: Set[str] = set()

    if clans_dir.exists():
        for fpath in sorted(clans_dir.glob("*.json")):
            raw = load_json(fpath)
            if raw is None:
                continue
            row = normalize_clan(raw)
            tag = row["clan_tag"]
            if tag and tag not in clan_tags_seen:
                clan_tags_seen.add(tag)
                clan_rows.append(row)
            else:
                logger.debug("Skipping duplicate clan tag %s", tag)

    if clan_rows:
        df_clans = pd.DataFrame(clan_rows)
        df_clans = _deduplicate_dataframe(df_clans, ["clan_tag"])
        df_clans.to_parquet(PROCESSED_BASE / "clans.parquet", index=False)
        logger.info("Written %d clans.", len(df_clans))
    else:
        logger.info("No clan data found.")

    # ------------------------------------------------------------------
    # 2. Clan members (relationship table)
    # ------------------------------------------------------------------
    members_dir = RAW_BASE / "members"
    member_rows: list = []
    member_keys: Set[Tuple[str, str]] = set()

    if members_dir.exists():
        for fpath in sorted(members_dir.glob("*.json")):
            clan_tag = fpath.stem   # filename without extension
            raw = load_json(fpath)
            if raw is None:
                continue
            items = raw.get("items", [])
            for member in items:
                player_tag = member.get("tag")
                if not player_tag:
                    continue
                key = (clan_tag, player_tag)
                if key not in member_keys:
                    member_keys.add(key)
                    member_rows.append(normalize_member(member, clan_tag))

    if member_rows:
        df_members = pd.DataFrame(member_rows)
        df_members = _deduplicate_dataframe(df_members, ["clan_tag", "player_tag"])
        df_members.to_parquet(PROCESSED_BASE / "clan_members.parquet", index=False)
        logger.info("Written %d clan-member relationships.", len(df_members))
    else:
        logger.info("No members data found.")

    # ------------------------------------------------------------------
    # 3. Players + nested lists
    # ------------------------------------------------------------------
    players_dir = RAW_BASE / "players"
    player_rows: list = []
    player_tags_seen: Set[str] = set()

    troops_rows: list = []
    heroes_rows: list = []
    equipment_rows: list = []
    spells_rows: list = []
    achievements_rows: list = []

    if players_dir.exists():
        for fpath in sorted(players_dir.glob("*.json")):
            raw = load_json(fpath)
            if raw is None:
                continue
            row = normalize_player(raw)
            tag = row["player_tag"]
            if not tag or tag in player_tags_seen:
                logger.debug("Skipping duplicate player %s", tag)
                continue

            player_tags_seen.add(tag)
            player_rows.append(row)

            troops_rows.extend(normalize_troops(tag, raw.get("troops")))
            heroes_rows.extend(normalize_heroes(tag, raw.get("heroes")))

            # The player JSON may store hero equipment under "heroEquipment" or "equipment".
            eq_list = raw.get("heroEquipment") or raw.get("equipment")
            equipment_rows.extend(normalize_hero_equipment(tag, eq_list))

            spells_rows.extend(normalize_spells(tag, raw.get("spells")))
            achievements_rows.extend(normalize_achievements(tag, raw.get("achievements")))

    if player_rows:
        df_players = pd.DataFrame(player_rows)
        df_players = _deduplicate_dataframe(df_players, ["player_tag"])
        df_players.to_parquet(PROCESSED_BASE / "players.parquet", index=False)
        logger.info("Written %d players.", len(df_players))
    else:
        logger.info("No player data found.")

    # Helper to write and dedup nested tables
    def _write_nested(rows: list, filename: str, keys: list) -> None:
        if not rows:
            logger.info("No data for %s.", filename)
            return
        df = pd.DataFrame(rows)
        df = _deduplicate_dataframe(df, keys)
        df.to_parquet(PROCESSED_BASE / filename, index=False)
        logger.info("Written %d rows to %s.", len(df), filename)

    _write_nested(troops_rows, "player_troops.parquet", ["player_tag", "troop_name", "village"])
    _write_nested(heroes_rows, "player_heroes.parquet", ["player_tag", "hero_name", "village"])
    _write_nested(equipment_rows, "player_hero_equipment.parquet",
                  ["player_tag", "equipment_name", "village"])
    _write_nested(spells_rows, "player_spells.parquet", ["player_tag", "spell_name", "village"])
    _write_nested(achievements_rows, "player_achievements.parquet",
                  ["player_tag", "achievement_name"])

    logger.info("Normalized table build complete.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    build_all_tables()

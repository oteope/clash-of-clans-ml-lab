import json
import logging
import pathlib
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

RAW_BASE = pathlib.Path("data/raw")
PROCESSED_BASE = pathlib.Path("data/processed")

CHUNK_SIZE = 10_000  # Number of rows collected in memory before writing a Parquet batch


class _ParquetBatchWriter:
    """Writes incremental batches to a single Parquet file using pyarrow.

    The schema is determined by the first batch that actually writes rows.
    Subsequent batches are aligned to this schema (missing columns filled with
    ``None`` and extra columns ignored).
    """

    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        self.writer: Optional[pq.ParquetWriter] = None
        self.schema: Optional[pa.Schema] = None
        self.columns: Optional[List[str]] = None

    def write_batch(self, rows: List[dict]) -> None:
        if not rows:
            return

        df = pd.DataFrame(rows)

        if self.writer is None:
            # First write – open writer and lock schema
            table = pa.Table.from_pandas(df)
            self.schema = table.schema
            self.columns = self.schema.names
            self.writer = pq.ParquetWriter(self.path, self.schema)
            self.writer.write_table(table)
            return

        # Subsequent writes – align columns to the frozen schema
        for col in self.columns:
            if col not in df.columns:
                df[col] = None
        df = df[self.columns]
        table = pa.Table.from_pandas(df)
        self.writer.write_table(table)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


# ----------------------------------------------------------------------
# Normalisation helpers – each returns a dict (or list of dicts) with
# snake_case keys and None for missing fields.
# ----------------------------------------------------------------------

def load_json(filepath: pathlib.Path) -> Optional[dict]:
    """Safely load a JSON file, returning None on failure."""
    try:
        with filepath.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to load %s: %s", filepath, exc)
        return None


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
    league_tier = member.get("leagueTier") or {}
    return {
        "clan_tag": clan_tag,
        "player_tag": member.get("tag"),
        "name": member.get("name"),
        "role": member.get("role"),
        "town_hall_level": member.get("townHallLevel"),
        "exp_level": member.get("expLevel"),
        "league_id": league.get("id"),
        "league_name": league.get("name"),
        "league_tier_id": league_tier.get("id"),
        "league_tier_name": league_tier.get("name"),
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
# Deduplication helper (kept for testing / future use)
# ----------------------------------------------------------------------

def _deduplicate_dataframe(df: pd.DataFrame, keys: list) -> pd.DataFrame:
    """Drop duplicate rows based on the given key columns, keeping the first occurrence."""
    return df.drop_duplicates(subset=keys, keep="first")


# ----------------------------------------------------------------------
# Main pipeline entry-point  (batch / streaming version)
# ----------------------------------------------------------------------

def build_all_tables() -> None:
    """Process all raw JSON files and write normalized Parquet tables.

    Data is processed in chunks to keep memory usage bounded even when
    there are millions of raw records.
    """
    logger.info("Starting normalized table build...")
    PROCESSED_BASE.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Clans
    # ------------------------------------------------------------------
    clans_dir = RAW_BASE / "clans"
    clan_rows: List[dict] = []
    clan_tags_seen: Set[str] = set()
    clan_writer = _ParquetBatchWriter(PROCESSED_BASE / "clans.parquet")

    if clans_dir.exists():
        for fpath in sorted(clans_dir.glob("*.json")):
            raw = load_json(fpath)
            if raw is None:
                continue
            row = normalize_clan(raw)
            tag = row.get("clan_tag")
            if not tag or tag in clan_tags_seen:
                logger.debug("Skipping duplicate clan tag %s", tag)
                continue
            clan_tags_seen.add(tag)
            clan_rows.append(row)
            if len(clan_rows) >= CHUNK_SIZE:
                clan_writer.write_batch(clan_rows)
                clan_rows.clear()

        # flush remaining
        clan_writer.write_batch(clan_rows)
        clan_writer.close()
        logger.info("Written %d clans.", len(clan_tags_seen))
    else:
        logger.info("No clan data found.")
        clan_writer.close()

    # ------------------------------------------------------------------
    # 2. Clan members (relationship table)
    # ------------------------------------------------------------------
    members_dir = RAW_BASE / "members"
    member_rows: List[dict] = []
    member_keys: Set[Tuple[str, str]] = set()
    member_writer = _ParquetBatchWriter(PROCESSED_BASE / "clan_members.parquet")

    if members_dir.exists():
        for fpath in sorted(members_dir.glob("*.json")):
            clan_tag = fpath.stem   # file name without extension
            raw = load_json(fpath)
            if raw is None:
                continue
            items = raw.get("items", [])
            for member in items:
                player_tag = member.get("tag")
                if not player_tag:
                    continue
                key = (clan_tag, player_tag)
                if key in member_keys:
                    continue
                member_keys.add(key)
                member_rows.append(normalize_member(member, clan_tag))
                if len(member_rows) >= CHUNK_SIZE:
                    member_writer.write_batch(member_rows)
                    member_rows.clear()

        member_writer.write_batch(member_rows)
        member_writer.close()
        logger.info("Written %d clan-member relationships.", len(member_keys))
    else:
        logger.info("No members data found.")
        member_writer.close()

    # ------------------------------------------------------------------
    # 3. Players + nested lists
    # ------------------------------------------------------------------
    players_dir = RAW_BASE / "players"
    player_rows: List[dict] = []
    player_tags_seen: Set[str] = set()
    player_writer = _ParquetBatchWriter(PROCESSED_BASE / "players.parquet")

    troops_rows: List[dict] = []
    heroes_rows: List[dict] = []
    equipment_rows: List[dict] = []
    spells_rows: List[dict] = []
    achievements_rows: List[dict] = []

    troops_writer = _ParquetBatchWriter(PROCESSED_BASE / "player_troops.parquet")
    heroes_writer = _ParquetBatchWriter(PROCESSED_BASE / "player_heroes.parquet")
    equipment_writer = _ParquetBatchWriter(PROCESSED_BASE / "player_hero_equipment.parquet")
    spells_writer = _ParquetBatchWriter(PROCESSED_BASE / "player_spells.parquet")
    achievements_writer = _ParquetBatchWriter(PROCESSED_BASE / "player_achievements.parquet")

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
            if len(player_rows) >= CHUNK_SIZE:
                player_writer.write_batch(player_rows)
                player_rows.clear()

            # nested lists – accumulate and flush per table
            new_troops = normalize_troops(tag, raw.get("troops"))
            troops_rows.extend(new_troops)
            if len(troops_rows) >= CHUNK_SIZE:
                troops_writer.write_batch(troops_rows)
                troops_rows.clear()

            new_heroes = normalize_heroes(tag, raw.get("heroes"))
            heroes_rows.extend(new_heroes)
            if len(heroes_rows) >= CHUNK_SIZE:
                heroes_writer.write_batch(heroes_rows)
                heroes_rows.clear()

            eq_list = raw.get("heroEquipment")   # only the real field
            new_equipment = normalize_hero_equipment(tag, eq_list)
            equipment_rows.extend(new_equipment)
            if len(equipment_rows) >= CHUNK_SIZE:
                equipment_writer.write_batch(equipment_rows)
                equipment_rows.clear()

            new_spells = normalize_spells(tag, raw.get("spells"))
            spells_rows.extend(new_spells)
            if len(spells_rows) >= CHUNK_SIZE:
                spells_writer.write_batch(spells_rows)
                spells_rows.clear()

            new_ach = normalize_achievements(tag, raw.get("achievements"))
            achievements_rows.extend(new_ach)
            if len(achievements_rows) >= CHUNK_SIZE:
                achievements_writer.write_batch(achievements_rows)
                achievements_rows.clear()

        # flush remaining for all player-related tables
        player_writer.write_batch(player_rows)
        player_writer.close()

        troops_writer.write_batch(troops_rows)
        troops_writer.close()

        heroes_writer.write_batch(heroes_rows)
        heroes_writer.close()

        equipment_writer.write_batch(equipment_rows)
        equipment_writer.close()

        spells_writer.write_batch(spells_rows)
        spells_writer.close()

        achievements_writer.write_batch(achievements_rows)
        achievements_writer.close()

        logger.info("Written %d players.", len(player_tags_seen))
    else:
        logger.info("No player data found.")
        player_writer.close()
        troops_writer.close()
        heroes_writer.close()
        equipment_writer.close()
        spells_writer.close()
        achievements_writer.close()

    logger.info("Normalized table build complete.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    build_all_tables()

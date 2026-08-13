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


def _safe_json_dumps(value: Any) -> Optional[str]:
    """Convert a nested dict/list to a JSON string, or None for missing/invalid data."""
    if value is None:
        return None
    try:
        return json.dumps(value, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _clear_processed_outputs(processed_base: pathlib.Path) -> None:
    """Remove all previous *.parquet output files from the processed directory.

    This ensures that a rebuild starts from a clean state without touching raw data.
    """
    if not processed_base.exists():
        return
    for file_path in processed_base.glob("*.parquet"):
        try:
            file_path.unlink()
        except OSError as exc:
            logger.warning("No se pudo eliminar %s: %s", file_path, exc)


class _ParquetBatchWriter:
    """Writes incremental batches to a single Parquet file using pyarrow.

    The schema is determined either by an explicitly provided ``schema`` or
    (if none is supplied) by the first batch that actually writes rows.
    Subsequent batches are aligned to this schema and cast as needed.
    """

    def __init__(self, path: pathlib.Path, schema: Optional[pa.Schema] = None) -> None:
        self.path = path
        self.writer: Optional[pq.ParquetWriter] = None
        self.schema = schema
        self.columns: Optional[List[str]] = schema.names if schema is not None else None

    def write_batch(self, rows: List[dict]) -> None:
        if not rows:
            return

        df = pd.DataFrame(rows)

        if self.schema is not None:
            # Force exact schema columns and order; fill missing with None
            for col in self.schema.names:
                if col not in df.columns:
                    df[col] = None
            df = df[self.schema.names]

            if self.writer is None:
                self.writer = pq.ParquetWriter(self.path, self.schema)

            try:
                table = pa.Table.from_pandas(df, schema=self.schema, preserve_index=False)
            except Exception as exc:
                logger.debug(
                    "No se pudo crear la tabla con el esquema explícito para %s: %s. "
                    "Se intentará inferir y cast.",
                    self.path,
                    exc,
                )
                table = pa.Table.from_pandas(df)
                table = table.cast(self.schema)

            self.writer.write_table(table)
            return

        # Legacy / no explicit schema path: infer from first batch, then align
        if self.writer is None:
            table = pa.Table.from_pandas(df)
            self.schema = table.schema
            self.columns = self.schema.names
            self.writer = pq.ParquetWriter(self.path, self.schema)
            self.writer.write_table(table)
            return

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
# Parquet schemas
# ----------------------------------------------------------------------

def _get_parquet_schemas() -> Dict[str, pa.Schema]:
    """Define explicit schemas for all output tables.

    Using explicit schemas prevents type drift between batches and ensures
    consistent columns even when the first batch has mostly missing values.
    """
    return {
        "clans": pa.schema([
            ("clan_tag", pa.string()),
            ("name", pa.string()),
            ("description", pa.string()),
            ("clan_level", pa.int64()),
            ("clan_points", pa.int64()),
            ("clan_builder_base_points", pa.int64()),
            ("clan_capital_points", pa.int64()),
            ("members", pa.int64()),
            ("required_trophies", pa.int64()),
            ("war_frequency", pa.string()),
            ("war_win_streak", pa.int64()),
            ("war_wins", pa.int64()),
            ("war_ties", pa.int64()),
            ("war_losses", pa.int64()),
            ("war_log_public", pa.bool_()),
            ("war_league", pa.string()),      # JSON serializado
            ("capital_league", pa.string()),  # JSON serializado
            ("location_id", pa.int64()),
            ("location_name", pa.string()),
            ("type", pa.string()),
            ("is_family_friendly", pa.bool_()),
        ]),
        "clan_members": pa.schema([
            ("clan_tag", pa.string()),
            ("player_tag", pa.string()),
            ("name", pa.string()),
            ("role", pa.string()),
            ("town_hall_level", pa.int64()),
            ("exp_level", pa.int64()),
            ("league_id", pa.int64()),
            ("league_name", pa.string()),
            ("league_tier_id", pa.int64()),
            ("league_tier_name", pa.string()),
            ("trophies", pa.int64()),
            ("builder_base_trophies", pa.int64()),
            ("clan_rank", pa.int64()),
            ("previous_clan_rank", pa.int64()),
            ("donations", pa.int64()),
            ("donations_received", pa.int64()),
            ("capital_contributions", pa.int64()),
        ]),
        "players": pa.schema([
            ("player_tag", pa.string()),
            ("name", pa.string()),
            ("town_hall_level", pa.int64()),
            ("exp_level", pa.int64()),
            ("trophies", pa.int64()),
            ("best_trophies", pa.int64()),
            ("war_stars", pa.int64()),
            ("attack_wins", pa.int64()),
            ("defense_wins", pa.int64()),
            ("builder_hall_level", pa.int64()),
            ("builder_base_trophies", pa.int64()),
            ("best_builder_base_trophies", pa.int64()),
            ("donations", pa.int64()),
            ("donations_received", pa.int64()),
            ("clan_capital_contributions", pa.int64()),
            ("clan_tag", pa.string()),
            ("clan_name", pa.string()),
            ("clan_level", pa.int64()),
        ]),
        "player_troops": pa.schema([
            ("player_tag", pa.string()),
            ("troop_name", pa.string()),
            ("level", pa.int64()),
            ("max_level", pa.int64()),
            ("village", pa.string()),
        ]),
        "player_heroes": pa.schema([
            ("player_tag", pa.string()),
            ("hero_name", pa.string()),
            ("level", pa.int64()),
            ("max_level", pa.int64()),
            ("village", pa.string()),
        ]),
        "player_hero_equipment": pa.schema([
            ("player_tag", pa.string()),
            ("equipment_name", pa.string()),
            ("level", pa.int64()),
            ("max_level", pa.int64()),
            ("village", pa.string()),
        ]),
        "player_spells": pa.schema([
            ("player_tag", pa.string()),
            ("spell_name", pa.string()),
            ("level", pa.int64()),
            ("max_level", pa.int64()),
            ("village", pa.string()),
        ]),
        "player_achievements": pa.schema([
            ("player_tag", pa.string()),
            ("achievement_name", pa.string()),
            ("stars", pa.int64()),
            ("value", pa.int64()),
            ("target", pa.int64()),
            ("village", pa.string()),
        ]),
    }


# ----------------------------------------------------------------------
# Normalisation helpers – each returns a dict (or list of dicts) with
# snake_case keys and None for missing fields.
# ----------------------------------------------------------------------

def load_json(filepath: pathlib.Path) -> Optional[dict]:
    """Safely load a JSON file, returning None on failure."""
    try:
        with filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning("JSON no dict en %s (tipo=%s)", filepath, type(data).__name__)
            return None
        return data
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
        "war_league": _safe_json_dumps(raw.get("warLeague")),
        "capital_league": _safe_json_dumps(raw.get("capitalLeague")),
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
        "capital_contributions": member.get("clanCapitalContributions"),
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

def _is_list_of_dicts(value: Any) -> bool:
    """Return True if value is a non-empty list of dict-like objects."""
    if not isinstance(value, list):
        return False
    return all(isinstance(item, dict) for item in value)


def normalize_troops(player_tag: str, troops: Optional[list]) -> list:
    rows = []
    if troops is None:
        return rows
    if not _is_list_of_dicts(troops):
        logger.warning("troops no es una lista de dicts para el jugador %s", player_tag)
        return rows
    for t in troops:
        if not isinstance(t, dict):
            continue
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
    if heroes is None:
        return rows
    if not _is_list_of_dicts(heroes):
        logger.warning("heroes no es una lista de dicts para el jugador %s", player_tag)
        return rows
    for h in heroes:
        if not isinstance(h, dict):
            continue
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
    if equipment_list is None:
        return rows
    if not _is_list_of_dicts(equipment_list):
        logger.warning("heroEquipment no es una lista de dicts para el jugador %s", player_tag)
        return rows
    for eq in equipment_list:
        if not isinstance(eq, dict):
            continue
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
    if spells is None:
        return rows
    if not _is_list_of_dicts(spells):
        logger.warning("spells no es una lista de dicts para el jugador %s", player_tag)
        return rows
    for s in spells:
        if not isinstance(s, dict):
            continue
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
    if achievements is None:
        return rows
    if not _is_list_of_dicts(achievements):
        logger.warning("achievements no es una lista de dicts para el jugador %s", player_tag)
        return rows
    for a in achievements:
        if not isinstance(a, dict):
            continue
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
# Relationship validation (non-destructive, only logs warnings)
# ----------------------------------------------------------------------

def _log_relationship_coverage(processed_base: pathlib.Path) -> None:
    """Log coverage warnings for foreign keys without discarding any row.

    Missing clan/player references are expected because raw data may
    have different coverage across extracted entities.
    """
    members_path = processed_base / "clan_members.parquet"
    clans_path = processed_base / "clans.parquet"
    players_path = processed_base / "players.parquet"

    if not members_path.exists():
        return

    member_clan_tags = set(
        pq.read_table(members_path, columns=["clan_tag"])["clan_tag"].to_pandas().dropna().unique()
    )
    member_player_tags = set(
        pq.read_table(members_path, columns=["player_tag"])["player_tag"].to_pandas().dropna().unique()
    )

    if clans_path.exists():
        clan_tags = set(
            pq.read_table(clans_path, columns=["clan_tag"])["clan_tag"].to_pandas().dropna().unique()
        )
        missing_clans = member_clan_tags - clan_tags
        if missing_clans:
            logger.warning(
                "clan_members contiene %d clan_tag(s) no presentes en clans.parquet",
                len(missing_clans),
            )
            logger.debug("Ejemplos: %s", list(missing_clans)[:5])
    else:
        logger.warning("clans.parquet no existe; no se puede validar clan_tag.")

    if players_path.exists():
        player_tags = set(
            pq.read_table(players_path, columns=["player_tag"])["player_tag"].to_pandas().dropna().unique()
        )
        missing_players = member_player_tags - player_tags
        if missing_players:
            logger.warning(
                "clan_members contiene %d player_tag(s) no presentes en players.parquet",
                len(missing_players),
            )
            logger.debug("Ejemplos: %s", list(missing_players)[:5])
    else:
        logger.warning("players.parquet no existe; no se puede validar player_tag.")


# ----------------------------------------------------------------------
# Main pipeline entry-point  (batch / streaming version)
# ----------------------------------------------------------------------

def build_all_tables(
    raw_base: pathlib.Path = RAW_BASE,
    processed_base: pathlib.Path = PROCESSED_BASE,
) -> None:
    """Process all raw JSON files and write normalized Parquet tables.

    Data is processed in chunks to keep memory usage bounded even when
    there are millions of raw records.
    """
    logger.info("Starting normalized table build...")
    processed_base.mkdir(parents=True, exist_ok=True)
    _clear_processed_outputs(processed_base)

    schemas = _get_parquet_schemas()

    # ------------------------------------------------------------------
    # 1. Clans
    # ------------------------------------------------------------------
    clans_dir = raw_base / "clans"
    clan_rows: List[dict] = []
    clan_tags_seen: Set[str] = set()
    clan_writer = _ParquetBatchWriter(processed_base / "clans.parquet", schema=schemas["clans"])

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

        clan_writer.write_batch(clan_rows)
        clan_writer.close()
        logger.info("Written %d clans.", len(clan_tags_seen))
    else:
        logger.info("No clan data found.")
        clan_writer.close()

    # ------------------------------------------------------------------
    # 2. Clan members (relationship table)
    # ------------------------------------------------------------------
    members_dir = raw_base / "members"
    member_rows: List[dict] = []
    member_keys: Set[Tuple[str, str]] = set()
    member_writer = _ParquetBatchWriter(processed_base / "clan_members.parquet", schema=schemas["clan_members"])

    if members_dir.exists():
        for fpath in sorted(members_dir.glob("*.json")):
            clan_tag = fpath.stem   # file name without extension
            raw = load_json(fpath)
            if raw is None:
                continue
            items = raw.get("items", [])
            if not isinstance(items, list):
                logger.warning("Formato inesperado en %s: items no es lista", fpath)
                continue
            for member in items:
                if not isinstance(member, dict):
                    continue
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
    players_dir = raw_base / "players"
    player_rows: List[dict] = []
    player_tags_seen: Set[str] = set()
    player_writer = _ParquetBatchWriter(processed_base / "players.parquet", schema=schemas["players"])

    troops_rows: List[dict] = []
    heroes_rows: List[dict] = []
    equipment_rows: List[dict] = []
    spells_rows: List[dict] = []
    achievements_rows: List[dict] = []

    troops_keys: Set[Tuple[str, str, Optional[str]]] = set()
    heroes_keys: Set[Tuple[str, str, Optional[str]]] = set()
    equipment_keys: Set[Tuple[str, str, Optional[str]]] = set()
    spells_keys: Set[Tuple[str, str, Optional[str]]] = set()
    achievements_keys: Set[Tuple[str, str, Optional[str]]] = set()

    troops_writer = _ParquetBatchWriter(processed_base / "player_troops.parquet", schema=schemas["player_troops"])
    heroes_writer = _ParquetBatchWriter(processed_base / "player_heroes.parquet", schema=schemas["player_heroes"])
    equipment_writer = _ParquetBatchWriter(
        processed_base / "player_hero_equipment.parquet",
        schema=schemas["player_hero_equipment"],
    )
    spells_writer = _ParquetBatchWriter(processed_base / "player_spells.parquet", schema=schemas["player_spells"])
    achievements_writer = _ParquetBatchWriter(
        processed_base / "player_achievements.parquet",
        schema=schemas["player_achievements"],
    )

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

            # nested lists – accumulate and flush per table, also deduplicate
            new_troops = normalize_troops(tag, raw.get("troops"))
            for troop in new_troops:
                key = (troop["player_tag"], troop["troop_name"], troop["village"])
                if key not in troops_keys:
                    troops_keys.add(key)
                    troops_rows.append(troop)
            if len(troops_rows) >= CHUNK_SIZE:
                troops_writer.write_batch(troops_rows)
                troops_rows.clear()

            new_heroes = normalize_heroes(tag, raw.get("heroes"))
            for hero in new_heroes:
                key = (hero["player_tag"], hero["hero_name"], hero["village"])
                if key not in heroes_keys:
                    heroes_keys.add(key)
                    heroes_rows.append(hero)
            if len(heroes_rows) >= CHUNK_SIZE:
                heroes_writer.write_batch(heroes_rows)
                heroes_rows.clear()

            eq_list = raw.get("heroEquipment")
            new_equipment = normalize_hero_equipment(tag, eq_list)
            for eq in new_equipment:
                key = (eq["player_tag"], eq["equipment_name"], eq["village"])
                if key not in equipment_keys:
                    equipment_keys.add(key)
                    equipment_rows.append(eq)
            if len(equipment_rows) >= CHUNK_SIZE:
                equipment_writer.write_batch(equipment_rows)
                equipment_rows.clear()

            new_spells = normalize_spells(tag, raw.get("spells"))
            for spell in new_spells:
                key = (spell["player_tag"], spell["spell_name"], spell["village"])
                if key not in spells_keys:
                    spells_keys.add(key)
                    spells_rows.append(spell)
            if len(spells_rows) >= CHUNK_SIZE:
                spells_writer.write_batch(spells_rows)
                spells_rows.clear()

            new_ach = normalize_achievements(tag, raw.get("achievements"))
            for ach in new_ach:
                key = (ach["player_tag"], ach["achievement_name"], ach["village"])
                if key not in achievements_keys:
                    achievements_keys.add(key)
                    achievements_rows.append(ach)
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
    _log_relationship_coverage(processed_base)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    build_all_tables()

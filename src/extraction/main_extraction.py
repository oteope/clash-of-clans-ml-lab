"""
Main entry point for the Phase 1 Extraction layer – Recursive Clan/Player Crawler.

Uses a diversified search strategy: different filter combinations are applied
to the /clans endpoint so that each run explores new slices of the clan population.
"""
import asyncio
import datetime
import json
import logging
import pathlib

from src.extraction.api_client import CoCClient
from src.extraction.config import (
    verify_ip,
    SEARCHES_PER_REGION_PER_RUN,
)
from src.extraction.search_config import (
    generate_search_configurations,
    load_search_history,
    save_search_history,
    select_unused_configs,
)
from src.extraction.storage import save_raw_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Multi‑location seeding – at least 10 major regions (preserved from original).
# ---------------------------------------------------------------------------
LOCATION_IDS: list[str] = [
    "global",        # Worldwide (will be resolved to None => no location filter)
    "32000006",      # Spain
    "32000107",      # United States
    "32000154",      # Mexico
    "32000038",      # Brazil
    "32000053",      # Argentina
    "32000018",      # France
    "32000049",      # Germany
    "32000066",      # Italy
    "32000179",      # Japan
    "32000225",      # China (if accessible)
    "32000208",      # South Korea
    "32000196",      # Australia
    "32000119",      # Canada
]

# How many clan‑processing coroutines can run in parallel.
# This value should stay well below the client's semaphore to avoid starvation,
# while still providing sufficient concurrency for the (often IO‑bound) work.
BATCH_SIZE = 5


# ---------------------------------------------------------------------------
# Disk‑based de‑duplication helpers
# ---------------------------------------------------------------------------

def _raw_file_exists(category: str, identifier: str) -> bool:
    """Return True if the raw JSON file for *category* / *identifier* already exists."""
    target = pathlib.Path("data") / "raw" / category / f"{identifier}.json"
    return target.exists()


# ---------------------------------------------------------------------------
# Crawler core
# ---------------------------------------------------------------------------

async def process_clans(client: CoCClient,
                        initial_clan_tags: set[str]) -> None:
    """
    Process clans recursively until no new clans are discovered.

    For each unprocessed clan:
      - save its overview (skipped if already on disk)
      - save its members (skipped if already on disk; still loaded for traversal)
      - for each member, save their full player profile (skipped if already on disk)
        and add the member's clan tag to the pending set.
    """
    processed_clans: set[str] = set()
    processed_players: set[str] = set()
    pending_clan_tags: set[str] = set(initial_clan_tags)

    while pending_clan_tags:
        # Grab a batch of tags to process concurrently
        batch: list[str] = []
        for _ in range(BATCH_SIZE):
            if not pending_clan_tags:
                break
            tag = pending_clan_tags.pop()
            batch.append(tag)

        tasks = [
            _process_one_clan(client, tag,
                              processed_clans, processed_players,
                              pending_clan_tags)
            for tag in batch
        ]
        await asyncio.gather(*tasks)

    logger.info(
        "Crawl finished. Clans processed: %d, Players saved: %d",
        len(processed_clans), len(processed_players),
    )


async def _process_one_clan(
    client: CoCClient,
    clan_tag: str,
    processed_clans: set[str],
    processed_players: set[str],
    pending_clan_tags: set[str],
) -> None:
    """Fetch and persist one clan and its members, then extend pending set."""

    # Deduplication via in‑memory set
    if clan_tag in processed_clans:
        return
    processed_clans.add(clan_tag)

    # ------------------------------------------------------------------
    # 1. Clan overview
    # ------------------------------------------------------------------
    if _raw_file_exists("clans", clan_tag):
        logger.info("Clan %s already on disk – skipping API call.", clan_tag)
    else:
        clan = await client.get_clan(clan_tag)
        save_raw_json(clan, "clans", clan_tag)

    # ------------------------------------------------------------------
    # 2. Members (cursor‑paginated).  If already stored, load from disk
    #    to avoid an API call while still obtaining the member list.
    # ------------------------------------------------------------------
    if _raw_file_exists("members", clan_tag):
        logger.info("Members file already on disk for %s – loading from storage.", clan_tag)
        members_path = pathlib.Path("data") / "raw" / "members" / f"{clan_tag}.json"
        with members_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        members: list[dict] = data.get("items", [])
    else:
        members = await client.get_clan_members(clan_tag)
        save_raw_json({"items": members}, "members", clan_tag)

    logger.debug("Clan %s has %d members", clan_tag, len(members))

    # ------------------------------------------------------------------
    # 3. For each member: save full profile and extend clan scope
    # ------------------------------------------------------------------
    for member in members:
        player_tag = member.get("tag")
        if not player_tag:
            continue

        if player_tag in processed_players:
            continue

        # Player profile de‑duplication via disk
        if _raw_file_exists("players", player_tag):
            logger.info("Player file already on disk for %s – skipping API call.", player_tag)
            processed_players.add(player_tag)
        else:
            player = await client.get_player(player_tag)
            save_raw_json(player, "players", player_tag)
            processed_players.add(player_tag)

        # Discover new clan via membership (independent of player fetch)
        member_clan_tag = (member.get("clan") or {}).get("tag")
        if member_clan_tag and member_clan_tag not in processed_clans:
            pending_clan_tags.add(member_clan_tag)


# ---------------------------------------------------------------------------
# Helper for diversified search (stores history only on success)
# ---------------------------------------------------------------------------
async def _search_clans(client: CoCClient,
                        location_id: str | None,
                        cfg: dict) -> list[dict]:
    """Execute a single clan search with the given configuration."""
    return await client.search_clans(location_id=location_id, **cfg)


async def _perform_search(
    client: CoCClient,
    loc_id: str | None,
    cfg: dict,
    search_history: list,
    all_new_seed: set[str],
) -> tuple[set[str], int, int]:
    """
    Execute one diversified search and update *search_history* and *all_new_seed*.

    Returns (all_tags, result_count, genuinely_new_count).
    On error the history is NOT modified and an empty tuple is returned.
    """
    # Build a logging label for this location
    loc_label = loc_id if loc_id is not None else "global"

    try:
        clans = await _search_clans(client, loc_id, cfg)
    except Exception as exc:
        logger.error(
            "Search failed for location %s with filters %s: %s",
            loc_label,
            cfg,
            exc,
        )
        return set(), 0, 0

    tags = {entry.get("tag") for entry in clans if entry.get("tag")}
    results = len(clans)

    genuinely_new = {
        tag for tag in tags
        if not _raw_file_exists("clans", tag) and tag not in all_new_seed
    }
    all_new_seed.update(genuinely_new)

    entry = {
        "location_id": loc_label,
        "filters": cfg,
        "used_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "results": results,
        "new_clans": len(genuinely_new),
    }
    search_history.append(entry)

    logger.info(
        "Search completed: %d results, %d new clans.",
        results,
        len(genuinely_new),
    )

    return tags, results, len(genuinely_new)


# ---------------------------------------------------------------------------
# Location ID resolution
# ---------------------------------------------------------------------------
async def _resolve_location_ids(client: CoCClient, raw_ids: list[str]) -> list[str | None]:
    """
    Convert symbolic location names (e.g. 'global') into numeric location IDs
    or None when the location represents a worldwide search.

    Fetches the real location list from the API; if that fails, falls back to
    a predefined mapping for known symbolic names.
    """
    # 1. Fetch location list from the API
    try:
        api_locations = await client.get_locations()
    except Exception:
        logger.warning("Failed to fetch location list from API. Will rely on fallback mapping.")
        api_locations = []

    # 2. Build a name -> id lookup (case‑insensitive)
    name_to_id: dict[str, str] = {}
    for loc in api_locations:
        name = (loc.get("name") or "").strip().lower()
        loc_id = loc.get("id")
        if name and loc_id is not None:
            name_to_id[name] = str(loc_id)

    # 3. Prepare the result
    resolved: list[str | None] = []
    for raw in raw_ids:
        # If the entry already looks like a numeric ID, keep it as‑is
        if raw.isdigit():
            resolved.append(raw)
            continue

        # Otherwise treat it as a symbolic name
        normalized = raw.strip().lower()
        if normalized == "global":
            # "global" means "do not filter by location"
            resolved.append(None)
            continue

        numeric_id = name_to_id.get(normalized)
        if numeric_id:
            resolved.append(numeric_id)
        else:
            logger.warning("Unknown location name '%s' – skipping.", raw)

    return resolved


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    """Verify IP, seed the crawler with diversified searches, and run the main loop."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger.info("Verifying public IP whitelisting …")
    await verify_ip()
    logger.info(
        "IP verified. Starting diversified discovery across %d locations.",
        len(LOCATION_IDS),
    )

    client = CoCClient()
    try:
        # 1. Resolve location IDs (replace symbolic names like "global")
        location_ids = await _resolve_location_ids(client, LOCATION_IDS)
        logger.info("Resolved location IDs: %s", location_ids)

        # 2. Load or generate the search configuration pool and history
        search_pool = generate_search_configurations()
        search_history = load_search_history()
        all_new_seed: set[str] = set()

        # 3. For each resolved location entry, select unused configurations and
        #    execute searches.  When entry is None (worldwide search) we pass
        #    location_id=None to the API and use "global" as the history key.
        for loc_id in location_ids:
            # Determine the record key for search_history and the label for logs
            record_loc = loc_id if loc_id is not None else "global"
            logger.info("Selecting unused configurations for location %s ...", record_loc)
            selected = select_unused_configs(
                record_loc,
                search_history,
                search_pool,
                max_searches=SEARCHES_PER_REGION_PER_RUN,
            )
            logger.info(
                "Selected %d configuration(s) for location %s",
                len(selected),
                record_loc,
            )

            for cfg in selected:
                logger.info("  - filters: %s", cfg)
                # The helper handles all logic (API call, dedup, history).
                await _perform_search(client, loc_id, cfg, search_history, all_new_seed)

        # 4. Persist the updated history
        save_search_history(search_history)
        logger.info(
            "Initial seed from search configurations contains %d unique clan tags.",
            len(all_new_seed),
        )

        # 5. Run the recursive crawl using only the genuinely new clans
        await process_clans(client, all_new_seed)
    finally:
        await client.close()
        logger.info("Client closed. Extraction finished.")


if __name__ == "__main__":
    asyncio.run(main())

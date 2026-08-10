"""
Main entry point for the Phase 1 Extraction layer – Recursive Clan/Player Crawler.

Seeds clan discovery from location rankings, then recursively follows clan
memberships.  Disk‑based de‑duplication avoids re‑fetching resources that are
already stored under data/raw/.
"""
import asyncio
import json
import logging
import pathlib

from src.extraction.api_client import CoCClient
from src.extraction.config import verify_ip
from src.extraction.storage import save_raw_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Multi‑location seeding – at least 10 major regions.
# ---------------------------------------------------------------------------
LOCATION_IDS: list[str] = [
    "global",        # Worldwide
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
# Discovery helpers
# ---------------------------------------------------------------------------

async def discover_clans_from_locations(client: CoCClient,
                                        location_ids: list[str]) -> set[str]:
    """Collect distinct clan tags from the top‑200 rankings of each location."""
    clan_tags: set[str] = set()

    for loc_id in location_ids:
        logger.info("Fetching clan rankings for location %s …", loc_id)
        ranking = await client.get_location_clans(location_id=loc_id, limit=200)
        location_tags = set()
        for entry in ranking:
            tag = entry.get("tag")
            if tag:
                location_tags.add(tag)
        clan_tags.update(location_tags)
        logger.info("Location %s contributed %d unique clan tags", loc_id, len(location_tags))

    return clan_tags


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
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    """Verify IP, seed the crawler from location rankings, and run the main loop."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger.info("Verifying public IP whitelisting …")
    await verify_ip()
    logger.info("IP verified. Starting discovery across %d locations.", len(LOCATION_IDS))

    client = CoCClient()
    try:
        # 1. Seed discovery from multi‑location rankings (top 200 each)
        initial_clans = await discover_clans_from_locations(client, LOCATION_IDS)
        logger.info("Initial seed contains %d unique clan tags", len(initial_clans))

        # 2. Recursive crawl
        await process_clans(client, initial_clans)
    finally:
        await client.close()
        logger.info("Client closed. Extraction finished.")


if __name__ == "__main__":
    asyncio.run(main())

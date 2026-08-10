"""
Main entry point for the Phase 1 Extraction layer.

Orchestrates async fetching of player, clan, clan members (with pagination)
and war telemetry.  All code, logs, and docstrings are in English.
"""
import asyncio
import logging

from src.extraction.api_client import CoCClient
from src.extraction.config import verify_ip
from src.extraction.storage import save_raw_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration – replace with your actual tags
# ---------------------------------------------------------------------------
PLAYER_TAGS = [
    "#P98VLLL",
    "#YOUR_PLAYER_TAG",
]

CLAN_TAGS = [
    "#220P9CR",
    "#YOUR_CLAN_TAG",
]

WAR_CLAN_TAGS = CLAN_TAGS  # Poll wars for the same clans (can be separate list)


async def fetch_and_save_players(client: CoCClient) -> None:
    """Fetch each player and persist raw JSON."""
    tasks = []
    for tag in PLAYER_TAGS:
        tasks.append(_fetch_and_save_player(client, tag))
    await asyncio.gather(*tasks)


async def _fetch_and_save_player(client: CoCClient, tag: str) -> None:
    raw = await client.get_player(tag)
    save_raw_json(raw, "players", tag)


async def fetch_and_save_clans(client: CoCClient) -> None:
    """Fetch clan info plus all members (paginated) and save both."""
    tasks = []
    for tag in CLAN_TAGS:
        tasks.append(_fetch_and_save_clan(client, tag))
    await asyncio.gather(*tasks)


async def _fetch_and_save_clan(client: CoCClient, tag: str) -> None:
    # Clan overview
    clan_data = await client.get_clan(tag)
    save_raw_json(clan_data, "clans", tag)

    # Paginated members
    members = await client.get_clan_members(tag)
    if members:
        # Save as a list (still raw JSON)
        save_raw_json({"items": members}, "members", tag)


async def poll_wars(client: CoCClient, clan_tags: list[str]) -> None:
    """
    Check current war for every clan and save a full snapshot when the war
    is in 'warEnded' state.  This data is ephemeral and must be captured
    before the next war begins.
    """
    tasks = []
    for tag in clan_tags:
        tasks.append(_poll_war(client, tag))
    await asyncio.gather(*tasks)


async def _poll_war(client: CoCClient, tag: str) -> None:
    war = await client.get_current_war(tag)
    if war is None:
        logger.info("No active war for clan %s (404)", tag)
        return

    state = war.get("state", "")
    logger.info("Clan %s war state: %s", tag, state)

    if state == "warEnded":
        save_raw_json(war, "wars", tag)
        logger.info("Captured ended war for clan %s", tag)


async def main() -> None:
    """Entry point: verify IP, run all fetchers, close client."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    logger.info("Verifying public IP whitelisting …")
    await verify_ip()
    logger.info("IP verified.")

    client = CoCClient()
    try:
        await asyncio.gather(
            fetch_and_save_players(client),
            fetch_and_save_clans(client),
            poll_wars(client, WAR_CLAN_TAGS),
        )
    finally:
        await client.close()
        logger.info("Client closed. Extraction finished.")


if __name__ == "__main__":
    asyncio.run(main())

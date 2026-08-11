"""
Configuration module for the Clash of Clans ML Lab project.

Loads environment variables from .env, checks the current public IP,
and compares it against an optional expected IP for blocking warnings.
"""

import os
import logging
import pathlib
from typing import Optional

from dotenv import load_dotenv
import aiohttp

# Logging configuration
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Load environment variables from .env
load_dotenv()

# Supercell token
COC_API_TOKEN: Optional[str] = os.getenv("COC_API_TOKEN")

# Expected IP (optional)
EXPECTED_IP: Optional[str] = os.getenv("EXPECTED_IP")

# ---------------------------------------------------------------------------
# Diversified search configuration settings
# ---------------------------------------------------------------------------
SEARCH_HISTORY_PATH: pathlib.Path = pathlib.Path("data") / "raw" / "search_history.json"
# Maximum results requested per search call (API limit per page is 200)
SEARCH_LIMIT: int = 200
# Number of different search configurations tried per region in a single run
SEARCHES_PER_REGION_PER_RUN: int = 5
# After all configurations for a region have been used, they become eligible
# again after this many minutes (here ~30 days).
SEARCH_COOLDOWN_MINUTES: int = 60 * 24 * 30


async def get_public_ip() -> str:
    """
    Retrieve the current public IP using api.ipify.org.

    Returns:
        str: Public IP in IPv4 format.

    Raises:
        RuntimeError: If the IP cannot be obtained.
    """
    url = "https://api.ipify.org?format=json"
    timeout = aiohttp.ClientTimeout(total=10)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                response.raise_for_status()
                data = await response.json()
                ip = data.get("ip")
                if not ip:
                    raise RuntimeError("Could not obtain IP from the response.")
                logger.info("Public IP obtained: %s", ip)
                return ip
    except aiohttp.ClientError as e:
        logger.error("Error obtaining public IP: %s", e)
        raise RuntimeError(f"Error obtaining public IP: {e}") from e
    except Exception as e:
        logger.error("Unexpected error obtaining public IP: %s", e)
        raise RuntimeError(f"Unexpected error obtaining public IP: {e}") from e


async def verify_ip() -> None:
    """
    Verify the current public IP against EXPECTED_IP, if defined.

    Logs a warning if they do not match.
    """
    if not EXPECTED_IP:
        logger.info("EXPECTED_IP not defined in .env, skipping IP verification.")
        return

    try:
        current_ip = await get_public_ip()
    except RuntimeError as e:
        logger.error("Could not verify IP: %s", e)
        return

    if current_ip != EXPECTED_IP:
        logger.warning(
            "Current public IP (%s) does not match the expected IP (%s). "
            "The Supercell token may be IP-blocked.",
            current_ip,
            EXPECTED_IP,
        )
    else:
        logger.info("Public IP matches the expected one (%s).", current_ip)


async def main() -> None:
    """
    Main function to perform the IP verification.
    """
    if not COC_API_TOKEN:
        logger.error("COC_API_TOKEN is not defined in .env")
        return

    logger.info("COC_API_TOKEN loaded successfully.")
    await verify_ip()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

"""
Clash of Clans API client for Phase 1 Extraction layer.

Pure extraction: returns raw JSON dictionaries as received from the API.
All code, docstrings, and logs are in English.
"""
import asyncio
import logging
import random

import aiohttp

from src.extraction.config import COC_API_TOKEN

logger = logging.getLogger(__name__)


class IPAddressNotWhitelisted(Exception):
    """Raised when the API returns 403 Forbidden, likely due to unwhitelisted IP."""


class CoCClient:
    """
    Asynchronous client for the Clash of Clans API.

    Enforces rate limiting using an asyncio.Semaphore (max 7 concurrent requests)
    to stay safely below 400 requests/minute.
    """

    BASE_URL = "https://api.clashofclans.com/v1"
    MAX_CONCURRENT_REQUESTS = 7  # Keep safely below API rate limit of ~6.7 req/s

    def __init__(self) -> None:
        self._token: str = COC_API_TOKEN
        self._session: aiohttp.ClientSession | None = None
        # Semaphore to limit concurrent requests
        self._semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_REQUESTS)

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    @staticmethod
    def _format_tag(tag: str) -> str:
        """
        Ensure the tag starts with '#' and is URL-encoded.

        The official API expects tags to be URL-encoded with '#' replaced by '%23'.
        """
        if not tag.startswith("#"):
            tag = f"#{tag}"
        # Replace '#' with '%23' for use in URL path
        return tag.replace("#", "%23")

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    async def get_player(self, player_tag: str) -> dict | None:
        """
        Retrieve player information.

        Returns the raw JSON dictionary or None if the player is not found (404).
        Raises IPAddressNotWhitelisted on 403.
        Retries with exponential backoff on 429.
        """
        encoded_tag = self._format_tag(player_tag)
        path = f"/players/{encoded_tag}"
        return await self._request("GET", path)

    async def get_clan(self, clan_tag: str) -> dict | None:
        """
        Retrieve clan information.

        Returns the raw JSON dictionary or None if not found (404).
        Raises IPAddressNotWhitelisted on 403.
        Retries with exponential backoff on 429.
        """
        encoded_tag = self._format_tag(clan_tag)
        path = f"/clans/{encoded_tag}"
        return await self._request("GET", path)

    async def get_clan_members(self, clan_tag: str) -> list[dict]:
        """
        Fetch all clan members using cursor‑based pagination.

        Returns a list of member dictionaries (may be empty if clan not found).
        Raises IPAddressNotWhitelisted on 403.
        """
        encoded_tag = self._format_tag(clan_tag)
        members: list[dict] = []
        limit = 100  # API maximum
        cursor: str | None = None

        while True:
            path = f"/clans/{encoded_tag}/members?limit={limit}"
            if cursor:
                path += f"&after={cursor}"

            data = await self._request("GET", path)
            if data is None:
                # Clan not found (404) → nothing to return
                break

            items = data.get("items", [])
            members.extend(items)

            if len(items) < limit:
                # Last page reached
                break

            # Extract the next cursor from the paging object
            paging = data.get("paging") or {}
            cursors = paging.get("cursors") or {}
            cursor = cursors.get("after")
            if not cursor:
                break

        return members

    async def get_current_war(self, clan_tag: str) -> dict | None:
        """
        Retrieve current war information for the given clan.

        Returns the raw JSON dictionary or None if not found (404).
        Raises IPAddressNotWhitelisted on 403.
        Retries with exponential backoff on 429.
        """
        encoded_tag = self._format_tag(clan_tag)
        path = f"/clans/{encoded_tag}/currentwar"
        return await self._request("GET", path)

    # ------------------------------------------------------------------
    # Core request method with rate limiting, retries and error handling
    # ------------------------------------------------------------------

    async def _request(self, method: str, path: str) -> dict | None:
        url = f"{self.BASE_URL}{path}"

        if not self._session:
            self._session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self._token}"}
            )

        # Retry settings
        max_retries = 5
        base_backoff = 1  # seconds

        for attempt in range(max_retries + 1):
            async with self._semaphore:  # Rate‑limit enforcement
                try:
                    async with self._session.request(method, url) as resp:
                        status = resp.status

                        if status == 200:
                            data = await resp.json()
                            return data  # Return the raw JSON

                        if status == 404:
                            logger.info(
                                "Resource not found for path %s (status 404).",
                                path,
                            )
                            return None

                        if status == 403:
                            # IP not whitelisted – do not retry
                            raise IPAddressNotWhitelisted(
                                "Received 403 Forbidden. Verify that your current IP "
                                "is whitelisted in the Clash of Clans developer portal."
                            )

                        if status == 429:
                            # Rate limit exceeded – retry with exponential backoff + jitter
                            if attempt < max_retries:
                                # Wait according to Retry-After header if present
                                retry_after = resp.headers.get("Retry-After")
                                if retry_after is not None:
                                    wait_time = float(retry_after)
                                else:
                                    # Exponential backoff with jitter
                                    wait_time = base_backoff * (2 ** attempt)
                                    jitter = random.uniform(0, 1)
                                    wait_time += jitter
                                logger.warning(
                                    "Rate limit reached, retrying in %.1f seconds... "
                                    "(attempt %d/%d)",
                                    wait_time,
                                    attempt + 1,
                                    max_retries,
                                )
                                await asyncio.sleep(wait_time)
                                continue
                            else:
                                # Exhausted retries; propagate error
                                resp.raise_for_status()

                        # Other unexpected status codes
                        resp.raise_for_status()

                except IPAddressNotWhitelisted:
                    raise  # propagate without retry
                except aiohttp.ClientError as exc:
                    # Network-level error, e.g. connection reset
                    if attempt < max_retries:
                        wait_time = base_backoff * (2 ** attempt)
                        logger.warning(
                            "Client error encountered: %s. Retrying in %.1f seconds... "
                            "(attempt %d/%d)",
                            exc,
                            wait_time,
                            attempt + 1,
                            max_retries,
                        )
                        await asyncio.sleep(wait_time + random.uniform(0, 1))
                        continue
                    raise

        # Should never reach here
        raise RuntimeError("Unexpected exit from retry loop")

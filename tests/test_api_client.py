import unittest
from unittest.mock import AsyncMock
import asyncio

from src.extraction.api_client import CoCClient


class TestApiClientSearch(unittest.TestCase):
    def test_search_clans_builds_camel_case_params(self):
        """Verify that search_clans translates snake_case arguments to
        the camelCase query parameters expected by the Clash of Clans API."""

        async def run():
            client = CoCClient()
            mocked_request = AsyncMock(return_value={"items": []})
            # bypass the semaphore and session creation
            client._request = mocked_request

            await client.search_clans(
                location_id="loc123",
                min_members=1,
                max_members=10,
                min_clan_points=200,
                min_clan_level=5,
            )

            called_path = mocked_request.call_args[0][1]  # second positional arg
            self.assertIn("minMembers=1", called_path)
            self.assertIn("maxMembers=10", called_path)
            self.assertIn("minClanPoints=200", called_path)
            self.assertIn("minClanLevel=5", called_path)
            self.assertIn("locationId=loc123", called_path)
            self.assertNotIn("min_members", called_path)

        asyncio.run(run())

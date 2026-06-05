"""Tests for SerpAPIClient. HTTP mocked with respx; no real network."""

from __future__ import annotations

import httpx
import pytest
import respx

from sources.serpapi_client import SerpAPIClient
from tests.conftest import count_usage

_URL = "https://serpapi.test/search"

_OK_BODY = {
    "search_metadata": {"total_searches_left": 87},
    "organic_results": [
        {"title": "Edtech A", "link": "https://a.test", "snippet": "seed funding", "position": 1},
        {"title": "Edtech B", "link": "https://b.test", "snippet": "series a", "position": 2},
    ],
}


@pytest.mark.asyncio
@respx.mock
async def test_happy_path(test_settings):
    respx.get(_URL).mock(return_value=httpx.Response(200, json=_OK_BODY))
    client = SerpAPIClient(test_settings)

    results = await client.search("edtech seed funding")

    assert len(results) == 2
    assert results[0].title == "Edtech A"
    assert results[0].position == 1
    assert count_usage(test_settings, "serpapi") == 1


@pytest.mark.asyncio
@respx.mock
async def test_retry_on_500_then_success(test_settings):
    route = respx.get(_URL).mock(
        side_effect=[
            httpx.Response(500),
            httpx.Response(500),
            httpx.Response(200, json=_OK_BODY),
        ]
    )
    client = SerpAPIClient(test_settings)

    results = await client.search("edtech")

    assert route.call_count == 3
    assert len(results) == 2


@pytest.mark.asyncio
@respx.mock
async def test_rate_limit_429_returns_empty(test_settings):
    route = respx.get(_URL).mock(return_value=httpx.Response(429))
    client = SerpAPIClient(test_settings)

    results = await client.search("edtech")

    assert route.call_count == 3
    assert results == []


@pytest.mark.asyncio
@respx.mock
async def test_auth_failure_401_does_not_raise(test_settings):
    respx.get(_URL).mock(return_value=httpx.Response(401, json={"error": "bad key"}))
    client = SerpAPIClient(test_settings)

    results = await client.search("edtech")

    assert results == []


@pytest.mark.asyncio
@respx.mock
async def test_usage_tracking_written(test_settings):
    respx.get(_URL).mock(return_value=httpx.Response(200, json=_OK_BODY))
    client = SerpAPIClient(test_settings)

    await client.search("edtech")

    assert count_usage(test_settings, "serpapi") == 1


@pytest.mark.asyncio
@respx.mock
async def test_monthly_limit_blocks_call(test_settings):
    # Lower the limit to 0 so the guardrail trips immediately without any HTTP.
    test_settings.SERPAPI_MONTHLY_LIMIT = 0
    route = respx.get(_URL).mock(return_value=httpx.Response(200, json=_OK_BODY))
    client = SerpAPIClient(test_settings)

    results = await client.search("edtech")

    assert results == []
    assert route.call_count == 0

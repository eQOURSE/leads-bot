"""Tests for ApifyMultiKeyClient. HTTP mocked with respx; no real network."""

from __future__ import annotations

import httpx
import pytest
import respx

from sources.apify_client import ApifyMultiKeyClient
from tests.conftest import count_usage

_RUN_URL = (
    "https://api.apify.test/v2/acts/apify~google-search-scraper/"
    "run-sync-get-dataset-items"
)

_OK_ITEMS = [
    {
        "organicResults": [
            {"title": "Result 1", "url": "https://r1.test", "description": "d1", "position": 1},
            {"title": "Result 2", "url": "https://r2.test", "description": "d2", "position": 2},
        ]
    }
]


@pytest.mark.asyncio
@respx.mock
async def test_happy_path_google_search(test_settings):
    respx.post(_RUN_URL).mock(return_value=httpx.Response(200, json=_OK_ITEMS))
    client = ApifyMultiKeyClient(test_settings)

    results = await client.google_search("edtech seed funding", num_results=10)

    assert len(results) == 2
    assert results[0].title == "Result 1"
    assert count_usage(test_settings, "apify") == 1


@pytest.mark.asyncio
@respx.mock
async def test_retry_on_500_then_success(test_settings):
    route = respx.post(_RUN_URL).mock(
        side_effect=[
            httpx.Response(500),
            httpx.Response(500),
            httpx.Response(200, json=_OK_ITEMS),
        ]
    )
    client = ApifyMultiKeyClient(test_settings)

    results = await client.google_search("edtech")

    assert route.call_count == 3
    assert len(results) == 2


@pytest.mark.asyncio
@respx.mock
async def test_rate_limit_429_returns_empty(test_settings):
    route = respx.post(_RUN_URL).mock(return_value=httpx.Response(429))
    client = ApifyMultiKeyClient(test_settings)

    results = await client.google_search("edtech")

    assert route.call_count == 3
    assert results == []


@pytest.mark.asyncio
@respx.mock
async def test_auth_failure_401_does_not_raise(test_settings):
    respx.post(_RUN_URL).mock(return_value=httpx.Response(401, json={}))
    client = ApifyMultiKeyClient(test_settings)

    results = await client.google_search("edtech")

    assert results == []


@pytest.mark.asyncio
@respx.mock
async def test_usage_tracking_written(test_settings):
    respx.post(_RUN_URL).mock(return_value=httpx.Response(200, json=_OK_ITEMS))
    client = ApifyMultiKeyClient(test_settings)

    await client.google_search("edtech")

    assert count_usage(test_settings, "apify") == 1


@pytest.mark.asyncio
async def test_all_tokens_exhausted_returns_empty(test_settings):
    client = ApifyMultiKeyClient(test_settings)
    # Drain every token's estimated credit.
    for token in client.credits:
        client.credits[token] = 0.0

    results = await client.google_search("edtech")

    assert results == []


@pytest.mark.asyncio
@respx.mock
async def test_token_charged_after_call(test_settings):
    respx.post(_RUN_URL).mock(return_value=httpx.Response(200, json=_OK_ITEMS))
    client = ApifyMultiKeyClient(test_settings)
    before = sum(client.credits.values())

    await client.google_search("edtech")

    after = sum(client.credits.values())
    assert after < before

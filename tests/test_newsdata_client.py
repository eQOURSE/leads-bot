"""Tests for NewsDataClient. All HTTP is mocked with respx; no real network."""

from __future__ import annotations

import httpx
import pytest
import respx

from sources.newsdata_client import NewsDataClient
from tests.conftest import count_usage

_URL = "https://newsdata.test/api/1/latest"

_OK_BODY = {
    "status": "success",
    "totalResults": 2,
    "results": [
        {
            "title": "Acme raises $10M Series A",
            "link": "https://news.test/acme",
            "pubDate": "2026-05-01 12:00:00",
            "source_id": "techcrunch",
            "description": "Acme, an edtech startup, raised a Series A.",
        },
        {
            "title": "Beta seed round",
            "link": "https://news.test/beta",
            "pubDate": "2026-05-02 09:30:00",
            "source_id": "tech_eu",
            "description": "Beta closed a seed round.",
        },
    ],
}


@pytest.mark.asyncio
@respx.mock
async def test_happy_path(test_settings):
    respx.get(_URL).mock(return_value=httpx.Response(200, json=_OK_BODY))
    client = NewsDataClient(test_settings)

    items = await client.search_funding_news(keywords=["edtech"])

    assert len(items) == 2
    assert items[0].title == "Acme raises $10M Series A"
    assert items[0].source_name == "techcrunch"
    assert count_usage(test_settings, "newsdata") == 1


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
    client = NewsDataClient(test_settings)

    items = await client.search_funding_news(keywords=["edtech"])

    assert route.call_count == 3
    assert len(items) == 2


@pytest.mark.asyncio
@respx.mock
async def test_rate_limit_429_returns_empty(test_settings):
    # 429 is retryable; after exhausting attempts the client returns [].
    route = respx.get(_URL).mock(return_value=httpx.Response(429))
    client = NewsDataClient(test_settings)

    items = await client.search_funding_news(keywords=["edtech"])

    assert route.call_count == 3
    assert items == []


@pytest.mark.asyncio
@respx.mock
async def test_auth_failure_401_does_not_raise(test_settings):
    respx.get(_URL).mock(return_value=httpx.Response(401, json={"status": "error"}))
    client = NewsDataClient(test_settings)

    items = await client.search_funding_news(keywords=["edtech"])

    assert items == []


@pytest.mark.asyncio
@respx.mock
async def test_usage_tracking_written(test_settings):
    respx.get(_URL).mock(return_value=httpx.Response(200, json=_OK_BODY))
    client = NewsDataClient(test_settings)

    await client.search_company_news("Acme")

    assert count_usage(test_settings, "newsdata") == 1

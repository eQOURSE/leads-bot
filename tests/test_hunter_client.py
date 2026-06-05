"""Tests for HunterClient. HTTP mocked with respx; no real network."""

from __future__ import annotations

import httpx
import pytest
import respx

from sources.hunter_client import HunterClient
from tests.conftest import count_usage

_DOMAIN_URL = "https://api.hunter.test/v2/domain-search"
_FINDER_URL = "https://api.hunter.test/v2/email-finder"

_DOMAIN_BODY = {
    "data": {
        "domain": "acme.com",
        "pattern": "{first}",
        "emails": [{"value": "jane@acme.com", "first_name": "Jane"}],
    },
    "meta": {"limit": 25, "used": 3},
}

_FINDER_BODY = {
    "data": {
        "email": "jane.doe@acme.com",
        "score": 95,
        "position": "CTO",
        "linkedin_url": "https://linkedin.com/in/janedoe",
    },
    "meta": {"limit": 25, "used": 4},
}


@pytest.mark.asyncio
@respx.mock
async def test_happy_path_domain_search(test_settings):
    respx.get(_DOMAIN_URL).mock(return_value=httpx.Response(200, json=_DOMAIN_BODY))
    client = HunterClient(test_settings)

    data = await client.domain_search("acme.com")

    assert data["pattern"] == "{first}"
    assert count_usage(test_settings, "hunter") == 1


@pytest.mark.asyncio
@respx.mock
async def test_email_finder_returns_prospect(test_settings):
    respx.get(_FINDER_URL).mock(return_value=httpx.Response(200, json=_FINDER_BODY))
    client = HunterClient(test_settings)

    prospect = await client.email_finder("acme.com", "Jane", "Doe")

    assert prospect is not None
    assert prospect.email == "jane.doe@acme.com"
    # score 95 normalized onto 0-1 scale
    assert prospect.email_confidence == pytest.approx(0.95)


@pytest.mark.asyncio
@respx.mock
async def test_retry_on_500_then_success(test_settings):
    route = respx.get(_DOMAIN_URL).mock(
        side_effect=[
            httpx.Response(500),
            httpx.Response(500),
            httpx.Response(200, json=_DOMAIN_BODY),
        ]
    )
    client = HunterClient(test_settings)

    data = await client.domain_search("acme.com")

    assert route.call_count == 3
    assert data["domain"] == "acme.com"


@pytest.mark.asyncio
@respx.mock
async def test_rate_limit_429_returns_empty(test_settings):
    route = respx.get(_DOMAIN_URL).mock(return_value=httpx.Response(429))
    client = HunterClient(test_settings)

    data = await client.domain_search("acme.com")

    assert route.call_count == 3
    assert data == {}


@pytest.mark.asyncio
@respx.mock
async def test_auth_failure_401_does_not_raise(test_settings):
    respx.get(_DOMAIN_URL).mock(return_value=httpx.Response(401, json={"errors": []}))
    client = HunterClient(test_settings)

    data = await client.domain_search("acme.com")

    assert data == {}


@pytest.mark.asyncio
@respx.mock
async def test_usage_tracking_written(test_settings):
    respx.get(_DOMAIN_URL).mock(return_value=httpx.Response(200, json=_DOMAIN_BODY))
    client = HunterClient(test_settings)

    await client.domain_search("acme.com")

    assert count_usage(test_settings, "hunter") == 1


@pytest.mark.asyncio
@respx.mock
async def test_monthly_limit_blocks_without_force(test_settings):
    test_settings.HUNTER_MONTHLY_CALL_LIMIT = 0
    route = respx.get(_DOMAIN_URL).mock(return_value=httpx.Response(200, json=_DOMAIN_BODY))
    client = HunterClient(test_settings)

    blocked = await client.domain_search("acme.com")
    assert blocked == {}
    assert route.call_count == 0

    # force=True overrides the guardrail
    forced = await client.domain_search("acme.com", force=True)
    assert forced["domain"] == "acme.com"
    assert route.call_count == 1

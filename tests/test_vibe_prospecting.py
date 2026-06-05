"""Tests for VibeProspectingClient. HTTP mocked with respx; no real network."""

from __future__ import annotations

import httpx
import pytest
import respx

from sources.vibe_prospecting import VibeProspectingClient
from tests.conftest import count_usage

_SEARCH_URL = "https://api.vibeprospecting.test/v1/businesses/search"
_PROSPECTS_URL = "https://api.vibeprospecting.test/v1/prospects/search"

_COMPANIES_BODY = {
    "credits_remaining": 95,
    "data": [
        {
            "domain": "acme.com",
            "name": "Acme Inc",
            "industry": "edtech",
            "naics_codes": ["611420"],
            "size_range": "51-200",
            "country": "US",
            "funding_amount_usd": 10000000,
            "funding_stage": "series-a",
            "confidence": 0.9,
        }
    ],
}

_PROSPECTS_BODY = {
    "credits_remaining": 90,
    "data": [
        {
            "full_name": "Jane Doe",
            "title": "CTO",
            "company_domain": "acme.com",
            "email": "jane@acme.com",
            "email_confidence": 0.8,
        }
    ],
}


@pytest.mark.asyncio
@respx.mock
async def test_happy_path_search_companies(test_settings):
    respx.post(_SEARCH_URL).mock(return_value=httpx.Response(200, json=_COMPANIES_BODY))
    client = VibeProspectingClient(test_settings)

    companies = await client.search_funded_companies(naics_codes=["611420"])

    assert len(companies) == 1
    assert companies[0].domain == "acme.com"
    assert companies[0].funding_stage == "series-a"
    assert count_usage(test_settings, "vibe_prospecting") == 1


@pytest.mark.asyncio
@respx.mock
async def test_find_prospects_happy_path(test_settings):
    respx.post(_PROSPECTS_URL).mock(return_value=httpx.Response(200, json=_PROSPECTS_BODY))
    client = VibeProspectingClient(test_settings)

    prospects = await client.find_prospects(business_domains=["acme.com"])

    assert len(prospects) == 1
    assert prospects[0].full_name == "Jane Doe"


@pytest.mark.asyncio
@respx.mock
async def test_retry_on_500_then_success(test_settings):
    route = respx.post(_SEARCH_URL).mock(
        side_effect=[
            httpx.Response(500),
            httpx.Response(500),
            httpx.Response(200, json=_COMPANIES_BODY),
        ]
    )
    client = VibeProspectingClient(test_settings)

    companies = await client.search_funded_companies()

    assert route.call_count == 3
    assert len(companies) == 1


@pytest.mark.asyncio
@respx.mock
async def test_rate_limit_429_returns_empty(test_settings):
    route = respx.post(_SEARCH_URL).mock(return_value=httpx.Response(429))
    client = VibeProspectingClient(test_settings)

    companies = await client.search_funded_companies()

    assert route.call_count == 3
    assert companies == []


@pytest.mark.asyncio
@respx.mock
async def test_auth_failure_401_does_not_raise(test_settings):
    respx.post(_SEARCH_URL).mock(return_value=httpx.Response(401, json={}))
    client = VibeProspectingClient(test_settings)

    companies = await client.search_funded_companies()

    assert companies == []


@pytest.mark.asyncio
@respx.mock
async def test_usage_tracking_written(test_settings):
    respx.post(_SEARCH_URL).mock(return_value=httpx.Response(200, json=_COMPANIES_BODY))
    client = VibeProspectingClient(test_settings)

    await client.search_funded_companies()

    assert count_usage(test_settings, "vibe_prospecting") == 1

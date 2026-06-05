"""Tests for CompaniesAPIClient. HTTP mocked with respx; no real network.

Mirrors the real thecompaniesapi.com v2 contract: GET requests with a Bearer
token, a JSON-array ``query`` param, a nested ``domain`` object, and credit
info under ``meta``.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from sources.companies_api_client import CompaniesAPIClient
from tests.conftest import count_usage

_SEARCH_URL = "https://api.companies.test/v2/companies"
_DOMAIN_URL = "https://api.companies.test/v2/companies/acme.com"

_SEARCH_BODY = {
    "meta": {"cost": 3, "credits": 480, "total": 1},
    "companies": [
        {
            "domain": {"domain": "acme.com", "tld": "com"},
            "name": "Acme Inc",
            "about": {
                "name": "Acme Inc",
                "industries": ["edtech"],
                "totalEmployees": "51-200",
            },
            "locations": {
                "headquarters": {
                    "country": {"name": "United States", "code": "us"},
                    "region": {"name": "California"},
                }
            },
        }
    ],
}

_DOMAIN_BODY = {
    "meta": {"cost": 1, "credits": 479},
    "domain": {"domain": "acme.com", "tld": "com"},
    "name": "Acme Inc",
    "about": {"name": "Acme Inc", "industries": ["edtech"]},
    "locations": {"headquarters": {"country": {"name": "United States", "code": "us"}}},
}


@pytest.mark.asyncio
@respx.mock
async def test_happy_path_search(test_settings):
    respx.get(_SEARCH_URL).mock(return_value=httpx.Response(200, json=_SEARCH_BODY))
    client = CompaniesAPIClient(test_settings)

    companies = await client.search_by_filters(industries=["edtech"], countries=["us"])

    assert len(companies) == 1
    assert companies[0].domain == "acme.com"
    assert companies[0].name == "Acme Inc"
    assert companies[0].hq_country == "United States"
    assert count_usage(test_settings, "companies_api") == 1


@pytest.mark.asyncio
@respx.mock
async def test_enrich_by_domain(test_settings):
    respx.get(_DOMAIN_URL).mock(return_value=httpx.Response(200, json=_DOMAIN_BODY))
    client = CompaniesAPIClient(test_settings)

    company = await client.enrich_by_domain("acme.com")

    assert company is not None
    assert company.name == "Acme Inc"
    assert company.domain == "acme.com"


@pytest.mark.asyncio
@respx.mock
async def test_retry_on_500_then_success(test_settings):
    route = respx.get(_SEARCH_URL).mock(
        side_effect=[
            httpx.Response(500),
            httpx.Response(500),
            httpx.Response(200, json=_SEARCH_BODY),
        ]
    )
    client = CompaniesAPIClient(test_settings)

    companies = await client.search_by_filters(industries=["edtech"])

    assert route.call_count == 3
    assert len(companies) == 1


@pytest.mark.asyncio
@respx.mock
async def test_rate_limit_429_returns_empty(test_settings):
    route = respx.get(_SEARCH_URL).mock(return_value=httpx.Response(429))
    client = CompaniesAPIClient(test_settings)

    companies = await client.search_by_filters(industries=["edtech"])

    assert route.call_count == 3
    assert companies == []


@pytest.mark.asyncio
@respx.mock
async def test_auth_failure_401_does_not_raise(test_settings):
    respx.get(_SEARCH_URL).mock(return_value=httpx.Response(401, json={}))
    client = CompaniesAPIClient(test_settings)

    companies = await client.search_by_filters(industries=["edtech"])

    assert companies == []


@pytest.mark.asyncio
@respx.mock
async def test_usage_tracking_written(test_settings):
    respx.get(_SEARCH_URL).mock(return_value=httpx.Response(200, json=_SEARCH_BODY))
    client = CompaniesAPIClient(test_settings)

    await client.search_by_filters(industries=["edtech"])

    assert count_usage(test_settings, "companies_api") == 1

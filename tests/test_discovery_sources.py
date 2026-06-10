"""Phase 11 — tests for Crunchbase + Wellfound Apify discovery clients
and the ICP→category mapping helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents._constants import (
    crunchbase_categories_for_icp,
    wellfound_markets_for_icp,
)
from sources.crunchbase_apify import CrunchbaseAPIfyClient
from sources.wellfound_apify import WellfoundAPIfyClient


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------

def test_crunchbase_categories_for_edtech():
    cats = crunchbase_categories_for_icp(["611710"], ["edtech", "k-12"])
    assert "Education" in cats
    assert "EdTech" in cats
    # de-duplicated
    assert len(cats) == len(set(cats))


def test_crunchbase_categories_for_ai():
    cats = crunchbase_categories_for_icp(["541512"], ["ai", "llm", "computer vision"])
    assert "Artificial Intelligence" in cats
    assert "Computer Vision" in cats


def test_wellfound_markets_for_icp():
    markets = wellfound_markets_for_icp(["edtech", "ai"])
    assert "Education" in markets
    assert "Artificial Intelligence" in markets


# ---------------------------------------------------------------------------
# Crunchbase client
# ---------------------------------------------------------------------------

def _cb_items():
    return [
        {
            "name": "Acme AI", "domain": "acme-ai.com",
            "short_description": "AI annotation platform",
            "category": "Artificial Intelligence", "employee_count": 35,
            "last_funding_type": "series_a", "funding_total": 8000000,
            "last_funding_at": "2026-05-01", "country": "United States",
        },
        {
            "name": "NoDomain Co", "employee_count": "11-50",
            "funding_total": "2,000,000",
        },
    ]


@pytest.mark.asyncio
async def test_crunchbase_maps_items_to_candidates(test_settings):
    test_settings.ENABLE_CRUNCHBASE_DISCOVERY = True
    client = CrunchbaseAPIfyClient(test_settings)

    async def fake_request(self, c, method, url, **kwargs):
        class R:
            def json(self_inner):
                return _cb_items()
        return R()

    with patch.object(CrunchbaseAPIfyClient, "_request", new=fake_request), \
         patch.object(CrunchbaseAPIfyClient, "_track", new=AsyncMock()):
        out = await client.search_recent_funding(
            industries=["541512"], keywords=["ai"], limit=50
        )

    assert len(out) == 2
    acme = next(c for c in out if c.name == "Acme AI")
    assert acme.domain == "acme-ai.com"
    assert acme.confidence == 0.85
    assert acme.size_range == "11-50"  # 35 → 11-50 bucket
    assert acme.funding_stage == "series_a"
    # company with no domain gets a .unknown slug
    nod = next(c for c in out if c.name == "NoDomain Co")
    assert nod.domain.endswith(".unknown")


@pytest.mark.asyncio
async def test_crunchbase_returns_empty_on_error(test_settings):
    test_settings.ENABLE_CRUNCHBASE_DISCOVERY = True
    client = CrunchbaseAPIfyClient(test_settings)

    async def boom(self, c, method, url, **kwargs):
        raise RuntimeError("actor 404")

    with patch.object(CrunchbaseAPIfyClient, "_request", new=boom):
        out = await client.search_recent_funding(industries=["541512"], limit=10)
    assert out == []


@pytest.mark.asyncio
async def test_crunchbase_no_tokens_returns_empty(test_settings):
    test_settings.ENABLE_CRUNCHBASE_DISCOVERY = True
    # Strip tokens
    test_settings.APIFY_TOKEN_1 = None
    test_settings.APIFY_TOKEN_2 = None
    test_settings.APIFY_TOKEN_3 = None
    test_settings.APIFY_TOKEN_4 = None
    client = CrunchbaseAPIfyClient(test_settings)
    out = await client.search_recent_funding(industries=["541512"], limit=10)
    assert out == []


# ---------------------------------------------------------------------------
# Wellfound client
# ---------------------------------------------------------------------------

def _wf_items():
    return [
        {
            "name": "Startup X", "website": "startupx.io",
            "description": "K-12 learning", "markets": ["Education"],
            "companySize": 20, "stage": "seed", "country": "United States",
            "totalRaised": 1500000,
        },
    ]


@pytest.mark.asyncio
async def test_wellfound_maps_items(test_settings):
    test_settings.ENABLE_WELLFOUND_DISCOVERY = True
    client = WellfoundAPIfyClient(test_settings)

    async def fake_request(self, c, method, url, **kwargs):
        class R:
            def json(self_inner):
                return {"items": _wf_items()}
        return R()

    with patch.object(WellfoundAPIfyClient, "_request", new=fake_request), \
         patch.object(WellfoundAPIfyClient, "_track", new=AsyncMock()):
        out = await client.search_recent_startups(
            industries=["611710"], keywords=["edtech"], limit=50
        )

    assert len(out) == 1
    s = out[0]
    assert s.domain == "startupx.io"
    assert s.confidence == 0.8
    assert s.size_range == "11-50"
    assert s.industry == "Education"

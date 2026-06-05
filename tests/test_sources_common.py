"""Tests for shared models and utilities (no network)."""

from __future__ import annotations

import pytest

from sources._utils import make_lead_hash, normalize_domain, track_usage
from sources.models import CompanyCandidate, ProspectCandidate, SearchResult
from tests.conftest import count_usage


def test_normalize_domain_variants():
    assert normalize_domain("https://www.Acme.com/path?x=1") == "acme.com"
    assert normalize_domain("blog.acme.co.uk/page") == "acme.co.uk"
    assert normalize_domain("HTTP://Example.COM") == "example.com"
    assert normalize_domain("") == ""


def test_make_lead_hash_is_stable_and_normalized():
    h1 = make_lead_hash("https://www.acme.com", "Jane Doe")
    h2 = make_lead_hash("acme.com", "jane doe")
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex


def test_company_candidate_confidence_bounds():
    with pytest.raises(Exception):
        CompanyCandidate(domain="acme.com", name="Acme", raw_source="t", confidence=1.5)


def test_company_candidate_domain_normalized():
    c = CompanyCandidate(
        domain="https://www.Acme.com/about",
        name="Acme",
        raw_source="t",
        confidence=0.5,
    )
    assert c.domain == "acme.com"


def test_prospect_email_confidence_normalized():
    p = ProspectCandidate(
        full_name="Jane",
        title="CTO",
        company_domain="acme.com",
        email="jane@acme.com",
        email_confidence=95,  # 0-100 scale
        source="t",
    )
    assert p.email_confidence == pytest.approx(0.95)


def test_search_result_defaults():
    r = SearchResult(title="x", url="https://x.test", position=1)
    assert r.snippet == ""


@pytest.mark.asyncio
async def test_track_usage_writes_row(test_settings):
    await track_usage("unit_test_source", 3, 100, test_settings)
    assert count_usage(test_settings, "unit_test_source") == 1

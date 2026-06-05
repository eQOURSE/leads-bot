"""Tests for ScrapeGraphClient — v2 SDK (scrapegraph-py >= 2.1.0).

All tests are offline: AsyncScrapeGraphAI is mocked with AsyncMock so no real
HTTP calls are made.

ApiResult shape: result.status, result.data.json_data, result.error
"""

from __future__ import annotations

from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sources.scrapegraph_client import ScrapeGraphClient
from tests.conftest import count_usage


def _make_success_result(json_data: dict) -> MagicMock:
    """Build a mock ApiResult with status='success'."""
    result = MagicMock()
    result.status = "success"
    result.data = MagicMock()
    result.data.json_data = json_data
    result.error = None
    return result


def _make_error_result(error: str = "api_error") -> MagicMock:
    """Build a mock ApiResult with status='error'."""
    result = MagicMock()
    result.status = "error"
    result.data = None
    result.error = error
    return result


def _make_credits_result(remaining: int = 42) -> MagicMock:
    result = MagicMock()
    result.status = "success"
    result.data = MagicMock()
    result.data.remaining = remaining
    result.error = None
    return result


@pytest.fixture
def mock_sgai_cls():
    """Patch AsyncScrapeGraphAI used inside scrapegraph_client."""
    with patch("sources.scrapegraph_client.AsyncScrapeGraphAI") as cls:  # type: ignore[attr-defined]
        instance = AsyncMock()
        cls.return_value.__aenter__ = AsyncMock(return_value=instance)
        cls.return_value.__aexit__ = AsyncMock(return_value=False)
        yield cls, instance


# ---------------------------------------------------------------------------
# test_extract_team_page_returns_members
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_team_page_returns_members(test_settings, mock_sgai_cls):
    _, sgai_instance = mock_sgai_cls
    sgai_instance.extract.return_value = _make_success_result({
        "members": [
            {"full_name": "Alice Founder", "title": "CEO", "linkedin_url": "https://li.test/alice"},
            {"full_name": "Bob Cto", "title": "CTO", "linkedin_url": None},
            {"full_name": "Carol VP", "title": "VP Engineering", "linkedin_url": None},
        ]
    })

    client = ScrapeGraphClient(test_settings)
    people = await client.extract_team_page("https://acme.com")

    assert len(people) == 3
    assert people[0].full_name == "Alice Founder"
    assert people[0].title == "CEO"
    assert people[0].source == "scrapegraph"
    assert count_usage(test_settings, "scrapegraph") == 1


# ---------------------------------------------------------------------------
# test_extract_falls_back_through_url_variants
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_falls_back_through_url_variants(test_settings, mock_sgai_cls):
    """First variant (/team) returns empty members; /about returns 2 members."""
    _, sgai_instance = mock_sgai_cls

    empty_result = _make_success_result({"members": []})
    about_result = _make_success_result({
        "members": [
            {"full_name": "Dan Dir", "title": "Director of Product", "linkedin_url": None},
            {"full_name": "Eve Head", "title": "Head of Sales", "linkedin_url": "https://li.test/eve"},
        ]
    })

    # extract is called for each URL variant: /team → empty, /about → 2 members
    sgai_instance.extract.side_effect = [empty_result, about_result]

    client = ScrapeGraphClient(test_settings)
    people = await client.extract_team_page("https://startup.io")

    assert len(people) == 2
    assert people[1].full_name == "Eve Head"
    # Two API calls were made (one for /team, one for /about)
    assert sgai_instance.extract.call_count == 2


# ---------------------------------------------------------------------------
# test_extract_handles_api_result_error_status
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_handles_api_result_error_status(test_settings, mock_sgai_cls):
    """API returns status='error' with error='invalid_url' → returns []."""
    _, sgai_instance = mock_sgai_cls
    sgai_instance.extract.return_value = _make_error_result("invalid_url")

    client = ScrapeGraphClient(test_settings)
    people = await client.extract_team_page("https://bad-url.com")

    assert people == []
    # No usage tracked for error result
    assert count_usage(test_settings, "scrapegraph") == 0


# ---------------------------------------------------------------------------
# test_extract_handles_exception
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_handles_exception(test_settings, mock_sgai_cls):
    """Network error (ConnectionError) → returns [], doesn't crash."""
    _, sgai_instance = mock_sgai_cls
    sgai_instance.extract.side_effect = ConnectionError("network down")

    client = ScrapeGraphClient(test_settings)
    people = await client.extract_team_page("https://acme.com")

    assert people == []
    assert count_usage(test_settings, "scrapegraph") == 0


# ---------------------------------------------------------------------------
# test_cache_hit_skips_api_call
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cache_hit_skips_api_call(test_settings, mock_sgai_cls):
    """Second call with the same URL should use the cache and skip AsyncScrapeGraphAI."""
    _, sgai_instance = mock_sgai_cls
    sgai_instance.extract.return_value = _make_success_result({
        "members": [
            {"full_name": "Frank CEO", "title": "CEO", "linkedin_url": None},
        ]
    })

    client = ScrapeGraphClient(test_settings)

    # First call — hits the API
    first = await client.extract_company_summary("https://cached.com")
    # Second call — should be served from cache
    second = await client.extract_company_summary("https://cached.com")

    assert first == second
    # Only one API call was made
    assert sgai_instance.extract.call_count == 1


# ---------------------------------------------------------------------------
# test_credits_remaining_parses_response
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_credits_remaining_parses_response(test_settings, mock_sgai_cls):
    _, sgai_instance = mock_sgai_cls
    sgai_instance.credits.return_value = _make_credits_result(remaining=77)

    client = ScrapeGraphClient(test_settings)
    credits = await client.get_credits_remaining()

    assert credits == 77


# ---------------------------------------------------------------------------
# test_auth_error_disables_scrapegraph_for_run
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_auth_error_disables_scrapegraph_for_run(test_settings, mock_sgai_cls):
    """An auth-related error should set _scrapegraph_available=False."""
    _, sgai_instance = mock_sgai_cls
    sgai_instance.extract.return_value = _make_error_result("auth_failed: invalid api key")

    client = ScrapeGraphClient(test_settings)
    assert client._scrapegraph_available is True

    people = await client.extract_team_page("https://acme.com")

    assert people == []
    assert client._scrapegraph_available is False

    # Subsequent call should be skipped without touching the API
    people2 = await client.extract_team_page("https://another.com")
    assert people2 == []
    # Still only 1 API call was made (the second was skipped)
    assert sgai_instance.extract.call_count == 1

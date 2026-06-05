"""Tests for AbstractAPIClient — Phase 6. All offline."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sources.abstract_api_client import AbstractAPIClient


def _make_response(status: int, body: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = body
    return resp


_VALID_RESPONSE = {
    "email": "test@example.com",
    "deliverability": "DELIVERABLE",
    "quality_score": "0.90",
    "is_valid_format": {"value": True},
    "is_disposable_email": {"value": False},
    "is_smtp_valid": {"value": True},
    "is_role_email": {"value": False},
    "is_catchall_email": {"value": False},
    "is_mx_found": {"value": True},
}


@pytest.fixture
def abstract_settings(test_settings):
    """test_settings with ABSTRACT_EMAIL_API_KEY populated."""
    test_settings.ABSTRACT_EMAIL_API_KEY = "test-abstract-key"
    return test_settings


def _mock_httpx_get(response):
    """Patch httpx.AsyncClient so .get() returns a given response."""
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=response)
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    return mock_ctx, mock_client


# ---------------------------------------------------------------------------
# test_validate_email_parses_response
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_validate_email_parses_response(abstract_settings):
    mock_ctx, mock_client = _mock_httpx_get(_make_response(200, _VALID_RESPONSE))
    with patch("sources.abstract_api_client.httpx.AsyncClient", return_value=mock_ctx):
        client = AbstractAPIClient(abstract_settings)
        result = await client.validate_email("test@example.com")

    assert result["deliverability"] == "DELIVERABLE"
    # Nested {value: bool} should be flattened
    assert result["is_smtp_valid"] is True
    assert result["is_catchall_email"] is False


# ---------------------------------------------------------------------------
# test_validate_email_401_returns_empty
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_validate_email_401_returns_empty(abstract_settings):
    mock_ctx, _ = _mock_httpx_get(_make_response(401, {"error": "invalid_api_key"}))
    with patch("sources.abstract_api_client.httpx.AsyncClient", return_value=mock_ctx):
        client = AbstractAPIClient(abstract_settings)
        result = await client.validate_email("test@example.com")

    assert result == {}


# ---------------------------------------------------------------------------
# test_validate_email_429_marks_exhausted
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_validate_email_429_marks_exhausted(abstract_settings):
    mock_ctx, _ = _mock_httpx_get(_make_response(429, {}))
    with patch("sources.abstract_api_client.httpx.AsyncClient", return_value=mock_ctx):
        client = AbstractAPIClient(abstract_settings)
        assert client._exhausted is False
        result = await client.validate_email("test@example.com")

    assert result == {}
    assert client._exhausted is True

    # Second call should be skipped — no new HTTP request
    mock_ctx2, mock_client2 = _mock_httpx_get(_make_response(200, _VALID_RESPONSE))
    with patch("sources.abstract_api_client.httpx.AsyncClient", return_value=mock_ctx2):
        result2 = await client.validate_email("other@example.com")

    assert result2 == {}
    assert mock_client2.get.call_count == 0


# ---------------------------------------------------------------------------
# test_cache_hit_skips_api_call
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cache_hit_skips_api_call(abstract_settings):
    mock_ctx, mock_client = _mock_httpx_get(_make_response(200, _VALID_RESPONSE))
    with patch("sources.abstract_api_client.httpx.AsyncClient", return_value=mock_ctx):
        client = AbstractAPIClient(abstract_settings)
        first = await client.validate_email("cached@example.com")
        second = await client.validate_email("cached@example.com")

    assert first == second
    # Only one HTTP call - second served from cache
    assert mock_client.get.call_count == 1

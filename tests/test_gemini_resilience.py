"""Phase 11 — tests for Gemini retry + model fallback resilience."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents import _gemini_wrapper
from agents._gemini_wrapper import (
    GeminiAgent,
    _is_rate_limit_error,
    _is_non_retryable_error,
)


class _Err(Exception):
    """Custom exception whose class name we control for detection tests."""


def _make_agent(test_settings):
    test_settings.GEMINI_MODEL_FALLBACK = "gemini-2.5-flash"
    return GeminiAgent("gemini-2.5-flash-lite", test_settings)


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

def test_rate_limit_detection_by_message():
    assert _is_rate_limit_error(Exception("503 UNAVAILABLE")) is True
    assert _is_rate_limit_error(Exception("429 RESOURCE_EXHAUSTED")) is True
    assert _is_rate_limit_error(Exception("model is overloaded")) is True
    assert _is_rate_limit_error(Exception("rate limit exceeded")) is True
    assert _is_rate_limit_error(Exception("bad json")) is False


def test_non_retryable_detection():
    assert _is_non_retryable_error(Exception("400 invalid argument")) is True
    assert _is_non_retryable_error(Exception("403 permission denied")) is True
    assert _is_non_retryable_error(Exception("503 unavailable")) is False


# ---------------------------------------------------------------------------
# Retry + fallback behavior
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retries_then_succeeds_on_primary(test_settings):
    agent = _make_agent(test_settings)
    calls = {"n": 0}

    def fake_sync(prompt, temp, mx, as_json, schema, model_name=None):
        calls["n"] += 1
        if calls["n"] < 2:
            raise Exception("503 UNAVAILABLE")
        return ("ok-text", 10)

    with patch.object(GeminiAgent, "_generate_sync", side_effect=fake_sync), \
         patch("asyncio.sleep", return_value=None), \
         patch.object(_gemini_wrapper, "track_usage", new=_async_noop):
        result = await agent._call("p", temperature=0.1)

    assert result == "ok-text"
    assert agent.retry_count == 1
    assert agent.fallback_count == 0


@pytest.mark.asyncio
async def test_falls_back_to_fallback_model(test_settings):
    agent = _make_agent(test_settings)
    seen_models = []

    def fake_sync(prompt, temp, mx, as_json, schema, model_name=None):
        seen_models.append(model_name)
        # Primary model always 503s; fallback succeeds.
        if model_name == "gemini-2.5-flash-lite":
            raise Exception("503 UNAVAILABLE")
        return ("fallback-text", 5)

    with patch.object(GeminiAgent, "_generate_sync", side_effect=fake_sync), \
         patch("asyncio.sleep", return_value=None), \
         patch.object(_gemini_wrapper, "track_usage", new=_async_noop):
        result = await agent._call("p", temperature=0.1)

    assert result == "fallback-text"
    assert agent.fallback_count == 1
    assert "gemini-2.5-flash" in seen_models


@pytest.mark.asyncio
async def test_non_retryable_returns_none_immediately(test_settings):
    agent = _make_agent(test_settings)
    calls = {"n": 0}

    def fake_sync(prompt, temp, mx, as_json, schema, model_name=None):
        calls["n"] += 1
        raise Exception("400 invalid argument")

    with patch.object(GeminiAgent, "_generate_sync", side_effect=fake_sync), \
         patch("asyncio.sleep", return_value=None), \
         patch.object(_gemini_wrapper, "track_usage", new=_async_noop):
        result = await agent._call("p", temperature=0.1)

    assert result is None
    assert calls["n"] == 1  # no retries on non-retryable error


@pytest.mark.asyncio
async def test_all_attempts_fail_returns_none(test_settings):
    agent = _make_agent(test_settings)

    def fake_sync(prompt, temp, mx, as_json, schema, model_name=None):
        raise Exception("503 UNAVAILABLE")

    with patch.object(GeminiAgent, "_generate_sync", side_effect=fake_sync), \
         patch("asyncio.sleep", return_value=None), \
         patch.object(_gemini_wrapper, "track_usage", new=_async_noop):
        result = await agent._call("p", temperature=0.1)

    assert result is None
    # 3 primary + 2 fallback = 5 attempts; retries counted on all but the last.
    assert agent.retry_count >= 3
    assert agent.fallback_count == 0  # fallback attempts also failed


async def _async_noop(*args, **kwargs):
    return None

"""Tests for the agent-level GeminiAgent wrapper.

HTTP is mocked with respx against the AI Studio generateContent endpoint. The
google-genai SDK uses a synchronous httpx client under the hood (the wrapper
runs it via asyncio.to_thread), which respx can intercept. No real network.
"""

from __future__ import annotations

import re

import httpx
import pytest
import respx
from pydantic import BaseModel

from agents._gemini_wrapper import GeminiAgent
from tests.conftest import count_usage

# Matches AI Studio: https://generativelanguage.googleapis.com/.../models/<m>:generateContent
_GEN_URL = re.compile(
    r"https://generativelanguage\.googleapis\.com/.*:generateContent"
)


class _Demo(BaseModel):
    name: str
    score: int


def _genai_response(text: str, total_tokens: int = 42) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "candidates": [{"content": {"parts": [{"text": text}]}}],
            "usageMetadata": {"totalTokenCount": total_tokens},
        },
    )


@pytest.mark.asyncio
@respx.mock
async def test_generate_json_happy_path(test_settings):
    respx.post(_GEN_URL).mock(
        return_value=_genai_response('{"name": "Acme", "score": 7}')
    )
    agent = GeminiAgent("gemini-2.5-flash-lite", test_settings)

    result = await agent.generate_json("extract", _Demo)

    assert result is not None
    assert result.name == "Acme"
    assert result.score == 7
    assert count_usage(test_settings, "gemini_gemini-2.5-flash-lite") == 1


@pytest.mark.asyncio
@respx.mock
async def test_generate_json_retries_on_bad_json_then_succeeds(test_settings):
    route = respx.post(_GEN_URL).mock(
        side_effect=[
            _genai_response("not json at all"),
            _genai_response('{"name": "Beta", "score": 3}'),
        ]
    )
    agent = GeminiAgent("gemini-2.5-flash-lite", test_settings)

    result = await agent.generate_json("extract", _Demo)

    assert route.call_count == 2
    assert result is not None
    assert result.name == "Beta"


@pytest.mark.asyncio
@respx.mock
async def test_generate_json_returns_none_after_exhausting_retries(test_settings):
    # Always return invalid JSON: 1 initial + 2 retries = 3 calls, then None.
    route = respx.post(_GEN_URL).mock(return_value=_genai_response("still not json"))
    agent = GeminiAgent("gemini-2.5-flash-lite", test_settings)

    result = await agent.generate_json("extract", _Demo)

    assert result is None
    assert route.call_count == 3


@pytest.mark.asyncio
@respx.mock
async def test_generate_text_happy_path(test_settings):
    respx.post(_GEN_URL).mock(return_value=_genai_response("Hello from Gemini."))
    agent = GeminiAgent("gemini-2.5-flash", test_settings)

    text = await agent.generate_text("write a greeting")

    assert text == "Hello from Gemini."
    assert count_usage(test_settings, "gemini_gemini-2.5-flash") == 1


@pytest.mark.asyncio
@respx.mock
async def test_generate_text_returns_empty_on_error(test_settings):
    respx.post(_GEN_URL).mock(return_value=httpx.Response(401, json={"error": {}}))
    agent = GeminiAgent("gemini-2.5-flash", test_settings)

    text = await agent.generate_text("write a greeting")

    assert text == ""


@pytest.mark.asyncio
@respx.mock
async def test_batch_generate_json(test_settings):
    respx.post(_GEN_URL).mock(
        return_value=_genai_response('{"name": "X", "score": 1}')
    )
    agent = GeminiAgent("gemini-2.5-flash-lite", test_settings)

    results = await agent.batch_generate_json(["a", "b", "c"], _Demo)

    assert len(results) == 3
    assert all(r is not None and r.name == "X" for r in results)


def test_auth_mode_ai_studio_builds_client(test_settings):
    # test_settings has GEMINI_API_KEY set, no GCP project -> ai_studio.
    assert test_settings.gemini_auth_mode == "ai_studio"
    agent = GeminiAgent("gemini-2.5-flash-lite", test_settings)
    client = agent._build_client()
    assert client is not None


def test_auth_mode_vertex_selected(test_settings, tmp_path, monkeypatch):
    # Simulate Vertex: no AI Studio key, GCP project set, creds file exists.
    creds = tmp_path / "sa.json"
    creds.write_text("{}", encoding="utf-8")
    vertex_settings = test_settings.model_copy(
        update={
            "GEMINI_API_KEY": None,
            "GCP_PROJECT_ID": "demo-project",
            "GOOGLE_APPLICATION_CREDENTIALS": str(creds),
        }
    )
    assert vertex_settings.gemini_auth_mode == "vertex"

    # Confirm the wrapper routes to the vertex Client construction path.
    captured = {}

    class _FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import google.genai as genai

    monkeypatch.setattr(genai, "Client", _FakeClient)
    agent = GeminiAgent("gemini-2.5-flash-lite", vertex_settings)
    agent._build_client()

    assert captured.get("vertexai") is True
    assert captured.get("project") == "demo-project"

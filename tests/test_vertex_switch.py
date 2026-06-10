"""Phase 12 — tests for Vertex AI client selection + Apify disable guards."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents._gemini_wrapper import GeminiAgent


# ---------------------------------------------------------------------------
# GeminiAgent client construction
# ---------------------------------------------------------------------------

def _vertex_settings(test_settings, tmp_path):
    """A settings object configured to select Vertex."""
    creds = tmp_path / "sa.json"
    creds.write_text("{}")
    test_settings.USE_VERTEX_AI = True
    test_settings.GCP_PROJECT_ID = "tutrain-automation"
    test_settings.GCP_REGION = "us-central1"
    test_settings.GOOGLE_APPLICATION_CREDENTIALS = str(creds)
    return test_settings


def test_gemini_agent_uses_vertex_when_configured(test_settings, tmp_path):
    settings = _vertex_settings(test_settings, tmp_path)
    assert settings.gemini_auth_mode == "vertex"

    fake_genai = MagicMock()
    with patch.dict("sys.modules", {"google.genai": fake_genai, "google": MagicMock(genai=fake_genai)}):
        # Patch the lazy `from google import genai` inside _build_client.
        with patch("agents._gemini_wrapper.GeminiAgent._build_client") as mock_build:
            agent = GeminiAgent("gemini-2.5-flash-lite", settings)
            # force client build
            agent._get_client()
            mock_build.assert_called_once()

    # Now verify the real _build_client passes vertexai=True.
    agent2 = GeminiAgent("gemini-2.5-flash-lite", settings)
    captured = {}

    class FakeGenai:
        @staticmethod
        def Client(**kwargs):
            captured.update(kwargs)
            return MagicMock()

    import types as _t
    google_mod = _t.ModuleType("google")
    genai_mod = _t.ModuleType("google.genai")
    genai_mod.Client = FakeGenai.Client
    google_mod.genai = genai_mod
    with patch.dict("sys.modules", {"google": google_mod, "google.genai": genai_mod}):
        agent2._build_client()
    assert captured.get("vertexai") is True
    assert captured.get("project") == "tutrain-automation"
    assert captured.get("location") == "us-central1"


def test_gemini_agent_falls_back_to_ai_studio_when_use_vertex_false(test_settings, tmp_path):
    settings = _vertex_settings(test_settings, tmp_path)
    settings.USE_VERTEX_AI = False
    settings.GEMINI_API_KEY = "ai-studio-key"
    assert settings.gemini_auth_mode == "ai_studio"

    agent = GeminiAgent("gemini-2.5-flash-lite", settings)
    captured = {}

    import types as _t
    google_mod = _t.ModuleType("google")
    genai_mod = _t.ModuleType("google.genai")
    genai_mod.Client = lambda **kw: captured.update(kw) or MagicMock()
    google_mod.genai = genai_mod
    with patch.dict("sys.modules", {"google": google_mod, "google.genai": genai_mod}):
        agent._build_client()
    assert captured.get("api_key") == "ai-studio-key"
    assert "vertexai" not in captured


def test_gemini_agent_falls_back_when_creds_missing(test_settings):
    # USE_VERTEX_AI true but no GCP creds file → not vertex; AI Studio key present.
    test_settings.USE_VERTEX_AI = True
    test_settings.GCP_PROJECT_ID = "proj"
    test_settings.GOOGLE_APPLICATION_CREDENTIALS = "/nonexistent/sa.json"
    test_settings.GEMINI_API_KEY = "ai-studio-key"
    assert test_settings.gemini_auth_mode == "ai_studio"


def test_client_is_cached(test_settings, tmp_path):
    settings = _vertex_settings(test_settings, tmp_path)
    agent = GeminiAgent("gemini-2.5-flash-lite", settings)

    sentinel = object()
    with patch.object(agent, "_build_client", return_value=sentinel) as mb:
        c1 = agent._get_client()
        c2 = agent._get_client()
    assert c1 is sentinel and c2 is sentinel
    mb.assert_called_once()  # built only once, then cached


# ---------------------------------------------------------------------------
# Apify discovery disable guards
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_crunchbase_disabled_returns_empty_without_api_call(test_settings):
    from sources.crunchbase_apify import CrunchbaseAPIfyClient

    test_settings.ENABLE_CRUNCHBASE_DISCOVERY = False
    client = CrunchbaseAPIfyClient(test_settings)

    with patch.object(CrunchbaseAPIfyClient, "_request") as mock_req:
        out = await client.search_recent_funding(industries=["541512"], limit=10)
    assert out == []
    mock_req.assert_not_called()


@pytest.mark.asyncio
async def test_wellfound_disabled_returns_empty_without_api_call(test_settings):
    from sources.wellfound_apify import WellfoundAPIfyClient

    test_settings.ENABLE_WELLFOUND_DISCOVERY = False
    client = WellfoundAPIfyClient(test_settings)

    with patch.object(WellfoundAPIfyClient, "_request") as mock_req:
        out = await client.search_recent_startups(industries=["611710"], limit=10)
    assert out == []
    mock_req.assert_not_called()


def test_default_settings_prefer_vertex_and_disable_discovery(test_settings):
    """Code defaults: Vertex preferred, discovery off."""
    from config.settings import Settings
    s = Settings(_env_file=None, GEMINI_API_KEY="k")
    assert s.USE_VERTEX_AI is True
    assert s.ENABLE_CRUNCHBASE_DISCOVERY is False
    assert s.ENABLE_WELLFOUND_DISCOVERY is False

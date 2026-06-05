"""Minimal Gemini text-generation helper.

Picks the auth path based on ``settings.gemini_auth_mode``:
  - "ai_studio": uses google-generativeai with GEMINI_API_KEY
  - "vertex":   uses vertexai with the configured project/region + service account

Exposes a single async ``generate_json_text`` that returns the raw model text.
Network/SDK errors are swallowed and an empty string is returned so callers can
degrade gracefully (the data layer never raises on normal failures).
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

from config.logging_config import setup_logging
from config.settings import Settings

_log = setup_logging("source.gemini")


def _generate_sync(settings: Settings, model: str, prompt: str) -> str:
    mode = settings.gemini_auth_mode
    if mode == "ai_studio":
        import google.generativeai as genai

        genai.configure(api_key=settings.GEMINI_API_KEY)
        gm = genai.GenerativeModel(model)
        resp = gm.generate_content(prompt)
        return (getattr(resp, "text", "") or "").strip()

    if mode == "vertex":
        os.environ.setdefault(
            "GOOGLE_APPLICATION_CREDENTIALS", settings.GOOGLE_APPLICATION_CREDENTIALS
        )
        import vertexai
        from vertexai.generative_models import GenerativeModel

        vertexai.init(project=settings.GCP_PROJECT_ID, location=settings.GCP_REGION)
        gm = GenerativeModel(model)
        resp = gm.generate_content(prompt)
        return (getattr(resp, "text", "") or "").strip()

    raise RuntimeError("No Gemini auth configured (set GEMINI_API_KEY or Vertex).")


async def generate_text(
    settings: Settings, model: str, prompt: str
) -> str:
    """Return generated text for ``prompt`` using ``model``; '' on failure."""
    try:
        return await asyncio.to_thread(_generate_sync, settings, model, prompt)
    except Exception as exc:  # noqa: BLE001
        _log.error("Gemini generation failed (model=%s): %s", model, exc)
        return ""

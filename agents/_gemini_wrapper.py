"""Agent-level Gemini client.

A richer wrapper than ``sources/_gemini.py`` (which is a minimal text helper for
the RSS parser). This one supports:

  - structured JSON output validated into a Pydantic model (``generate_json``)
  - free-form prose (``generate_text``)
  - concurrent batched JSON generation (``batch_generate_json``)

Auth follows ``settings.gemini_auth_mode``:
  - "ai_studio": google-genai Client(api_key=...)
  - "vertex":   google-genai Client(vertexai=True, project=..., location=...)

The synchronous google-genai call is run via ``asyncio.to_thread`` so the public
surface stays async, the event loop is never blocked, and HTTP is interceptable
by respx in tests (the SDK's async transport is not respx-friendly).

Usage is recorded in the ``api_usage`` table with source ``gemini_{model}``.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import List, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from config.logging_config import setup_logging
from config.settings import Settings
from sources._utils import track_usage

T = TypeVar("T", bound=BaseModel)

_MAX_JSON_RETRIES = 2
_BATCH_CONCURRENCY = 5


class GeminiAgent:
    """Shared Gemini client for agents."""

    def __init__(self, model_name: str, settings: Settings) -> None:
        self.model_name = model_name
        self.settings = settings
        self.log = setup_logging(f"agent.gemini.{model_name}")
        self._usage_source = f"gemini_{model_name}"

    # ----- client construction -------------------------------------------------

    def _build_client(self):
        """Construct a google-genai Client for the active auth mode."""
        from google import genai

        mode = self.settings.gemini_auth_mode
        if mode == "ai_studio":
            return genai.Client(api_key=self.settings.GEMINI_API_KEY)
        if mode == "vertex":
            import os

            os.environ.setdefault(
                "GOOGLE_APPLICATION_CREDENTIALS",
                self.settings.GOOGLE_APPLICATION_CREDENTIALS,
            )
            return genai.Client(
                vertexai=True,
                project=self.settings.GCP_PROJECT_ID,
                location=self.settings.GCP_REGION,
            )
        raise RuntimeError(
            "No Gemini auth configured (set GEMINI_API_KEY or configure Vertex)."
        )

    # ----- low-level sync calls (run in worker threads) ------------------------

    def _generate_sync(
        self,
        prompt: str,
        temperature: float,
        max_tokens: Optional[int],
        as_json: bool,
        schema: Optional[Type[BaseModel]],
    ) -> tuple[str, Optional[int]]:
        """Return (text, total_token_count). Runs in a thread."""
        from google.genai import types

        client = self._build_client()

        config_kwargs: dict = {"temperature": temperature}
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens
        if as_json:
            config_kwargs["response_mime_type"] = "application/json"
            if schema is not None:
                config_kwargs["response_schema"] = schema

        config = types.GenerateContentConfig(**config_kwargs)
        resp = client.models.generate_content(
            model=self.model_name, contents=prompt, config=config
        )

        text = (getattr(resp, "text", "") or "").strip()
        tokens = None
        usage = getattr(resp, "usage_metadata", None)
        if usage is not None:
            tokens = getattr(usage, "total_token_count", None)
        return text, tokens

    async def _call(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: Optional[int] = None,
        as_json: bool = False,
        schema: Optional[Type[BaseModel]] = None,
    ) -> Optional[str]:
        """Perform one Gemini call with timing, logging and usage tracking."""
        start = time.perf_counter()
        try:
            text, tokens = await asyncio.to_thread(
                self._generate_sync, prompt, temperature, max_tokens, as_json, schema
            )
        except Exception as exc:  # noqa: BLE001
            self.log.error("Gemini call failed (model=%s): %s", self.model_name, exc)
            return None

        latency_ms = (time.perf_counter() - start) * 1000.0
        self.log.info(
            "gemini | model=%s | tokens=%s | latency=%.0fms | json=%s",
            self.model_name,
            tokens if tokens is not None else "unknown",
            latency_ms,
            as_json,
        )
        try:
            await track_usage(self._usage_source, 1, None, self.settings)
        except Exception as exc:  # noqa: BLE001
            self.log.warning("Failed to track gemini usage: %s", exc)
        return text

    # ----- public API ----------------------------------------------------------

    async def generate_json(
        self,
        prompt: str,
        schema: Type[T],
        temperature: float = 0.2,
    ) -> Optional[T]:
        """Generate JSON and parse it into ``schema``.

        Retries up to ``_MAX_JSON_RETRIES`` times on parse/validation errors,
        re-prompting with the error text. Returns ``None`` on hard failure.
        """
        current_prompt = prompt
        last_error: Optional[str] = None

        for attempt in range(_MAX_JSON_RETRIES + 1):
            text = await self._call(
                current_prompt,
                temperature=temperature,
                as_json=True,
                schema=schema,
            )
            if not text:
                last_error = "empty response"
            else:
                try:
                    return schema.model_validate_json(text)
                except ValidationError as exc:
                    last_error = f"schema validation failed: {exc}"
                except (json.JSONDecodeError, ValueError) as exc:
                    last_error = f"invalid JSON: {exc}"

            if attempt < _MAX_JSON_RETRIES:
                self.log.warning(
                    "generate_json retry %d/%d (%s)",
                    attempt + 1,
                    _MAX_JSON_RETRIES,
                    last_error,
                )
                current_prompt = (
                    f"{prompt}\n\n"
                    f"Your previous response could not be parsed: {last_error}. "
                    "Return ONLY valid JSON matching the required schema."
                )

        self.log.error("generate_json failed after retries: %s", last_error)
        return None

    async def generate_text(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Generate free-form prose. Returns '' on failure."""
        text = await self._call(
            prompt, temperature=temperature, max_tokens=max_tokens
        )
        return text or ""

    async def batch_generate_json(
        self,
        prompts: List[str],
        schema: Type[T],
        temperature: float = 0.2,
    ) -> List[Optional[T]]:
        """Generate JSON for many prompts concurrently (semaphore-limited)."""
        semaphore = asyncio.Semaphore(_BATCH_CONCURRENCY)

        async def _one(p: str) -> Optional[T]:
            async with semaphore:
                return await self.generate_json(p, schema, temperature)

        return await asyncio.gather(*(_one(p) for p in prompts))

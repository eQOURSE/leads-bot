"""Base class and shared retry/logging policy for HTTP-based source clients."""

from __future__ import annotations

import time
from typing import Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from config.logging_config import setup_logging
from config.settings import Settings
from sources._utils import track_usage


def _is_retryable(exc: BaseException) -> bool:
    """Retry only on transient HTTP failures: 5xx or 429."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status >= 500 or status == 429
    return False


# Shared tenacity policy used by every client's HTTP call.
retry_policy = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    retry=retry_if_exception(_is_retryable),
    reraise=True,
)


class BaseSourceClient:
    """Common functionality for HTTP source clients.

    Subclasses set ``source_name`` and use ``_request`` to perform calls that
    are automatically retried, timed, and logged.
    """

    source_name: str = "base"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.log = setup_logging(f"source.{self.source_name}")

    @retry_policy
    async def _request(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        *,
        credits_remaining: Optional[int] = None,
        **kwargs,
    ) -> httpx.Response:
        """Perform a single HTTP request with retry, timing, and logging.

        Raises ``httpx.HTTPStatusError`` on non-2xx so the retry policy and the
        calling method's try/except can react. Callers are expected to catch
        errors and return typed-empty results for normal failures.
        """
        start = time.perf_counter()
        response = await client.request(method, url, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000.0

        self.log.info(
            "%s | %s %s | status=%s | latency=%.0fms | credits_remaining=%s",
            self.source_name,
            method.upper(),
            url,
            response.status_code,
            latency_ms,
            credits_remaining if credits_remaining is not None else "unknown",
        )

        response.raise_for_status()
        return response

    async def _track(
        self, credits: int, remaining: Optional[int] = None
    ) -> None:
        """Record usage for this source in the ``api_usage`` table."""
        try:
            await track_usage(self.source_name, credits, remaining, self.settings)
        except Exception as exc:  # noqa: BLE001 - usage tracking is best-effort
            self.log.warning("Failed to track usage for %s: %s", self.source_name, exc)

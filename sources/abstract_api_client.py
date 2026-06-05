"""AbstractAPI Email Validation client.

Free tier: 100 validations/month.
Budget is tracked via the api_usage table (source='abstract_api').
Remaining quota is inferred: 100 - monthly_used (no API endpoint for this).

Caching: results cached 30 days keyed by email to avoid burning quota on
the same address twice.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import httpx

from config.logging_config import setup_logging
from config.settings import Settings
from sources._cache import cache_get, cache_set
from sources._utils import month_call_count, track_usage

_DEFAULT_TIMEOUT = 20.0
_BASE_URL = "https://emailvalidation.abstractapi.com/v1/"
_MONTHLY_CAP = 100
_CACHE_TTL_DAYS = 30


class AbstractAPIClient:
    """Email validation via AbstractAPI with caching and quota tracking."""

    source_name = "abstract_api"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.api_key = settings.ABSTRACT_EMAIL_API_KEY
        self.log = setup_logging(f"source.{self.source_name}")
        self._exhausted = False  # set True on 429

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def validate_email(self, email: str) -> dict:
        """Validate one email address.

        Returns a dict with keys: deliverability, quality_score,
        is_valid_format, is_disposable_email, is_smtp_valid,
        is_role_email, is_catchall_email, is_mx_found.
        Returns {} on any error or quota exhaustion.
        """
        if self._exhausted:
            self.log.warning("abstract_api: quota exhausted for this run, skipping %s", email)
            return {}

        if not self.api_key:
            self.log.warning("abstract_api: ABSTRACT_EMAIL_API_KEY not set, skipping")
            return {}

        # Cache check
        cached = await cache_get(self.source_name, email, self.settings)
        if cached is not None:
            self.log.info("abstract_api: cache hit for %s", email)
            return cached

        try:
            async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
                resp = await client.get(
                    _BASE_URL,
                    params={"api_key": self.api_key, "email": email},
                )
        except Exception as exc:  # noqa: BLE001
            self.log.warning("abstract_api: request failed for %s: %s", email, exc)
            return {}

        if resp.status_code in (401, 403):
            self.log.error(
                "abstract_api: auth error %s — check ABSTRACT_EMAIL_API_KEY", resp.status_code
            )
            return {}

        if resp.status_code == 429:
            self.log.warning("abstract_api: quota exceeded (429) — disabling for this run")
            self._exhausted = True
            return {}

        if resp.status_code != 200:
            self.log.warning("abstract_api: unexpected status %s for %s", resp.status_code, email)
            return {}

        try:
            raw = resp.json()
        except Exception as exc:  # noqa: BLE001
            self.log.warning("abstract_api: invalid JSON for %s: %s", email, exc)
            return {}

        # Normalise: flatten .value out of nested dicts for easy access
        result = _flatten_abstract(raw)

        # Track usage and cache
        await track_usage(self.source_name, 1, await self.get_remaining_quota() - 1, self.settings)
        await cache_set(self.source_name, email, result, _CACHE_TTL_DAYS, self.settings)

        return result

    async def get_remaining_quota(self) -> int:
        """Return estimated remaining monthly quota (100 - used this month)."""
        used = await month_call_count(self.source_name, self.settings)
        return max(0, _MONTHLY_CAP - used)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _flatten_abstract(raw: dict) -> dict:
    """Flatten AbstractAPI's nested {value: bool} fields to plain booleans."""
    result: dict = {}
    for key, val in raw.items():
        if isinstance(val, dict) and "value" in val:
            result[key] = val["value"]
        else:
            result[key] = val
    return result

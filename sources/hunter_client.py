"""Hunter.io API client.

Free tier is only 25 searches/month, so usage is checked against the
``api_usage`` table before each call. A configurable soft limit
(``HUNTER_MONTHLY_CALL_LIMIT``, default 50) guards against overspend; pass
``force=True`` to override.
"""

from __future__ import annotations

from typing import Optional

import httpx

from config.settings import Settings
from sources._utils import month_call_count
from sources.base import BaseSourceClient
from sources.models import ProspectCandidate

_DEFAULT_TIMEOUT = 30.0


class HunterClient(BaseSourceClient):
    source_name = "hunter"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.api_key = settings.HUNTER_API_KEY
        self.base_url = settings.HUNTER_BASE_URL.rstrip("/")
        self.monthly_limit = settings.HUNTER_MONTHLY_CALL_LIMIT

    async def _over_limit(self, force: bool) -> bool:
        if force:
            return False
        used = await month_call_count(self.source_name, self.settings)
        if used >= self.monthly_limit:
            self.log.warning(
                "Hunter monthly call budget reached (%s/%s) — skipping call. "
                "Pass force=True to override.",
                used,
                self.monthly_limit,
            )
            return True
        return False

    async def domain_search(self, domain: str, force: bool = False) -> dict:
        if await self._over_limit(force):
            return {}

        url = f"{self.base_url}/domain-search"
        params = {"domain": domain, "api_key": self.api_key}
        try:
            async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
                resp = await self._request(client, "GET", url, params=params)
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            self.log.error("hunter.domain_search failed for %s: %s", domain, exc)
            return {}

        remaining = self._remaining_from(data)
        await self._track(1, remaining)
        return data.get("data", {})

    async def email_finder(
        self,
        domain: str,
        first_name: str,
        last_name: str,
        force: bool = False,
    ) -> Optional[ProspectCandidate]:
        if await self._over_limit(force):
            return None

        url = f"{self.base_url}/email-finder"
        params = {
            "domain": domain,
            "first_name": first_name,
            "last_name": last_name,
            "api_key": self.api_key,
        }
        try:
            async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
                resp = await self._request(client, "GET", url, params=params)
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            self.log.error("hunter.email_finder failed for %s: %s", domain, exc)
            return None

        remaining = self._remaining_from(data)
        await self._track(1, remaining)

        payload = data.get("data", {})
        if not payload.get("email"):
            return None
        try:
            return ProspectCandidate(
                full_name=f"{first_name} {last_name}".strip(),
                title=payload.get("position") or "",
                company_domain=domain,
                linkedin_url=payload.get("linkedin_url"),
                email=payload.get("email"),
                email_confidence=payload.get("score"),
                phone=payload.get("phone_number"),
                source="hunter",
            )
        except Exception as exc:  # noqa: BLE001
            self.log.warning("hunter.email_finder produced malformed prospect: %s", exc)
            return None

    async def get_account_info(self) -> dict:
        url = f"{self.base_url}/account"
        params = {"api_key": self.api_key}
        try:
            async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
                resp = await self._request(client, "GET", url, params=params)
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            self.log.error("hunter.get_account_info failed: %s", exc)
            return {}
        # account info is metadata; do not bill a usage credit
        return data.get("data", {})

    @staticmethod
    def _remaining_from(data: dict) -> Optional[int]:
        meta = data.get("meta") or {}
        results = meta.get("results")
        used = meta.get("used")
        limit = meta.get("limit")
        if limit is not None and used is not None:
            try:
                return int(limit) - int(used)
            except (TypeError, ValueError):
                return None
        if isinstance(results, int):
            return None
        return None

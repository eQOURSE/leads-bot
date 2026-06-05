"""thecompaniesapi.com client.

Free tier: 500 credits. Auth is a Bearer token. Search is a GET request whose
``query`` parameter is a JSON-encoded ARRAY of condition objects, each with
``attribute``, ``operator``, ``sign`` and ``values``. Base URL from settings.
"""

from __future__ import annotations

import json
from typing import Optional

import httpx

from config.settings import Settings
from sources.base import BaseSourceClient
from sources.models import CompanyCandidate

_DEFAULT_TIMEOUT = 30.0


class CompaniesAPIClient(BaseSourceClient):
    source_name = "companies_api"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.token = settings.COMPANIES_API_TOKEN
        self.base_url = settings.COMPANIES_API_BASE_URL.rstrip("/")

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
        }

    async def search_by_filters(
        self,
        industries: Optional[list[str]] = None,
        countries: Optional[list[str]] = None,
        employee_range: Optional[tuple[int, int]] = None,
        limit: int = 25,
    ) -> list[CompanyCandidate]:
        # The API expects a flat array of condition objects.
        conditions: list[dict] = []
        if industries:
            conditions.append(
                {
                    "attribute": "about.industries",
                    "operator": "or",
                    "sign": "equals",
                    "values": industries,
                }
            )
        if countries:
            conditions.append(
                {
                    "attribute": "locations.headquarters.country.code",
                    "operator": "or",
                    "sign": "equals",
                    "values": countries,
                }
            )
        if employee_range:
            conditions.append(
                {
                    "attribute": "about.totalEmployees",
                    "operator": "and",
                    "sign": "between",
                    "values": [employee_range[0], employee_range[1]],
                }
            )

        url = f"{self.base_url}/companies"
        params = {"size": limit}
        if conditions:
            params["query"] = json.dumps(conditions)
        try:
            async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
                resp = await self._request(
                    client, "GET", url, headers=self._headers(), params=params
                )
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            self.log.error("companies_api.search_by_filters failed: %s", exc)
            return []

        items = data.get("companies") or data.get("results") or []
        meta = data.get("meta") or {}
        await self._track(meta.get("cost", len(items)), meta.get("credits"))
        return [c for c in (self._parse(i) for i in items) if c is not None]

    async def enrich_by_domain(self, domain: str) -> Optional[CompanyCandidate]:
        url = f"{self.base_url}/companies/{domain}"
        return await self._enrich_single(url, source_label=f"domain:{domain}")

    async def enrich_by_email(self, email: str) -> Optional[CompanyCandidate]:
        url = f"{self.base_url}/companies/by-email/{email}"
        return await self._enrich_single(url, source_label=f"email:{email}")

    async def _enrich_single(
        self, url: str, source_label: str
    ) -> Optional[CompanyCandidate]:
        try:
            async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
                resp = await self._request(client, "GET", url, headers=self._headers())
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            self.log.error("companies_api enrich failed (%s): %s", source_label, exc)
            return None

        meta = data.get("meta") or {}
        await self._track(meta.get("cost", 1), meta.get("credits"))
        company = data.get("company") or data
        return self._parse(company)

    @staticmethod
    def _extract_domain(item: dict) -> Optional[str]:
        """Domain may be a bare string or a nested object with a ``domain`` key."""
        raw = item.get("domain") or item.get("website")
        if isinstance(raw, dict):
            return raw.get("domain") or raw.get("alias")
        return raw

    def _parse(self, item: dict) -> Optional[CompanyCandidate]:
        if not isinstance(item, dict):
            return None
        about = item.get("about") if isinstance(item.get("about"), dict) else {}
        domain = self._extract_domain(item)
        name = item.get("name") or about.get("name")
        if not domain or not name:
            return None

        # Headquarters location is nested under locations in the v2 response.
        locations = item.get("locations") if isinstance(item.get("locations"), dict) else {}
        hq = locations.get("headquarters") if isinstance(locations.get("headquarters"), dict) else {}
        hq_country = None
        hq_region = None
        if isinstance(hq.get("country"), dict):
            hq_country = hq["country"].get("name") or hq["country"].get("code")
        if isinstance(hq.get("region"), dict):
            hq_region = hq["region"].get("name")

        industries = about.get("industries") if isinstance(about.get("industries"), list) else None
        try:
            return CompanyCandidate(
                domain=domain,
                name=name,
                description=item.get("description")
                or (item.get("descriptions") or {}).get("primary")
                if isinstance(item.get("descriptions"), dict)
                else item.get("description"),
                industry=industries[0] if industries else item.get("industry"),
                naics_codes=item.get("naics_codes") or [],
                size_range=str(about.get("totalEmployees"))
                if about.get("totalEmployees")
                else None,
                hq_country=hq_country,
                hq_region=hq_region,
                website=domain,
                raw_source="companies_api",
                confidence=0.6,
            )
        except Exception as exc:  # noqa: BLE001
            self.log.warning("companies_api company skipped: %s", exc)
            return None

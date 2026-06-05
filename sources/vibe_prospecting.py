"""Vibe Prospecting (Explorium) data source client.

NOTE: "Vibe Prospecting" is Explorium's product; the underlying API is the
Explorium Data API. Exact endpoint paths will be provided later, so the path
segments are defined as class attributes and can be swapped without touching
the request logic. The base URL comes from settings.
"""

from __future__ import annotations

from typing import Optional

import httpx

from config.settings import Settings
from sources.base import BaseSourceClient
from sources.models import CompanyCandidate, ProspectCandidate

_DEFAULT_TIMEOUT = 30.0


class VibeProspectingClient(BaseSourceClient):
    source_name = "vibe_prospecting"

    # Endpoint paths (relative to settings.VIBE_PROSPECTING_BASE_URL).
    # Swap these once the exact Explorium paths are confirmed.
    PATH_SEARCH_COMPANIES = "/businesses/search"
    PATH_FIND_PROSPECTS = "/prospects/search"
    PATH_ENRICH_CONTACTS = "/prospects/contacts/enrich"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.api_key = settings.VIBE_PROSPECTING_API_KEY
        self.base_url = settings.VIBE_PROSPECTING_BASE_URL.rstrip("/")

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def search_funded_companies(
        self,
        naics_codes: Optional[list[str]] = None,
        linkedin_categories: Optional[list[str]] = None,
        funding_window_days: int = 90,
        country_codes: Optional[list[str]] = None,
        size_ranges: Optional[list[str]] = None,
        limit: int = 100,
    ) -> list[CompanyCandidate]:
        if country_codes is None:
            country_codes = ["US"]
        if size_ranges is None:
            size_ranges = ["11-50", "51-200", "201-500"]

        payload = {
            "filters": {
                "naics_codes": naics_codes or [],
                "linkedin_categories": linkedin_categories or [],
                "funding_window_days": funding_window_days,
                "country_codes": country_codes,
                "size_ranges": size_ranges,
            },
            "limit": limit,
        }
        url = f"{self.base_url}{self.PATH_SEARCH_COMPANIES}"

        try:
            async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
                resp = await self._request(
                    client, "POST", url, headers=self._headers(), json=payload
                )
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            self.log.error("vibe_prospecting.search_funded_companies failed: %s", exc)
            return []

        items = data.get("data") or data.get("businesses") or data.get("results") or []
        remaining = data.get("credits_remaining")
        await self._track(len(items), remaining)
        return [c for c in (self._parse_company(i) for i in items) if c is not None]

    async def find_prospects(
        self,
        business_domains: list[str],
        job_titles: Optional[list[str]] = None,
        job_departments: Optional[list[str]] = None,
        job_levels: Optional[list[str]] = None,
        has_email: bool = True,
    ) -> list[ProspectCandidate]:
        payload = {
            "business_domains": business_domains,
            "job_titles": job_titles or [],
            "job_departments": job_departments or [],
            "job_levels": job_levels or [],
            "has_email": has_email,
        }
        url = f"{self.base_url}{self.PATH_FIND_PROSPECTS}"

        try:
            async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
                resp = await self._request(
                    client, "POST", url, headers=self._headers(), json=payload
                )
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            self.log.error("vibe_prospecting.find_prospects failed: %s", exc)
            return []

        items = data.get("data") or data.get("prospects") or data.get("results") or []
        remaining = data.get("credits_remaining")
        await self._track(len(items), remaining)
        return [p for p in (self._parse_prospect(i) for i in items) if p is not None]

    async def enrich_prospect_contacts(
        self,
        prospect_ids: list[str],
        contact_types: Optional[list[str]] = None,
    ) -> list[ProspectCandidate]:
        if contact_types is None:
            contact_types = ["email", "phone"]

        payload = {"prospect_ids": prospect_ids, "contact_types": contact_types}
        url = f"{self.base_url}{self.PATH_ENRICH_CONTACTS}"

        try:
            async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
                resp = await self._request(
                    client, "POST", url, headers=self._headers(), json=payload
                )
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            self.log.error("vibe_prospecting.enrich_prospect_contacts failed: %s", exc)
            return []

        items = data.get("data") or data.get("prospects") or data.get("results") or []
        remaining = data.get("credits_remaining")
        await self._track(len(items), remaining)
        return [p for p in (self._parse_prospect(i) for i in items) if p is not None]

    # --- parsing helpers ---

    def _parse_company(self, item: dict) -> Optional[CompanyCandidate]:
        domain = item.get("domain") or item.get("website")
        name = item.get("name") or item.get("company_name")
        if not domain or not name:
            return None
        try:
            return CompanyCandidate(
                domain=domain,
                name=name,
                description=item.get("description"),
                industry=item.get("industry"),
                naics_codes=item.get("naics_codes") or [],
                linkedin_category=item.get("linkedin_category"),
                size_range=item.get("size_range"),
                revenue_range=item.get("revenue_range"),
                hq_country=item.get("hq_country") or item.get("country"),
                hq_region=item.get("hq_region") or item.get("region"),
                website=item.get("website"),
                funding_amount_usd=item.get("funding_amount_usd"),
                funding_stage=item.get("funding_stage"),
                funding_date=item.get("funding_date"),
                funding_source=item.get("funding_source") or "vibe_prospecting",
                raw_source="vibe_prospecting",
                confidence=float(item.get("confidence", 0.7)),
            )
        except Exception as exc:  # noqa: BLE001
            self.log.warning("Skipping malformed company: %s", exc)
            return None

    def _parse_prospect(self, item: dict) -> Optional[ProspectCandidate]:
        full_name = item.get("full_name") or item.get("name")
        title = item.get("title") or item.get("job_title") or ""
        domain = item.get("company_domain") or item.get("domain")
        if not full_name or not domain:
            return None
        try:
            return ProspectCandidate(
                full_name=full_name,
                title=title,
                company_domain=domain,
                linkedin_url=item.get("linkedin_url"),
                email=item.get("email"),
                email_confidence=item.get("email_confidence"),
                phone=item.get("phone"),
                source="vibe_prospecting",
            )
        except Exception as exc:  # noqa: BLE001
            self.log.warning("Skipping malformed prospect: %s", exc)
            return None

"""Phase 11 — Crunchbase discovery via an Apify actor.

ACTOR SELECTION (verify before relying on live output)
------------------------------------------------------
Apify store actors change over time. This client targets a Crunchbase search
actor with structured firmographic output. Default actor:

    actor: epctex/crunchbase-scraper
    url:   https://apify.com/epctex/crunchbase-scraper

Criteria used to pick it: >4 stars, recent updates, results-based pricing.
If that actor is deprecated or its schema changes, set CRUNCHBASE_APIFY_ACTOR
in .env to override, and adjust _map_item() to match the new output fields.

The client is fault-tolerant: any failure (bad actor, schema mismatch, no
credits) returns [] and logs — it never breaks the hunt. If the actor returns
0 items on 3 consecutive runs, an ERROR is logged (actor likely broken).

Budget: capped via ``maxItems`` so a run stays under ~$1.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import httpx

from agents._constants import crunchbase_categories_for_icp
from config.settings import Settings
from sources._utils import normalize_domain, utcnow
from sources.base import BaseSourceClient
from sources.models import CompanyCandidate

_DEFAULT_ACTOR = "epctex~crunchbase-scraper"
_TIMEOUT = 180.0
_COST_PER_RUN = 1.0  # conservative cap estimate (USD)

# Module-level zero-result streak tracker (across instances within a process).
_zero_streak = {"crunchbase": 0}


class CrunchbaseAPIfyClient(BaseSourceClient):
    source_name = "crunchbase"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.tokens = settings.apify_tokens
        self.actor = getattr(settings, "CRUNCHBASE_APIFY_ACTOR", None) or _DEFAULT_ACTOR

    def _pick_token(self) -> Optional[str]:
        return self.tokens[0] if self.tokens else None

    async def search_recent_funding(
        self,
        industries: list[str],
        funding_stages: Optional[list[str]] = None,
        days_back: int = 240,
        country: str = "United States",
        limit: int = 50,
        keywords: Optional[list[str]] = None,
    ) -> list[CompanyCandidate]:
        """Return recently-funded companies as CompanyCandidates. [] on failure."""
        if not getattr(self.settings, "ENABLE_CRUNCHBASE_DISCOVERY", False):
            self.log.info("CrunchbaseAPIfyClient disabled via ENABLE_CRUNCHBASE_DISCOVERY")
            return []
        token = self._pick_token()
        if not token:
            self.log.warning("crunchbase: no Apify token available")
            return []

        funding_stages = funding_stages or ["seed", "series_a"]
        categories = crunchbase_categories_for_icp(industries, keywords)
        cutoff = (date.today() - timedelta(days=days_back)).isoformat()

        # Generic, defensive input shape. Unknown keys are ignored by most actors.
        run_input = {
            "searchType": "organizations",
            "categories": categories or industries,
            "fundingStages": funding_stages,
            "announcedOnAfter": cutoff,
            "locations": [country],
            "maxItems": int(limit),          # budget cap
            "proxyConfiguration": {"useApifyProxy": True},
        }

        url = (
            f"{self.settings.APIFY_BASE_URL.rstrip('/')}"
            f"/acts/{self.actor}/run-sync-get-dataset-items"
        )
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await self._request(
                    client, "POST", url, params={"token": token}, json=run_input
                )
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            self.log.error("crunchbase actor failed: %s", exc)
            return []

        items = data if isinstance(data, list) else data.get("items", []) if isinstance(data, dict) else []

        # Zero-result streak monitoring.
        if not items:
            _zero_streak["crunchbase"] += 1
            if _zero_streak["crunchbase"] >= 3:
                self.log.error(
                    "crunchbase: 0 items for %d consecutive runs — actor %s may be broken",
                    _zero_streak["crunchbase"], self.actor,
                )
            else:
                self.log.warning("crunchbase: 0 items returned this run")
        else:
            _zero_streak["crunchbase"] = 0

        await self._track(int(_COST_PER_RUN * 100), None)

        candidates: list[CompanyCandidate] = []
        for item in items:
            mapped = self._map_item(item)
            if mapped is not None:
                candidates.append(mapped)

        self.log.info("crunchbase: %d candidates mapped from %d items", len(candidates), len(items))
        return candidates

    def _map_item(self, item: dict) -> Optional[CompanyCandidate]:
        """Map an actor result item → CompanyCandidate. Tolerant of field names."""
        if not isinstance(item, dict):
            return None

        name = item.get("name") or item.get("companyName") or item.get("organization") or ""
        if not name:
            return None

        raw_domain = (
            item.get("domain") or item.get("website") or item.get("homepage_url")
            or item.get("url") or ""
        )
        domain = normalize_domain(raw_domain) if raw_domain else ""
        if not domain or "." not in domain:
            slug = "".join(ch for ch in name.lower() if ch.isalnum())
            domain = f"{slug}.unknown" if slug else "unknown.unknown"

        funding_amount = item.get("funding_total") or item.get("lastFundingAmount") or item.get("funding_amount")
        funding_stage = item.get("last_funding_type") or item.get("fundingStage") or item.get("funding_stage")
        funding_date = self._parse_date(
            item.get("last_funding_at") or item.get("fundingDate") or item.get("announced_on")
        )
        size_range = self._map_employee_count(
            item.get("employee_count") or item.get("employeeCount") or item.get("num_employees")
        )

        try:
            return CompanyCandidate(
                domain=domain,
                name=name,
                description=item.get("short_description") or item.get("description"),
                industry=item.get("category") or item.get("industry"),
                size_range=size_range,
                hq_country=item.get("country") or item.get("hq_country"),
                hq_region=item.get("region") or item.get("city") or item.get("hq_region"),
                website=raw_domain or None,
                funding_amount_usd=self._to_float(funding_amount),
                funding_stage=funding_stage,
                funding_date=funding_date,
                funding_source="crunchbase",
                raw_source="crunchbase",
                confidence=0.85,
            )
        except Exception as exc:  # noqa: BLE001
            self.log.warning("crunchbase item skipped: %s", exc)
            return None

    @staticmethod
    def _to_float(v) -> Optional[float]:
        if v is None:
            return None
        try:
            if isinstance(v, str):
                v = v.replace("$", "").replace(",", "").strip()
            return float(v)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_date(v):
        if not v:
            return None
        from datetime import datetime
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
            try:
                return datetime.strptime(str(v)[:len(fmt) + 2], fmt).date()
            except (ValueError, TypeError):
                continue
        try:
            return datetime.fromisoformat(str(v).replace("Z", "")).date()
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _map_employee_count(v) -> Optional[str]:
        """Map a raw count or range to our ICP size buckets."""
        if v is None:
            return None
        s = str(v)
        # Already a range like "11-50"
        if "-" in s:
            return s
        try:
            n = int(float(s))
        except (TypeError, ValueError):
            return None
        if n <= 10:
            return "1-10"
        if n <= 50:
            return "11-50"
        if n <= 200:
            return "51-200"
        if n <= 500:
            return "201-500"
        return "500+"

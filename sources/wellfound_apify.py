"""Phase 11 — AngelList / Wellfound discovery via an Apify actor.

ACTOR SELECTION (verify before relying on live output)
------------------------------------------------------
Default actor:

    actor: curious_coder/wellfound-scraper
    url:   https://apify.com/curious_coder/wellfound-scraper

Criteria: recently updated, results-based pricing. Override with
WELLFOUND_APIFY_ACTOR in .env if the actor changes; adjust _map_item() to
match the new schema.

Fault-tolerant: any failure returns [] and logs; never breaks the hunt.
Budget capped via ``maxItems`` (~$1/run). Logs ERROR after 3 consecutive
zero-result runs.
"""

from __future__ import annotations

from typing import Optional

import httpx

from agents._constants import wellfound_markets_for_icp
from config.settings import Settings
from sources._utils import normalize_domain
from sources.base import BaseSourceClient
from sources.models import CompanyCandidate

_DEFAULT_ACTOR = "curious_coder~wellfound-scraper"
_TIMEOUT = 180.0
_COST_PER_RUN = 1.0

_zero_streak = {"wellfound": 0}


class WellfoundAPIfyClient(BaseSourceClient):
    source_name = "wellfound"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.tokens = settings.apify_tokens
        self.actor = getattr(settings, "WELLFOUND_APIFY_ACTOR", None) or _DEFAULT_ACTOR

    def _pick_token(self) -> Optional[str]:
        return self.tokens[0] if self.tokens else None

    async def search_recent_startups(
        self,
        industries: list[str],
        founded_after_year: int = 2022,
        funding_min: int = 500_000,
        limit: int = 50,
        keywords: Optional[list[str]] = None,
    ) -> list[CompanyCandidate]:
        """Return startups matching the markets as CompanyCandidates. [] on failure."""
        token = self._pick_token()
        if not token:
            self.log.warning("wellfound: no Apify token available")
            return []

        markets = wellfound_markets_for_icp(keywords) or industries

        run_input = {
            "markets": markets,
            "foundedAfter": int(founded_after_year),
            "minFunding": int(funding_min),
            "maxItems": int(limit),
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
            self.log.error("wellfound actor failed: %s", exc)
            return []

        items = data if isinstance(data, list) else data.get("items", []) if isinstance(data, dict) else []

        if not items:
            _zero_streak["wellfound"] += 1
            if _zero_streak["wellfound"] >= 3:
                self.log.error(
                    "wellfound: 0 items for %d consecutive runs — actor %s may be broken",
                    _zero_streak["wellfound"], self.actor,
                )
            else:
                self.log.warning("wellfound: 0 items returned this run")
        else:
            _zero_streak["wellfound"] = 0

        await self._track(int(_COST_PER_RUN * 100), None)

        candidates: list[CompanyCandidate] = []
        for item in items:
            mapped = self._map_item(item)
            if mapped is not None:
                candidates.append(mapped)

        self.log.info("wellfound: %d candidates mapped from %d items", len(candidates), len(items))
        return candidates

    def _map_item(self, item: dict) -> Optional[CompanyCandidate]:
        if not isinstance(item, dict):
            return None

        name = item.get("name") or item.get("companyName") or item.get("startupName") or ""
        if not name:
            return None

        raw_domain = item.get("website") or item.get("domain") or item.get("companyUrl") or ""
        domain = normalize_domain(raw_domain) if raw_domain else ""
        if not domain or "." not in domain:
            slug = "".join(ch for ch in name.lower() if ch.isalnum())
            domain = f"{slug}.unknown" if slug else "unknown.unknown"

        size_range = self._map_employee_count(
            item.get("employee_count") or item.get("companySize") or item.get("teamSize")
        )

        try:
            return CompanyCandidate(
                domain=domain,
                name=name,
                description=item.get("description") or item.get("highConcept"),
                industry=(item.get("markets") or [None])[0] if isinstance(item.get("markets"), list) else item.get("market"),
                size_range=size_range,
                hq_country=item.get("country") or item.get("hq_country"),
                hq_region=item.get("location") or item.get("hq_region"),
                website=raw_domain or None,
                funding_amount_usd=self._to_float(item.get("funding_total") or item.get("totalRaised")),
                funding_stage=item.get("stage") or item.get("fundingStage"),
                funding_source="wellfound",
                raw_source="wellfound",
                confidence=0.8,
            )
        except Exception as exc:  # noqa: BLE001
            self.log.warning("wellfound item skipped: %s", exc)
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
    def _map_employee_count(v) -> Optional[str]:
        if v is None:
            return None
        s = str(v)
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

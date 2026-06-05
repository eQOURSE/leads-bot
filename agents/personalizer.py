"""Personalizer agent — Phase 7.

For each company in an EnrichedResult that has at least one sendable DM,
builds a PersonalizationContext by:
  1. Extracting recent news from the company's website (ScrapeGraph)
  2. Searching for press coverage (NewsData)
  3. Synthesising a specific hook via Gemini Flash

Results are cached per domain for 24 hours to avoid redundant API spend on
re-runs.

Budget guardrails per run (defaults):
  - scrapegraph_cap_per_run = 15
  - newsdata_cap_per_run    = 15
"""

from __future__ import annotations

import json
import time
from datetime import timedelta
from typing import TYPE_CHECKING, Optional

from agents._models import (
    EnrichedResult,
    IcpStrategy,
    PersonalizationContext,
    QualifiedCandidate,
)
from config.logging_config import setup_logging
from config.settings import Settings
from sources._cache import cache_get, cache_set
from sources._utils import utcnow

if TYPE_CHECKING:
    from agents.icp_strategist import IcpStrategist
    from agents._gemini_wrapper import GeminiAgent
    from sinks.sqlite_store import LeadStore
    from sources.newsdata_client import NewsDataClient
    from sources.scrapegraph_client import ScrapeGraphClient

_CACHE_METHOD = "personalizer_hook"
_CACHE_TTL_HOURS = 24
_CACHE_TTL_DAYS = 1   # SQLite cache uses whole days; 24h ≈ 1 day


class Personalizer:
    """Build per-company personalization hooks from recent news + Gemini synthesis."""

    def __init__(
        self,
        settings: Settings,
        icp_strategist: "IcpStrategist",
        gemini_agent: "GeminiAgent",          # should use GEMINI_MODEL_PERSONALIZER
        scrapegraph_client: "ScrapeGraphClient",
        newsdata_client: "NewsDataClient",
        lead_store: "LeadStore",
    ) -> None:
        self.settings = settings
        self.icp_strategist = icp_strategist
        self.gemini = gemini_agent
        self.scrapegraph = scrapegraph_client
        self.newsdata = newsdata_client
        self.lead_store = lead_store
        self.log = setup_logging("agent.personalizer")

    # =========================================================================
    # Public API
    # =========================================================================

    async def build_hooks_for_enriched_result(
        self,
        enriched_result: EnrichedResult,
        scrapegraph_cap_per_run: int = 15,
        newsdata_cap_per_run: int = 15,
        cache_ttl_hours: int = _CACHE_TTL_HOURS,
    ) -> dict[str, PersonalizationContext]:
        """Return domain → PersonalizationContext for all sendable companies."""
        start_ts = time.perf_counter()
        segment = enriched_result.segment

        icp = self.icp_strategist.load_strategy(segment)

        sg_remaining = scrapegraph_cap_per_run
        nd_remaining = newsdata_cap_per_run

        # Collect unique (domain, qualified_candidate) pairs with at least one
        # sendable DM (email confidence > 0), tier_1 first.
        seen_domains: set[str] = set()
        work_items: list[tuple[str, QualifiedCandidate, str]] = []  # (domain, qc, tier)

        for priority_tier in ("tier_1", "tier_2"):
            for ec in enriched_result.enriched_candidates:
                qc: QualifiedCandidate = ec.candidate_with_people.qualified
                if qc.tier != priority_tier:
                    continue
                domain: str = getattr(qc.candidate, "domain", "") or ""  # type: ignore[attr-defined]
                if not domain or domain in seen_domains:
                    continue
                has_sendable = any(
                    edm.email_result.email and edm.email_result.confidence > 0
                    for edm in ec.enriched_dms
                )
                if has_sendable:
                    seen_domains.add(domain)
                    work_items.append((domain, qc, priority_tier))

        result: dict[str, PersonalizationContext] = {}
        api_credits: dict[str, int] = {"scrapegraph": 0, "newsdata": 0, "gemini": 0}

        for domain, qc, _tier in work_items:
            # Cache check
            cached = await cache_get(_CACHE_METHOD, domain, self.settings)
            if cached is not None:
                self.log.info("personalizer: cache hit for %s", domain)
                try:
                    result[domain] = PersonalizationContext(**cached)
                    continue
                except Exception:  # noqa: BLE001
                    pass  # stale / malformed cache, rebuild

            candidate = qc.candidate  # type: ignore[attr-defined]
            name: str = getattr(candidate, "name", "") or domain
            website: Optional[str] = getattr(candidate, "website", None)
            base_url = website or f"https://{domain}"

            # ScrapeGraph: recent news from company site
            scrape_data: dict = {}
            if sg_remaining > 0:
                try:
                    scrape_data = await self.scrapegraph.extract_recent_news(base_url)
                    sg_remaining -= 1
                    api_credits["scrapegraph"] += 1
                    if sg_remaining == 0:
                        self.log.info("personalizer: ScrapeGraph cap reached (%d)", scrapegraph_cap_per_run)
                except Exception as exc:  # noqa: BLE001
                    self.log.warning("personalizer: scrapegraph failed for %s: %s", domain, exc)

            # NewsData: recent press coverage
            news_items: list = []
            if nd_remaining > 0:
                try:
                    news_items = await self.newsdata.search_company_news(name, days_back=60)
                    nd_remaining -= 1
                    api_credits["newsdata"] += 1
                    if nd_remaining == 0:
                        self.log.info("personalizer: NewsData cap reached (%d)", newsdata_cap_per_run)
                except Exception as exc:  # noqa: BLE001
                    self.log.warning("personalizer: newsdata failed for %s: %s", name, exc)

            # Gemini Flash: synthesize hook
            ctx = await self._synthesize_hook(qc, scrape_data, news_items, icp)
            api_credits["gemini"] += 1
            result[domain] = ctx

            # Cache
            try:
                await cache_set(
                    _CACHE_METHOD,
                    domain,
                    ctx.model_dump(mode="json"),
                    _CACHE_TTL_DAYS,
                    self.settings,
                )
            except Exception as exc:  # noqa: BLE001
                self.log.warning("personalizer: cache write failed for %s: %s", domain, exc)

        duration = time.perf_counter() - start_ts
        self.log.info(
            "Personalizer[%s]: companies=%d sg=%d nd=%d gemini=%d (%.1fs)",
            segment,
            len(result),
            api_credits["scrapegraph"],
            api_credits["newsdata"],
            api_credits["gemini"],
            duration,
        )

        return result

    # =========================================================================
    # Private: Gemini Flash synthesis
    # =========================================================================

    async def _synthesize_hook(
        self,
        qualified: QualifiedCandidate,
        scrape_data: dict,
        news_items: list,
        icp: IcpStrategy,
    ) -> PersonalizationContext:
        candidate = qualified.candidate  # type: ignore[attr-defined]
        domain: str = getattr(candidate, "domain", "") or ""
        name: str = getattr(candidate, "name", domain) or domain
        description: str = getattr(candidate, "description", "") or ""
        funding_stage: str = getattr(candidate, "funding_stage", "") or ""
        funding_amount = getattr(candidate, "funding_amount_usd", None)
        funding_date = getattr(candidate, "funding_date", None)
        industry: str = getattr(candidate, "industry", "") or ""

        announcements = scrape_data.get("announcements") or []
        if isinstance(announcements, list):
            announcements_text = "; ".join(
                str(a.get("summary", a) if isinstance(a, dict) else a)
                for a in announcements[:5]
            ) or "none found"
        else:
            announcements_text = str(announcements) or "none found"

        recent_news_text = "; ".join(
            f"{item.title} ({getattr(item, 'published_at', '')})"
            for item in news_items[:3]
        ) or "none found"

        funding_line = "N/A"
        if funding_stage or funding_amount or funding_date:
            parts = [funding_stage or ""]
            if funding_amount:
                parts.append(f"${funding_amount:,.0f}")
            if funding_date:
                parts.append(f"on {funding_date}")
            funding_line = " ".join(p for p in parts if p)

        prompt = f"""You are building a single-sentence personalization hook for cold outreach.

About the company:
- Name: {name}
- Domain: {domain}
- Description: {description or "N/A"}
- Funding: {funding_line}
- Industry: {industry or "N/A"}

Recent announcements from their site:
{announcements_text}

Recent news mentions:
{recent_news_text}

Our segment context:
- We offer: {icp.value_prop_one_liner}
- Pain we solve: {icp.outreach_angle.pain_hypothesis}

Output strict JSON:
{{
  "company_one_liner": "what they do, in your own words, 1 sentence under 20 words",
  "recent_milestone": "the most specific recent event we can reference, OR null if nothing recent",
  "pain_hypothesis_specific": "how the generic ICP pain hypothesis applies to THIS company, 1 sentence",
  "why_now_hook": "the single opening line we would lead the email with — must reference the recent milestone if any, must be specific not generic, max 25 words",
  "personalization_quality": "high | medium | low — your own judgment of whether this hook is good enough to send"
}}

Rules:
- NEVER say "I came across..." or "I noticed..."
- NEVER use the words "synergies", "leverage", "circle back"
- "why_now_hook" must mention a specific detail (number, date, role, product feature) — generic "saw you are in AI" is unacceptable
- If recent_milestone is null AND no specific detail exists, set personalization_quality="low"
"""

        ctx = await self.gemini.generate_json(
            prompt, _PersonalizationContextRaw, temperature=0.4
        )

        if ctx is None:
            # Fallback: deterministic low-quality hook
            self.log.warning("personalizer: Gemini failed for %s, using fallback", domain)
            return PersonalizationContext(
                domain=domain,
                company_one_liner=description[:100] if description else f"{name} operates in {industry or 'the tech sector'}.",
                recent_milestone=None,
                pain_hypothesis_specific=icp.outreach_angle.pain_hypothesis,
                why_now_hook=f"Saw {name} recently {'raised ' + funding_line if funding_stage else 'is growing'} — worth a quick conversation.",
                personalization_quality="low",
                built_at=utcnow(),
            )

        return PersonalizationContext(
            domain=domain,
            company_one_liner=ctx.company_one_liner,
            recent_milestone=ctx.recent_milestone,
            pain_hypothesis_specific=ctx.pain_hypothesis_specific,
            why_now_hook=ctx.why_now_hook,
            personalization_quality=ctx.personalization_quality,
            built_at=utcnow(),
        )


# Internal schema for Gemini output (no domain/built_at — we add those)
from pydantic import BaseModel
from typing import Literal


class _PersonalizationContextRaw(BaseModel):
    company_one_liner: str
    recent_milestone: Optional[str] = None
    pain_hypothesis_specific: str
    why_now_hook: str
    personalization_quality: Literal["high", "medium", "low"]

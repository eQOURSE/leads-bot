"""Company Hunter agent — Phase 3.

Discovers funded EdTech / AI startups by running three data-source hunts in
parallel (RSS feeds, SerpAPI, NewsData), merging and deduplicating the results,
applying ICP filters, and optionally enriching the top candidates with
firmographics from TheCompaniesAPI.

Budget guardrails:
  - SerpAPI:      exactly 1 call per hunt
  - NewsData:     at most 2 calls per hunt
  - CompaniesAPI: at most enrichment_top_n calls per hunt (default 5)
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
import time
from datetime import date, timedelta
from typing import TYPE_CHECKING, Optional

from agents._gemini_wrapper import GeminiAgent
from agents._models import ExtractedFromSearch, HuntResult
from config.logging_config import setup_logging
from config.settings import Settings
from sources._utils import normalize_domain, utcnow
from sources.models import CompanyCandidate, NewsItem, SearchResult

if TYPE_CHECKING:
    from agents._models import IcpStrategy
    from agents.icp_strategist import IcpStrategist
    from sinks.sqlite_store import LeadStore
    from sources.companies_api_client import CompaniesAPIClient
    from sources.newsdata_client import NewsDataClient
    from sources.rss_feeds import RSSFundingMonitor
    from sources.serpapi_client import SerpAPIClient

# Matches bare IPs like 192.168.1.1 or simple single-label names like "localhost"
_IP_RE = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")


class CompanyHunter:
    """Parallel multi-source funded company discoverer."""

    def __init__(
        self,
        settings: Settings,
        icp_strategist: "IcpStrategist",
        rss_client: "RSSFundingMonitor",
        serpapi_client: "SerpAPIClient",
        newsdata_client: "NewsDataClient",
        companies_api_client: "CompaniesAPIClient",
        lead_store: "LeadStore",
        crunchbase_client=None,
        wellfound_client=None,
    ) -> None:
        self.settings = settings
        self.icp_strategist = icp_strategist
        self.rss_client = rss_client
        self.serpapi_client = serpapi_client
        self.newsdata_client = newsdata_client
        self.companies_api_client = companies_api_client
        self.crunchbase_client = crunchbase_client
        self.wellfound_client = wellfound_client
        self.lead_store = lead_store
        self.log = setup_logging("agent.company_hunter")
        self._gemini = GeminiAgent(
            settings.GEMINI_MODEL_RSS_PARSER, settings
        )
        # Phase 11 — last-run measurement (read by the runner/CLI).
        self.last_metrics: dict = {}

    # =========================================================================
    # Public API
    # =========================================================================

    async def hunt(
        self,
        segment: str,
        target_count: int = 50,
        enrichment_top_n: int = 5,
        skip_seen_within_days: int = 30,
        bypass_dedupe: bool = False,
    ) -> HuntResult:
        started_at = utcnow()
        start_ts = time.perf_counter()
        errors: list[str] = []
        api_credits: dict[str, int] = {}

        # 1 — load ICP
        icp = self.icp_strategist.load_strategy(segment)

        # 2 — open run record
        run_id = await self.lead_store.create_run(segment)

        # Reset per-run resilience/resolution metrics on shared clients.
        try:
            self.rss_client.reset_resolution_metrics()
        except Exception:  # noqa: BLE001
            pass

        # 3 — parallel sub-hunts (RSS, SerpAPI, NewsData + optional Crunchbase, Wellfound)
        tasks = [
            self._safe_hunt("rss", self._hunt_via_rss(icp), errors),
            self._safe_hunt("serpapi", self._hunt_via_serpapi(icp), errors),
            self._safe_hunt("newsdata", self._hunt_via_newsdata(icp), errors),
        ]
        use_crunchbase = (
            self.crunchbase_client is not None
            and getattr(self.settings, "ENABLE_CRUNCHBASE_DISCOVERY", True)
        )
        use_wellfound = (
            self.wellfound_client is not None
            and getattr(self.settings, "ENABLE_WELLFOUND_DISCOVERY", True)
        )
        if use_crunchbase:
            tasks.append(self._safe_hunt("crunchbase", self._hunt_via_crunchbase(icp), errors))
        if use_wellfound:
            tasks.append(self._safe_hunt("wellfound", self._hunt_via_wellfound(icp), errors))

        results = await asyncio.gather(*tasks)

        rss_task, serp_task, news_task = results[0], results[1], results[2]
        idx = 3
        cb_task: list = []
        wf_task: list = []
        if use_crunchbase:
            cb_task = results[idx]; idx += 1
        if use_wellfound:
            wf_task = results[idx]; idx += 1

        source_counts = {
            "rss": len(rss_task),
            "serpapi": len(serp_task),
            "newsdata": len(news_task),
            "crunchbase": len(cb_task),
            "wellfound": len(wf_task),
        }
        api_credits["serpapi"] = 1
        api_credits["newsdata"] = 2
        if use_crunchbase:
            api_credits["crunchbase"] = 1
        if use_wellfound:
            api_credits["wellfound"] = 1

        # 4 — merge (all sources)
        merged = self._merge_candidates(rss_task, serp_task, news_task, cb_task, wf_task)
        hunted_raw = sum(source_counts.values())

        # 5 — ICP filter
        filtered = self._apply_icp_filters(merged, icp)

        # 6 — dedupe
        after_dedupe = await self._dedupe_against_seen(
            filtered, skip_seen_within_days, bypass_dedupe
        )

        # 7 — sort by confidence desc
        after_dedupe.sort(key=lambda c: c.confidence, reverse=True)

        # 8 — enrich top N
        enriched_candidates = await self._enrich_with_firmographics(
            after_dedupe[:enrichment_top_n], enrichment_top_n
        )
        # Merge enriched back into the full list
        enriched_domains = {normalize_domain(c.domain) for c in enriched_candidates}
        final = [
            next(
                (e for e in enriched_candidates
                 if normalize_domain(e.domain) == normalize_domain(c.domain)),
                c,
            )
            for c in after_dedupe
        ]
        api_credits["companies_api"] = len(enriched_candidates)

        # Count Gemini calls from api_usage (best-effort)
        from sources._utils import _month_call_count_sync
        gemini_key = f"gemini_{self.settings.GEMINI_MODEL_RSS_PARSER}"
        api_credits["gemini"] = 0  # placeholder; actual tracked via usage table

        self.log.info(
            "Hunter[%s] RSS=%d SerpAPI=%d NewsData=%d → merged=%d "
            "→ filtered=%d → after_dedupe=%d → enriched_top=%d",
            segment,
            source_counts["rss"],
            source_counts["serpapi"],
            source_counts["newsdata"],
            len(merged),
            len(filtered),
            len(after_dedupe),
            len(enriched_candidates),
        )

        completed_at = utcnow()
        duration = time.perf_counter() - start_ts

        # Phase 11 — capture per-run measurement for the runner/CLI.
        try:
            resolution_rate = self.rss_client.article_link_resolution_rate
            resolution_rate = float(resolution_rate)
        except (TypeError, ValueError, AttributeError):
            resolution_rate = 0.0
        if 0.0 < resolution_rate < 0.30:
            self.log.warning(
                "Hunter[%s] article_link_resolution_rate=%.2f < 0.30 — extraction weak",
                segment, resolution_rate,
            )
        real_domains = sum(
            1 for c in final if c.domain and not c.domain.endswith(".unknown")
        )
        self.last_metrics = {
            "source_contributions": dict(source_counts),
            "hunted_raw": hunted_raw,
            "after_domain_resolution": real_domains,
            "after_dedupe": len(after_dedupe),
            "article_link_resolution_rate": round(resolution_rate, 3),
            "apify_discovery_spend_estimate_usd": (
                (1.0 if use_crunchbase else 0.0) + (1.0 if use_wellfound else 0.0)
            ),
        }

        # 9 — update run record
        await self.lead_store.update_run(
            run_id,
            completed_at=completed_at,
            status="completed",
            candidates_found=len(final),
            enriched_count=len(enriched_candidates),
            api_credits_used=api_credits,
            error_log="; ".join(errors) if errors else None,
        )

        # 10 — mark domains seen
        await self.lead_store.mark_domains_seen(
            [normalize_domain(c.domain) for c in final if c.domain]
        )

        return HuntResult(
            segment=segment,
            run_id=run_id,
            candidates=final,
            source_counts=source_counts,
            merged_count=len(merged),
            after_filter=len(filtered),
            after_dedupe=len(after_dedupe),
            enriched_count=len(enriched_candidates),
            api_credits_used=api_credits,
            errors=errors,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
        )

    # =========================================================================
    # Sub-hunts
    # =========================================================================

    async def _safe_hunt(
        self,
        source: str,
        coro,
        errors: list[str],
    ) -> list[CompanyCandidate]:
        """Run a sub-hunt coroutine; catch any exception and log it."""
        try:
            return await coro
        except Exception as exc:  # noqa: BLE001
            msg = f"{source} sub-hunt failed: {type(exc).__name__}: {exc}"
            self.log.warning(msg)
            errors.append(msg)
            return []

    async def _hunt_via_rss(self, icp: "IcpStrategy") -> list[CompanyCandidate]:
        items: list[NewsItem] = await self.rss_client.fetch_recent_funding(
            since_days=icp.target_company_profile.funding_recency_days,
            keywords=["raises", "raised", "funding", "seed", "series"],
        )

        icp_keywords = [k.lower() for k in icp.target_industries.industry_keywords]
        candidates: list[CompanyCandidate] = []
        for item in items:
            # Keyword filter — at least one ICP keyword must appear in headline+snippet
            haystack = f"{item.title} {item.snippet}".lower()
            if not any(kw in haystack for kw in icp_keywords):
                continue

            company = await self.rss_client.extract_company_from_headline(item)
            if company is None:
                continue

            # Override with consistent raw_source and confidence
            candidates.append(
                company.model_copy(
                    update={"raw_source": "rss", "confidence": 0.6}
                )
            )
        return candidates

    async def _hunt_via_serpapi(self, icp: "IcpStrategy") -> list[CompanyCandidate]:
        keywords = icp.target_industries.industry_keywords
        primary_kw = keywords[0] if keywords else "edtech"
        recency_days = icp.target_company_profile.funding_recency_days
        cutoff_date = (date.today() - timedelta(days=recency_days)).isoformat()

        query = (
            f'"raised" ("seed" OR "series A" OR "series B") '
            f'"{primary_kw}" '
            f'site:techcrunch.com OR site:strictlyvc.com '
            f"after:{cutoff_date}"
        )

        results: list[SearchResult] = await self.serpapi_client.search(query, num=10)
        if not results:
            return []

        prompts = [
            (
                "Extract company funding data from this search result. "
                "Return strict JSON with keys: company_name (string or null), "
                "domain (string or null), funding_amount_usd (float or null), "
                "funding_stage (string or null), announcement_date (YYYY-MM-DD or null), "
                "description (string or null). If no company or funding found, "
                'set company_name to null.\n\n'
                f"Title: {r.title}\nURL: {r.url}\nSnippet: {r.snippet}"
            )
            for r in results
        ]

        extracted: list[Optional[ExtractedFromSearch]] = (
            await self._gemini.batch_generate_json(prompts, ExtractedFromSearch)
        )

        candidates: list[CompanyCandidate] = []
        for ex in extracted:
            if ex is None or not ex.company_name:
                continue
            domain = ex.domain or self._slug_domain(ex.company_name)
            try:
                candidates.append(
                    CompanyCandidate(
                        domain=domain,
                        name=ex.company_name,
                        description=ex.description,
                        funding_amount_usd=ex.funding_amount_usd,
                        funding_stage=ex.funding_stage,
                        funding_date=ex.announcement_date,
                        raw_source="serpapi",
                        confidence=0.7,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                self.log.warning("serpapi candidate skipped: %s", exc)
        return candidates

    async def _hunt_via_newsdata(self, icp: "IcpStrategy") -> list[CompanyCandidate]:
        icp_keywords = icp.target_industries.industry_keywords
        search_keywords = ["raised", "funding"] + list(icp_keywords[:3])

        # NewsData free tier is 48h window; cap at 2 calls
        items: list[NewsItem] = await self.newsdata_client.search_funding_news(
            keywords=search_keywords,
            days_back=2,
            countries=[],   # global search for better coverage on free tier
            categories=[],
        )

        if not items:
            return []

        prompts = [
            (
                "Extract company funding data from this news headline. "
                "Return strict JSON with keys: company_name (string or null), "
                "domain (string or null), funding_amount_usd (float or null), "
                "funding_stage (string or null), announcement_date (YYYY-MM-DD or null), "
                "description (string or null). If not a funding announcement, "
                'set company_name to null.\n\n'
                f"Title: {item.title}\nSnippet: {item.snippet}"
            )
            for item in items
        ]

        extracted: list[Optional[ExtractedFromSearch]] = (
            await self._gemini.batch_generate_json(prompts, ExtractedFromSearch)
        )

        candidates: list[CompanyCandidate] = []
        for ex in extracted:
            if ex is None or not ex.company_name:
                continue
            domain = ex.domain or self._slug_domain(ex.company_name)
            try:
                candidates.append(
                    CompanyCandidate(
                        domain=domain,
                        name=ex.company_name,
                        description=ex.description,
                        funding_amount_usd=ex.funding_amount_usd,
                        funding_stage=ex.funding_stage,
                        funding_date=ex.announcement_date,
                        raw_source="newsdata",
                        confidence=0.65,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                self.log.warning("newsdata candidate skipped: %s", exc)
        return candidates

    # =========================================================================
    # Phase 11 — Crunchbase & Wellfound discovery
    # =========================================================================

    async def _hunt_via_crunchbase(self, icp: "IcpStrategy") -> list[CompanyCandidate]:
        if self.crunchbase_client is None:
            return []
        prof = icp.target_company_profile
        return await self.crunchbase_client.search_recent_funding(
            industries=icp.target_industries.naics_codes,
            keywords=icp.target_industries.industry_keywords,
            funding_stages=[s.lower().replace(" ", "_") for s in prof.funding_stages],
            days_back=prof.funding_recency_days,
            country=(prof.geographies.countries[0] if prof.geographies.countries else "United States"),
            limit=50,
        )

    async def _hunt_via_wellfound(self, icp: "IcpStrategy") -> list[CompanyCandidate]:
        if self.wellfound_client is None:
            return []
        prof = icp.target_company_profile
        return await self.wellfound_client.search_recent_startups(
            industries=icp.target_industries.naics_codes,
            keywords=icp.target_industries.industry_keywords,
            founded_after_year=prof.founded_after_year,
            funding_min=500_000,
            limit=50,
        )

    # =========================================================================
    # Merge
    # =========================================================================

    def _merge_candidates(
        self,
        *source_lists: list[CompanyCandidate],
    ) -> list[CompanyCandidate]:
        merged: dict[str, CompanyCandidate] = {}

        all_candidates: list[CompanyCandidate] = []
        for lst in source_lists:
            all_candidates.extend(lst or [])

        for candidate in all_candidates:
            key = normalize_domain(candidate.domain)
            if not key:
                continue

            if key not in merged:
                merged[key] = candidate
            else:
                existing = merged[key]
                # Combine sources
                existing_sources = set(existing.raw_source.split("+"))
                new_sources = set(candidate.raw_source.split("+"))
                combined_sources = existing_sources | new_sources

                # Prefer non-null fields from the incoming candidate
                updates: dict = {
                    "raw_source": "+".join(sorted(combined_sources)),
                }
                for field in (
                    "name", "description", "industry", "size_range",
                    "hq_country", "hq_region", "website",
                    "funding_amount_usd", "funding_stage", "funding_date",
                ):
                    existing_val = getattr(existing, field, None)
                    new_val = getattr(candidate, field, None)
                    if existing_val is None and new_val is not None:
                        updates[field] = new_val

                # Multi-source confidence boost (+0.2, capped at 1.0)
                if len(combined_sources) > len(existing_sources):
                    updates["confidence"] = min(1.0, existing.confidence + 0.2)

                merged[key] = existing.model_copy(update=updates)

        return list(merged.values())

    # =========================================================================
    # ICP filter
    # =========================================================================

    def _apply_icp_filters(
        self,
        candidates: list[CompanyCandidate],
        icp: "IcpStrategy",
    ) -> list[CompanyCandidate]:
        recency_days = icp.target_company_profile.funding_recency_days
        cutoff = date.today() - timedelta(days=recency_days)
        allowed_countries = [c.lower() for c in icp.target_company_profile.geographies.countries]
        neg_keywords = [sig.lower() for sig in icp.negative_signals]

        results: list[CompanyCandidate] = []
        for c in candidates:
            # Domain validity check
            if not self._is_valid_domain(c.domain):
                continue

            # Funding recency filter
            if c.funding_date is not None:
                if c.funding_date < cutoff:
                    continue
            else:
                # No funding date: keep with confidence penalty
                c = c.model_copy(
                    update={"confidence": max(0.0, c.confidence - 0.1)}
                )

            # Country filter (only when ICP specifies countries)
            if allowed_countries and c.hq_country:
                if c.hq_country.lower() not in allowed_countries and \
                        c.hq_country.lower()[:2] not in allowed_countries:
                    continue

            # Negative signal pre-screen on description
            if c.description and self._matches_signals(c.description, neg_keywords):
                continue

            results.append(c)
        return results

    # =========================================================================
    # Dedupe
    # =========================================================================

    async def _dedupe_against_seen(
        self,
        candidates: list[CompanyCandidate],
        skip_seen_within_days: int,
        bypass: bool,
    ) -> list[CompanyCandidate]:
        if bypass:
            return candidates

        seen = await self.lead_store.get_seen_domains_within(skip_seen_within_days)
        return [
            c for c in candidates
            if normalize_domain(c.domain) not in seen
        ]

    # =========================================================================
    # Enrichment
    # =========================================================================

    async def _enrich_with_firmographics(
        self,
        top_candidates: list[CompanyCandidate],
        cap: int,
    ) -> list[CompanyCandidate]:
        enriched: list[CompanyCandidate] = []
        calls = 0
        for c in top_candidates:
            if calls >= cap:
                break
            # Only enrich when key firmographic fields are missing
            if c.size_range is not None and c.hq_country is not None:
                enriched.append(c)
                continue
            try:
                result = await self.companies_api_client.enrich_by_domain(c.domain)
                calls += 1
                if result is not None:
                    # Merge non-null fields from enrichment back into candidate
                    updates: dict = {}
                    for field in (
                        "description", "industry", "size_range", "revenue_range",
                        "hq_country", "hq_region", "website", "naics_codes",
                    ):
                        enriched_val = getattr(result, field, None)
                        if enriched_val and not getattr(c, field, None):
                            updates[field] = enriched_val
                    enriched.append(c.model_copy(update=updates))
                else:
                    enriched.append(c)
            except Exception as exc:  # noqa: BLE001
                self.log.warning("enrichment failed for %s: %s", c.domain, exc)
                enriched.append(c)
        # Append any remaining that we didn't process (beyond cap)
        for c in top_candidates[len(enriched):]:
            enriched.append(c)
        return enriched

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _slug_domain(company_name: str) -> str:
        """Derive a placeholder domain from a company name."""
        slug = "".join(ch for ch in company_name.lower() if ch.isalnum())
        return f"{slug}.unknown" if slug else "unknown.unknown"

    @staticmethod
    def _is_valid_domain(domain: str) -> bool:
        """Return False for IPs, single-label domains, or empty strings."""
        if not domain or "." not in domain:
            return False
        if _IP_RE.match(domain):
            return False
        try:
            ipaddress.ip_address(domain)
            return False
        except ValueError:
            pass
        # Placeholder domains we generate end in .unknown — keep them for
        # testing but flag them as lower confidence (handled upstream)
        return True

    @staticmethod
    def _matches_signals(text: str, signal_keywords: list[str]) -> bool:
        """Return True if any signal keyword appears in the text."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in signal_keywords)

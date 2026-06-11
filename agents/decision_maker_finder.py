"""Decision-Maker Finder — Phase 5.

For each qualified company in a QualifiedResult, finds 1-2 decision-makers
using a three-tier source cascade:

  Tier 1 (cheapest, always tried first): ScrapeGraph team-page extraction
  Tier 2 (moderate cost, tier_1 leads only): Apify LinkedIn company profile
  Tier 3 (expensive, tier_1 leads only, 1 per run): Explorium / Vibe Prospecting

Budget guardrails per run:
  - ScrapeGraph: scrapegraph_cap_per_segment (default 5)
  - Apify LinkedIn: apify_cap_per_run (default 3).  Each attempt = 2 Apify calls.
  - Explorium: explorium_cap_per_run (default 1)

Domains on the news-source blacklist or ending in .unknown → immediately
flagged as needs_manual_lookup; no API calls made.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

from agents._constants import (
    NEWS_SOURCE_DOMAINS,
    domain_is_news_source,
    seniority_score,
)
from agents._models import (
    DecisionMaker,
    EnhancedQualifiedResult,
    QualifiedCandidate,
    QualifiedCandidateWithPeople,
    QualifiedResult,
)
from config.logging_config import setup_logging
from config.settings import Settings
from sources._utils import utcnow


def _classify_error(exc: Exception) -> str:
    """Map an exception raised by a source client to a specific reason code.

    Inspects the exception text for HTTP status codes / known markers so the
    DM-finder can record *why* a source failed rather than a generic
    "no_results". Used by Phase 13 lookup_attempts enrichment.
    """
    msg = str(exc).lower()
    # HTTP status codes (clients raise httpx errors whose str() contains them)
    if "401" in msg or "403" in msg or "unauthorized" in msg or "forbidden" in msg:
        return "auth_failed"
    if "404" in msg or "not found" in msg:
        return "url_not_found"
    if "429" in msg or "rate limit" in msg or "too many requests" in msg:
        return "rate_limited"
    if "timeout" in msg or "timed out" in msg or isinstance(exc, TimeoutError):
        return "timeout"
    return "error"


if TYPE_CHECKING:
    from agents.icp_strategist import IcpStrategist
    from sinks.sqlite_store import LeadStore
    from sources.apify_client import ApifyMultiKeyClient
    from sources.scrapegraph_client import ScrapeGraphClient
    from sources.vibe_prospecting import VibeProspectingClient


class DecisionMakerFinder:
    """Finds decision-makers for qualified companies using a source cascade."""

    def __init__(
        self,
        settings: Settings,
        icp_strategist: "IcpStrategist",
        scrapegraph_client: "ScrapeGraphClient",
        apify_client: "ApifyMultiKeyClient",
        vibe_prospecting_client: "VibeProspectingClient",
        lead_store: "LeadStore",
    ) -> None:
        self.settings = settings
        self.icp_strategist = icp_strategist
        self.scrapegraph_client = scrapegraph_client
        self.apify_client = apify_client
        self.vibe_prospecting_client = vibe_prospecting_client
        self.lead_store = lead_store
        self.log = setup_logging("agent.decision_maker_finder")

    # =========================================================================
    # Public API
    # =========================================================================

    async def find_for_qualified(
        self,
        qualified_result: QualifiedResult,
        max_per_company: int = 2,
        scrapegraph_cap_per_segment: int = 5,
        apify_cap_per_run: int = 3,
        explorium_cap_per_run: int = 1,
    ) -> EnhancedQualifiedResult:
        started_at = utcnow()
        start_ts = time.perf_counter()
        segment = qualified_result.segment
        run_id = qualified_result.run_id

        # Load ICP for this segment
        icp = self.icp_strategist.load_strategy(segment)

        # Counters
        sg_calls_remaining = scrapegraph_cap_per_segment
        # Apify: each linkedin attempt = 2 calls, so track in units of 2
        apify_calls_remaining = apify_cap_per_run
        explorium_calls_remaining = explorium_cap_per_run

        sg_hits = apify_hits = explorium_hits = manual_lookup = 0

        # Process tier_1 first, then tier_2
        all_candidates = sorted(
            qualified_result.qualified,
            key=lambda q: (0 if q.tier == "tier_1" else 1),
        )

        candidates_with_people: list[QualifiedCandidateWithPeople] = []
        needs_manual_lookup: list[QualifiedCandidate] = []

        api_credits_used: dict[str, int] = {
            "scrapegraph": 0,
            "apify": 0,
            "explorium": 0,
        }

        for qualified in all_candidates:
            candidate = qualified.candidate  # type: ignore[attr-defined]
            domain: str = getattr(candidate, "domain", "") or ""
            attempts: dict[str, str] = {}

            # --- Blacklist / unknown domain check ---
            if domain.endswith(".unknown") or domain_is_news_source(domain):
                self.log.info(
                    "DecisionMaker[%s]: %s → needs_manual_lookup (domain: %s)",
                    run_id,
                    getattr(candidate, "name", domain),
                    domain,
                )
                manual_lookup += 1
                needs_manual_lookup.append(qualified)
                candidates_with_people.append(
                    QualifiedCandidateWithPeople(
                        qualified=qualified,
                        decision_makers=[],
                        lookup_status="needs_manual_lookup",
                        lookup_attempts={"reason": "blacklisted_or_unknown_domain"},
                    )
                )
                continue

            decision_makers: list[DecisionMaker] = []
            # Phase 13: track how many raw DMs each source produced (pre-ICP-filter)
            raw_source_counts: dict[str, int] = {}

            # -----------------------------------------------------------------
            # Tier A: ScrapeGraph
            # -----------------------------------------------------------------
            if sg_calls_remaining > 0:
                sg_dms, sg_reason = await self._find_via_scrapegraph(qualified, icp)
                if sg_dms:
                    decision_makers.extend(sg_dms)
                    sg_hits += len(sg_dms)
                    api_credits_used["scrapegraph"] += 1
                    raw_source_counts["scrapegraph"] = len(sg_dms)
                    attempts["scrapegraph"] = f"found_{len(sg_dms)}"
                else:
                    attempts["scrapegraph"] = sg_reason
                sg_calls_remaining -= 1
                if sg_calls_remaining == 0:
                    self.log.info(
                        "DecisionMaker[%s]: ScrapeGraph cap reached (%d)",
                        run_id,
                        scrapegraph_cap_per_segment,
                    )
            else:
                attempts["scrapegraph"] = "not_attempted"

            # -----------------------------------------------------------------
            # Tier B: Apify LinkedIn (tier_1 only, fallback when SG returned nothing)
            # -----------------------------------------------------------------
            if (
                not decision_makers
                and qualified.tier == "tier_1"
                and apify_calls_remaining > 0
            ):
                apify_dms, apify_reason = await self._find_via_apify_linkedin(qualified, icp)
                if apify_dms:
                    decision_makers.extend(apify_dms)
                    apify_hits += len(apify_dms)
                    api_credits_used["apify"] += 2
                    raw_source_counts["apify"] = len(apify_dms)
                    attempts["apify"] = f"found_{len(apify_dms)}"
                else:
                    attempts["apify"] = apify_reason
                apify_calls_remaining -= 1
                if apify_calls_remaining == 0:
                    self.log.info(
                        "DecisionMaker[%s]: Apify cap reached (%d)",
                        run_id,
                        apify_cap_per_run,
                    )
            elif not decision_makers and qualified.tier == "tier_2":
                attempts["apify"] = "not_attempted"
            elif not decision_makers and apify_calls_remaining == 0:
                attempts["apify"] = "not_attempted"

            # -----------------------------------------------------------------
            # Tier C: Explorium (tier_1 only, last resort)
            # -----------------------------------------------------------------
            if (
                not decision_makers
                and qualified.tier == "tier_1"
                and explorium_calls_remaining > 0
            ):
                explorium_dms, explorium_reason = await self._find_via_explorium(qualified, icp)
                if explorium_dms:
                    decision_makers.extend(explorium_dms)
                    explorium_hits += len(explorium_dms)
                    api_credits_used["explorium"] += 1
                    raw_source_counts["explorium"] = len(explorium_dms)
                    attempts["explorium"] = f"found_{len(explorium_dms)}"
                else:
                    attempts["explorium"] = explorium_reason
                explorium_calls_remaining -= 1
                if explorium_calls_remaining == 0:
                    self.log.info(
                        "DecisionMaker[%s]: Explorium cap reached (%d)",
                        run_id,
                        explorium_cap_per_run,
                    )
            elif not decision_makers:
                attempts.setdefault("explorium", "not_attempted")

            # -----------------------------------------------------------------
            # Filter, rank, cap
            # -----------------------------------------------------------------
            filtered = self._filter_by_icp(decision_makers, icp)
            final_dms = self._rank_and_cap(filtered, max_per_company)

            # Phase 13: if a source produced raw DMs but the ICP filter removed
            # all of them, relabel that source's attempt as filtered_out_by_icp
            # so we can see exactly where the candidate was lost.
            if not final_dms and raw_source_counts:
                for src, cnt in raw_source_counts.items():
                    if cnt > 0 and attempts.get(src, "").startswith("found_"):
                        attempts[src] = "filtered_out_by_icp"

            if final_dms:
                status = "found"
            elif domain.endswith(".unknown") or domain_is_news_source(domain):
                status = "needs_manual_lookup"
            else:
                status = "no_decision_maker"

            # Phase 13: log exactly why a candidate yielded no decision-maker
            if status == "no_decision_maker":
                self.log.info(
                    "DM-finder no_decision_maker for %s (%s): attempts=%s",
                    getattr(candidate, "name", domain),
                    domain,
                    attempts,
                )

            candidates_with_people.append(
                QualifiedCandidateWithPeople(
                    qualified=qualified,
                    decision_makers=final_dms,
                    lookup_status=status,
                    lookup_attempts=attempts,
                )
            )

        # Stats
        tier_1_count = sum(
            1 for q in qualified_result.qualified if q.tier == "tier_1"
        )
        tier_2_count = sum(
            1 for q in qualified_result.qualified if q.tier == "tier_2"
        )
        stats = {
            "tier_1": tier_1_count,
            "tier_2": tier_2_count,
            "sg_hits": sg_hits,
            "apify_hits": apify_hits,
            "explorium_hits": explorium_hits,
            "manual_lookup": manual_lookup,
            "total_dms_found": sg_hits + apify_hits + explorium_hits,
        }

        self.log.info(
            "DecisionMaker[%s]: tier_1=%d tier_2=%d → sg_hits=%d apify_hits=%d "
            "explorium_hits=%d manual_lookup=%d",
            run_id,
            tier_1_count,
            tier_2_count,
            sg_hits,
            apify_hits,
            explorium_hits,
            manual_lookup,
        )

        # Update run record
        try:
            await self.lead_store.update_run(
                run_id,
                api_credits_used=api_credits_used,
                status="decision_makers_found",
            )
        except Exception as exc:  # noqa: BLE001
            self.log.warning("Failed to update run record: %s", exc)

        completed_at = utcnow()
        return EnhancedQualifiedResult(
            segment=segment,
            run_id=run_id,
            candidates_with_people=candidates_with_people,
            needs_manual_lookup=needs_manual_lookup,
            stats=stats,
            api_credits_used=api_credits_used,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=time.perf_counter() - start_ts,
        )

    # =========================================================================
    # Sub-finder methods (private)
    # =========================================================================

    async def _find_via_scrapegraph(
        self,
        qualified: QualifiedCandidate,
        icp,
    ) -> tuple[list[DecisionMaker], str]:
        """Extract decision-makers via ScrapeGraph team-page extraction.

        Returns (decision_makers, reason_code). reason_code is one of
        "found", "fetched_but_empty", or an error code from _classify_error.
        """
        candidate = qualified.candidate  # type: ignore[attr-defined]
        domain: str = getattr(candidate, "domain", "") or ""
        website: Optional[str] = getattr(candidate, "website", None)

        # Build base URL
        if website:
            base_url = website.rstrip("/")
        elif domain:
            base_url = f"https://{domain}"
        else:
            return [], "url_not_found"

        try:
            prospects = await self.scrapegraph_client.extract_team_page(base_url)
        except Exception as exc:  # noqa: BLE001
            reason = _classify_error(exc)
            self.log.warning(
                "scrapegraph extract_team_page failed for %s: %s (%s)",
                base_url, exc, reason,
            )
            return [], reason

        results: list[DecisionMaker] = []
        for p in prospects:
            score = seniority_score(p.title)
            results.append(
                DecisionMaker(
                    full_name=p.full_name,
                    title=p.title,
                    linkedin_url=p.linkedin_url,
                    source="scrapegraph",
                    seniority_score=score,
                )
            )
        if not results:
            return [], "fetched_but_empty"
        return results, "found"

    async def _find_via_apify_linkedin(
        self,
        qualified: QualifiedCandidate,
        icp,
    ) -> tuple[list[DecisionMaker], str]:
        """Find decision-makers via Apify LinkedIn company profile scrape.

        Step A: Google search for company LinkedIn URL.
        Step B: Apify LinkedIn company scraper to get employees.
        Counts as 2 Apify calls toward the cap.

        Returns (decision_makers, reason_code).
        """
        candidate = qualified.candidate  # type: ignore[attr-defined]
        name: str = getattr(candidate, "name", "") or ""
        domain: str = getattr(candidate, "domain", "") or ""

        if not name:
            return [], "url_not_found"

        # Step A: find LinkedIn company URL
        try:
            search_results = await self.apify_client.google_search(
                f'"{name}" site:linkedin.com/company', num_results=3
            )
        except Exception as exc:  # noqa: BLE001
            reason = _classify_error(exc)
            self.log.warning(
                "apify google_search for LinkedIn failed (%s): %s (%s)",
                name, exc, reason,
            )
            return [], reason

        linkedin_url: Optional[str] = None
        for r in search_results:
            if "linkedin.com/company/" in (r.url or ""):
                linkedin_url = r.url
                break

        if not linkedin_url:
            self.log.info("apify: no LinkedIn company URL found for %s", name)
            return [], "url_not_found"

        # Step B: scrape company employees
        try:
            company_data = await self.apify_client.linkedin_company(linkedin_url)
        except Exception as exc:  # noqa: BLE001
            reason = _classify_error(exc)
            self.log.warning(
                "apify linkedin_company failed for %s: %s (%s)",
                linkedin_url, exc, reason,
            )
            return [], reason

        # Extract employees list from the scraped data
        employees: list[dict] = []
        if isinstance(company_data, dict):
            employees = (
                company_data.get("employees")
                or company_data.get("members")
                or company_data.get("people")
                or []
            )

        if not employees:
            return [], "fetched_but_empty"

        # Filter by ICP target titles / levels
        target_titles_lower = [t.lower() for t in (icp.target_titles or [])]
        target_levels_lower = [lv.lower() for lv in (icp.target_levels or [])]

        results: list[DecisionMaker] = []
        for emp in employees:
            if not isinstance(emp, dict):
                continue
            full_name = emp.get("name") or emp.get("full_name") or ""
            title = emp.get("title") or emp.get("headline") or ""
            if not full_name or not title:
                continue

            title_lower = title.lower()
            # ICP filter — must match a target title or target level
            title_match = any(t in title_lower for t in target_titles_lower)
            level_match = any(lv in title_lower for lv in target_levels_lower)
            if not title_match and not level_match:
                continue

            score = seniority_score(title)
            results.append(
                DecisionMaker(
                    full_name=full_name,
                    title=title,
                    linkedin_url=emp.get("linkedin_url") or emp.get("url"),
                    source="apify_linkedin",
                    seniority_score=score,
                )
            )

        if not results:
            # employees came back but none matched ICP titles/levels
            return [], "filtered_out_by_icp"
        return results, "found"

    async def _find_via_explorium(
        self,
        qualified: QualifiedCandidate,
        icp,
    ) -> tuple[list[DecisionMaker], str]:
        """Find decision-makers via Explorium / Vibe Prospecting API.

        Returns (decision_makers, reason_code).
        """
        candidate = qualified.candidate  # type: ignore[attr-defined]
        domain: str = getattr(candidate, "domain", "") or ""

        if not domain:
            return [], "url_not_found"

        try:
            prospects = await self.vibe_prospecting_client.find_prospects(
                business_domains=[domain],
                job_titles=icp.target_titles[:10] if icp.target_titles else [],
                job_departments=icp.target_departments or [],
                job_levels=icp.target_levels or [],
                has_email=False,  # Phase 6 handles emails
            )
        except Exception as exc:  # noqa: BLE001
            reason = _classify_error(exc)
            self.log.warning(
                "explorium find_prospects failed for %s: %s (%s)",
                domain, exc, reason,
            )
            return [], reason

        results: list[DecisionMaker] = []
        for p in prospects:
            score = seniority_score(p.title)
            results.append(
                DecisionMaker(
                    full_name=p.full_name,
                    title=p.title,
                    linkedin_url=p.linkedin_url,
                    source="explorium",
                    seniority_score=score,
                )
            )
        if not results:
            return [], "fetched_but_empty"
        return results, "found"

    # =========================================================================
    # Filtering and ranking
    # =========================================================================

    def _filter_by_icp(
        self,
        decision_makers: list[DecisionMaker],
        icp,
    ) -> list[DecisionMaker]:
        """Filter decision-makers by ICP target titles or seniority threshold.

        Rules:
        - Keep if title contains any ICP target title (case-insensitive substring).
        - OR if seniority_score(title) >= 65 (Director level or above as backstop).
        - Drop if title contains "intern", "assistant", "associate"
          (unless preceded by "Senior").
        """
        target_titles_lower = [t.lower() for t in (getattr(icp, "target_titles", None) or [])]
        results: list[DecisionMaker] = []

        for dm in decision_makers:
            title_lower = dm.title.lower()

            # Drop junior roles unless prefixed with "Senior"
            is_junior = any(
                jr in title_lower
                for jr in ("intern", "assistant", "associate")
            )
            if is_junior and "senior" not in title_lower:
                continue

            # ICP title match
            title_match = any(t in title_lower for t in target_titles_lower)
            # Seniority backstop (Director level+)
            senior_enough = dm.seniority_score >= 65

            if title_match or senior_enough:
                results.append(dm)

        return results

    def _rank_and_cap(
        self,
        decision_makers: list[DecisionMaker],
        max_per_company: int,
    ) -> list[DecisionMaker]:
        """Rank by seniority desc; tiebreak prefers those with linkedin_url."""
        ranked = sorted(
            decision_makers,
            key=lambda dm: (
                -dm.seniority_score,
                0 if dm.linkedin_url else 1,
            ),
        )
        return ranked[:max_per_company]

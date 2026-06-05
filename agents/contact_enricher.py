"""Contact Enricher — Phase 6.

Takes an EnhancedQualifiedResult (decision-makers with empty email/phone)
and populates email addresses via a cascading strategy:

  1. Hunter domain_search → pattern + known emails (cached 30 days)
  2. Hunter email_finder (tier_1 only, direct lookup by name)
  3. Pattern-based candidate generation + SMTP verification
  4. AbstractAPI deliverability check (tier_1 only, boosts confidence)
  5. Common-prefix fallback for companies with no DMs (founder@, ceo@, …)
  6. Explorium phone enrichment for top tier_1 (optional, 1 call/run)

Budget per run (defaults):
  - hunter_domain_cap  = 3   (one call per unique domain, cached after)
  - hunter_finder_cap  = 2   (name-specific lookup, tier_1 only)
  - abstract_api_cap   = 5   (deliverability check, tier_1 final email only)
  - explorium_cap      = 1   (phone, top tier_1 candidate only)
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from agents._email_patterns import (
    _apply_pattern,
    _default_patterns,
    _name_matches_email,
    _split_name,
)
from agents._models import (
    DecisionMaker,
    DomainPattern,
    EmailResult,
    EnhancedQualifiedResult,
    EnrichedCandidate,
    EnrichedDecisionMaker,
    EnrichedResult,
    QualifiedCandidateWithPeople,
)
from config.logging_config import setup_logging
from config.settings import Settings
from sources._cache import cache_get, cache_set
from sources._utils import utcnow

if TYPE_CHECKING:
    from agents.icp_strategist import IcpStrategist
    from sinks.sqlite_store import LeadStore
    from sources.abstract_api_client import AbstractAPIClient
    from sources.hunter_client import HunterClient
    from sources.smtp_verifier import SMTPVerifier
    from sources.vibe_prospecting import VibeProspectingClient

_PATTERN_CACHE_METHOD = "domain_pattern"
_PATTERN_CACHE_TTL = 30   # days

_COMMON_PREFIXES = ["founder", "ceo", "hello", "info", "contact", "team"]


class ContactEnricher:
    """Enrich decision-makers with email addresses using a budget-conscious cascade."""

    def __init__(
        self,
        settings: Settings,
        hunter_client: "HunterClient",
        abstract_api_client: "AbstractAPIClient",
        smtp_verifier: "SMTPVerifier",
        vibe_prospecting_client: "VibeProspectingClient",
        lead_store: "LeadStore",
    ) -> None:
        self.settings = settings
        self.hunter = hunter_client
        self.abstract_api = abstract_api_client
        self.smtp = smtp_verifier
        self.vibe = vibe_prospecting_client
        self.lead_store = lead_store
        self.log = setup_logging("agent.contact_enricher")

    # =========================================================================
    # Public API
    # =========================================================================

    async def enrich(
        self,
        enhanced_result: EnhancedQualifiedResult,
        hunter_domain_cap: int = 3,
        hunter_finder_cap: int = 2,
        abstract_api_cap: int = 5,
        explorium_cap: int = 1,
    ) -> EnrichedResult:
        started_at = utcnow()
        start_ts = time.perf_counter()
        segment = enhanced_result.segment
        run_id = enhanced_result.run_id

        # Mutable counters
        self._hunter_domain_remaining = hunter_domain_cap
        self._hunter_finder_remaining = hunter_finder_cap
        self._abstract_api_remaining = abstract_api_cap
        self._explorium_remaining = explorium_cap

        # Stats
        stats: dict = {
            "dms_total": 0,
            "hunter_domain_calls": 0,
            "hunter_finder_calls": 0,
            "smtp_verifies": 0,
            "abstract_api_calls": 0,
            "explorium_calls": 0,
            "emails_found": 0,
            "by_source": {},
        }
        api_credits: dict[str, int] = {
            "hunter_domain": 0,
            "hunter_finder": 0,
            "abstract_api": 0,
            "explorium": 0,
            "smtp": 0,
        }

        enriched_candidates: list[EnrichedCandidate] = []

        # Process tier_1 first, then tier_2
        sorted_cwp = sorted(
            enhanced_result.candidates_with_people,
            key=lambda c: (0 if c.qualified.tier == "tier_1" else 1),
        )

        for cwp in sorted_cwp:
            candidate = cwp.qualified.candidate  # type: ignore[attr-defined]
            domain: str = getattr(candidate, "domain", "") or ""
            tier: str = cwp.qualified.tier

            # Skip blacklisted / unresolvable domains
            if cwp.lookup_status == "needs_manual_lookup":
                enriched_candidates.append(
                    EnrichedCandidate(
                        candidate_with_people=cwp,
                        enriched_dms=[],
                        enrichment_status="skipped",
                    )
                )
                continue

            # Get domain email pattern (Hunter or cache)
            domain_pattern = await self._get_domain_pattern(domain, tier, stats, api_credits)

            enriched_dms: list[EnrichedDecisionMaker] = []

            if cwp.decision_makers:
                for dm in cwp.decision_makers:
                    stats["dms_total"] += 1
                    email_result = await self._resolve_email_for_dm(
                        dm, domain, domain_pattern, tier, stats, api_credits
                    )
                    enriched_dms.append(
                        EnrichedDecisionMaker(
                            decision_maker=dm,
                            email_result=email_result,
                        )
                    )
                    if email_result.email:
                        stats["emails_found"] += 1
                        src = email_result.source
                        stats["by_source"][src] = stats["by_source"].get(src, 0) + 1

            # Determine enrichment_status
            if not cwp.decision_makers:
                enrichment_status = "no_emails"
                # Common-prefix fallback for tier_1 companies with no DMs
                company_email: Optional[EmailResult] = None
                if tier == "tier_1":
                    company_email = await self._try_common_prefixes(domain, stats, api_credits)
            else:
                found_count = sum(1 for edm in enriched_dms if edm.email_result.email)
                if found_count == len(enriched_dms):
                    enrichment_status = "full"
                elif found_count > 0:
                    enrichment_status = "partial"
                else:
                    enrichment_status = "no_emails"
                company_email = None

            enriched_candidates.append(
                EnrichedCandidate(
                    candidate_with_people=cwp,
                    enriched_dms=enriched_dms,
                    company_contact_email=company_email,
                    enrichment_status=enrichment_status,
                )
            )

        # Phone enrichment: top tier_1 with DMs, if explorium budget remains
        if self._explorium_remaining > 0:
            await self._enrich_phones(enriched_candidates, stats, api_credits)

        # Final stats
        stats["hunter_domain_calls"] = hunter_domain_cap - self._hunter_domain_remaining
        stats["hunter_finder_calls"] = hunter_finder_cap - self._hunter_finder_remaining
        stats["abstract_api_calls"] = abstract_api_cap - self._abstract_api_remaining
        stats["explorium_calls"] = explorium_cap - self._explorium_remaining

        self.log.info(
            "Enricher[%s]: dms=%d hunter_domain=%d hunter_finder=%d "
            "smtp_verified=%d abstract=%d explorium=%d final_emails=%d/%d",
            run_id,
            stats["dms_total"],
            stats["hunter_domain_calls"],
            stats["hunter_finder_calls"],
            stats["smtp_verifies"],
            stats["abstract_api_calls"],
            stats["explorium_calls"],
            stats["emails_found"],
            stats["dms_total"],
        )

        # Update run record
        try:
            await self.lead_store.update_run(
                run_id,
                api_credits_used=api_credits,
                status="contacts_enriched",
            )
        except Exception as exc:  # noqa: BLE001
            self.log.warning("Failed to update run record: %s", exc)

        completed_at = utcnow()
        return EnrichedResult(
            segment=segment,
            run_id=run_id,
            enriched_candidates=enriched_candidates,
            stats=stats,
            api_credits_used=api_credits,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=time.perf_counter() - start_ts,
        )

    # =========================================================================
    # Domain pattern retrieval
    # =========================================================================

    async def _get_domain_pattern(
        self,
        domain: str,
        tier: str,
        stats: dict,
        api_credits: dict,
    ) -> Optional[DomainPattern]:
        # Cache check first
        cached = await cache_get(_PATTERN_CACHE_METHOD, domain, self.settings)
        if cached is not None:
            self.log.info("contact_enricher: domain pattern cache hit for %s", domain)
            return DomainPattern(**cached)

        if tier != "tier_1" or self._hunter_domain_remaining <= 0:
            return None

        try:
            data = await self.hunter.domain_search(domain)
        except Exception as exc:  # noqa: BLE001
            self.log.warning("hunter.domain_search failed for %s: %s", domain, exc)
            return None

        if not data:
            return None

        self._hunter_domain_remaining -= 1
        api_credits["hunter_domain"] = api_credits.get("hunter_domain", 0) + 1

        pattern_str = data.get("pattern")
        org_name = data.get("organization")
        emails_raw = data.get("emails") or []
        known_emails = [e["value"] for e in emails_raw if e.get("value")]

        dp = DomainPattern(
            domain=domain,
            pattern=pattern_str,
            organization_name=org_name,
            known_emails=known_emails,
            fetched_at=utcnow(),
        )

        # Cache it
        await cache_set(
            _PATTERN_CACHE_METHOD,
            domain,
            dp.model_dump(mode="json"),
            _PATTERN_CACHE_TTL,
            self.settings,
        )
        return dp

    # =========================================================================
    # Per-DM email resolution cascade
    # =========================================================================

    async def _resolve_email_for_dm(
        self,
        dm: DecisionMaker,
        domain: str,
        domain_pattern: Optional[DomainPattern],
        tier: str,
        stats: dict,
        api_credits: dict,
    ) -> EmailResult:

        # Step 1: Hunter known email — exact name match
        if domain_pattern and domain_pattern.known_emails:
            for known_email in domain_pattern.known_emails:
                if _name_matches_email(dm.full_name, known_email):
                    self.log.info(
                        "contact_enricher: Hunter known email match for %s → %s",
                        dm.full_name, known_email,
                    )
                    return EmailResult(
                        email=known_email,
                        confidence=1.0,
                        source="hunter_known_email",
                        smtp_verified=False,
                    )

        # Step 2: Hunter email_finder (tier_1 only, within budget)
        if tier == "tier_1" and self._hunter_finder_remaining > 0:
            first, last = _split_name(dm.full_name)
            if first and last:
                try:
                    prospect = await self.hunter.email_finder(domain, first, last)
                except Exception as exc:  # noqa: BLE001
                    self.log.warning("hunter.email_finder error for %s: %s", dm.full_name, exc)
                    prospect = None

                self._hunter_finder_remaining -= 1
                api_credits["hunter_finder"] = api_credits.get("hunter_finder", 0) + 1

                if prospect and prospect.email:
                    return EmailResult(
                        email=prospect.email,
                        confidence=1.0,
                        source="hunter_finder",
                        smtp_verified=False,
                    )

        # Step 3: Generate candidate emails from pattern or defaults
        first, last = _split_name(dm.full_name)
        if not first or not last:
            return EmailResult(email=None, confidence=0.0, source="not_found")

        if domain_pattern and domain_pattern.pattern:
            candidates = [_apply_pattern(domain_pattern.pattern, first, last, domain)]
            candidates += _default_patterns(first, last, domain)[:2]
        else:
            candidates = _default_patterns(first, last, domain)

        # Step 4: SMTP verify each candidate
        verified_email: Optional[str] = None
        inconclusive_email: Optional[str] = None

        for candidate in candidates:
            try:
                smtp_result = await self.smtp.verify_email(candidate)
            except Exception as exc:  # noqa: BLE001
                self.log.warning("smtp_verifier error for %s: %s", candidate, exc)
                continue
            stats["smtp_verifies"] += 1
            api_credits["smtp"] = api_credits.get("smtp", 0) + 1

            if smtp_result.exists is True:
                verified_email = candidate
                break
            elif smtp_result.exists is None and inconclusive_email is None:
                inconclusive_email = candidate
                # Keep checking — a clear accept might still come

        chosen_email = verified_email or inconclusive_email

        if not chosen_email:
            return EmailResult(email=None, confidence=0.0, source="not_found")

        is_verified = verified_email is not None
        base_confidence = 0.7 if is_verified else 0.5
        source_trail = "pattern+smtp" if is_verified else "pattern+smtp_inconclusive"

        # Step 5: AbstractAPI deliverability check (tier_1 only, within budget)
        if tier == "tier_1" and self._abstract_api_remaining > 0:
            try:
                abstract_result = await self.abstract_api.validate_email(chosen_email)
            except Exception as exc:  # noqa: BLE001
                self.log.warning("abstract_api.validate_email error for %s: %s", chosen_email, exc)
                abstract_result = {}

            if abstract_result:
                self._abstract_api_remaining -= 1
                api_credits["abstract_api"] = api_credits.get("abstract_api", 0) + 1
                stats["abstract_api_calls"] = stats.get("abstract_api_calls", 0) + 1

                catchall = abstract_result.get("is_catchall_email", False)
                deliverability = abstract_result.get("deliverability")

                if catchall:
                    base_confidence = 0.4
                    source_trail += "+catchall_detected"
                elif deliverability == "DELIVERABLE":
                    base_confidence = 0.85
                    source_trail += "+abstract_valid"
                else:
                    base_confidence = 0.5
                    source_trail += "+abstract_uncertain"

                return EmailResult(
                    email=chosen_email,
                    confidence=base_confidence,
                    source=source_trail,
                    smtp_verified=is_verified,
                    catchall_detected=bool(catchall),
                    deliverability=deliverability,
                )

        return EmailResult(
            email=chosen_email,
            confidence=base_confidence,
            source=source_trail,
            smtp_verified=is_verified,
        )

    # =========================================================================
    # Common-prefix fallback
    # =========================================================================

    async def _try_common_prefixes(
        self,
        domain: str,
        stats: dict,
        api_credits: dict,
    ) -> Optional[EmailResult]:
        """Try generic company email prefixes for companies with no DMs (tier_1 only)."""
        for prefix in _COMMON_PREFIXES:
            email = f"{prefix}@{domain}"
            try:
                result = await self.smtp.verify_email(email)
            except Exception as exc:  # noqa: BLE001
                self.log.warning("smtp_verifier common prefix error for %s: %s", email, exc)
                continue
            stats["smtp_verifies"] += 1
            api_credits["smtp"] = api_credits.get("smtp", 0) + 1

            if result.exists is True:
                self.log.info("contact_enricher: common prefix hit → %s", email)
                return EmailResult(
                    email=email,
                    confidence=0.3,
                    source="common_prefix",
                    smtp_verified=True,
                )

        return None

    # =========================================================================
    # Phone enrichment
    # =========================================================================

    async def _enrich_phones(
        self,
        enriched_candidates: list[EnrichedCandidate],
        stats: dict,
        api_credits: dict,
    ) -> None:
        """Enrich phones for the single highest-value tier_1 candidate with DMs."""
        # Find the best tier_1 candidate that has enriched DMs
        best: Optional[EnrichedCandidate] = None
        for ec in enriched_candidates:
            if (
                ec.candidate_with_people.qualified.tier == "tier_1"
                and ec.enriched_dms
                and self._explorium_remaining > 0
            ):
                best = ec
                break

        if best is None:
            return

        domain: str = getattr(
            best.candidate_with_people.qualified.candidate, "domain", ""
        ) or ""
        if not domain:
            return

        dm_names = [
            edm.decision_maker.full_name for edm in best.enriched_dms
        ]
        self.log.info(
            "contact_enricher: phone enrichment for %s (%d DMs)",
            domain, len(dm_names),
        )

        try:
            prospects = await self.vibe.enrich_prospect_contacts(
                prospect_ids=[],  # Explorium expects prospect_ids; we pass domain instead
                contact_types=["phone"],
            )
        except Exception as exc:  # noqa: BLE001
            self.log.warning("explorium phone enrichment failed: %s", exc)
            return

        self._explorium_remaining -= 1
        api_credits["explorium"] = api_credits.get("explorium", 0) + 1
        stats["explorium_calls"] = stats.get("explorium_calls", 0) + 1

        # Match returned prospects back to enriched DMs by name
        prospect_map = {
            p.full_name.lower(): p.phone
            for p in prospects
            if p.phone
        }
        for edm in best.enriched_dms:
            phone = prospect_map.get(edm.decision_maker.full_name.lower())
            if phone:
                edm.phone = phone
                edm.phone_source = "explorium"

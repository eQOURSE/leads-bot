"""Qualifier agent — Phase 4.

Takes a HuntResult from the CompanyHunter, runs two-stage scoring
(cheap code-based pre-score, then Gemini Flash-Lite for survivors),
optionally resolves .unknown domains for high-scoring candidates,
and returns a QualifiedResult containing only candidates that clear
the ICP tier thresholds.

Two-stage savings:
  - Candidates with pre_score < 40 are dropped before any Gemini call.
  - Surviving candidates are batched 10-per-call to amortise prompt overhead.
  - Domain resolution only fires for .unknown domains with pre_score >= 60,
    capped at domain_resolution_cap per run.

Budget per run (3 segments, ~50 candidates each):
  - Gemini: ceil(survivors / 10) calls (logged if > 5 per run)
  - CompaniesAPI: <= domain_resolution_cap (default 5)
"""

from __future__ import annotations

import math
import time
from datetime import date, timedelta
from typing import TYPE_CHECKING, List, Optional, Tuple

from agents._gemini_wrapper import GeminiAgent
from agents._models import (
    BatchScoringResponse,
    GeminiScoringResult,
    HuntResult,
    IcpStrategy,
    QualifiedCandidate,
    QualifiedResult,
    QualifierSubScores,
)
from config.logging_config import setup_logging
from config.settings import Settings
from sources._utils import utcnow
from sources.models import CompanyCandidate

if TYPE_CHECKING:
    from agents.icp_strategist import IcpStrategist
    from sinks.sqlite_store import LeadStore
    from sources.companies_api_client import CompaniesAPIClient

_PRE_SCORE_THRESHOLD = 40   # below this → skip Gemini entirely
_RESOLUTION_PRESCORE_MIN = 60  # only resolve domains above this pre_score
_GEMINI_BATCH_SIZE = 10
_GEMINI_CALL_WARNING = 5    # log WARNING if more than this many Gemini calls/run

# ISO 3166-1 alpha-2 → common full-name variants (lowercase).
_COUNTRY_CODE_TO_NAMES: dict[str, list[str]] = {
    "us": ["united states", "united states of america", "usa"],
    "gb": ["united kingdom", "uk", "great britain", "england"],
    "ca": ["canada"],
    "au": ["australia"],
    "de": ["germany", "deutschland"],
    "fr": ["france"],
    "in": ["india"],
    "sg": ["singapore"],
    "nl": ["netherlands"],
    "ie": ["ireland"],
    "il": ["israel"],
}
_COUNTRY_NAME_TO_CODE: dict[str, str] = {
    name: code
    for code, names in _COUNTRY_CODE_TO_NAMES.items()
    for name in names
}


def _geo_matches(country_val: str, allowed: list[str]) -> bool:
    """Return True if country_val (from enrichment data) is in the allowed ICP list.

    Handles 2-letter codes ("us") vs. full names ("United States") in either direction.
    """
    if country_val in allowed:
        return True
    # Full name in data → look up its 2-letter code → check against allowed
    code = _COUNTRY_NAME_TO_CODE.get(country_val)
    if code and code in allowed:
        return True
    # 2-letter code in allowed → expand to full names → check against country_val
    for a in allowed:
        if country_val in _COUNTRY_CODE_TO_NAMES.get(a, []):
            return True
    return False

# Max points from pre-scoring components (must sum to 70)
_MAX_FUNDING_RECENCY = 40
_MAX_REACHABILITY = 10
_MAX_GEOGRAPHY = 10
_MAX_SIZE_MATCH = 10


class Qualifier:
    """Two-stage ICP scorer + domain resolver."""

    def __init__(
        self,
        settings: Settings,
        icp_strategist: "IcpStrategist",
        gemini_agent: GeminiAgent,
        companies_api_client: "CompaniesAPIClient",
        lead_store: "LeadStore",
    ) -> None:
        self.settings = settings
        self.icp_strategist = icp_strategist
        self.gemini = gemini_agent
        self.companies_api_client = companies_api_client
        self.lead_store = lead_store
        self.log = setup_logging("agent.qualifier")

    # =========================================================================
    # Public API
    # =========================================================================

    async def qualify(
        self,
        hunt_result: HuntResult,
        domain_resolution_cap: int = 5,
    ) -> QualifiedResult:
        started_at = utcnow()
        start_ts = time.perf_counter()
        segment = hunt_result.segment
        run_id = hunt_result.run_id
        api_credits: dict[str, int] = {"gemini": 0, "companies_api": 0}
        stats: dict = {}

        candidates: list[CompanyCandidate] = hunt_result.candidates  # type: ignore[assignment]

        # 1 — load ICP
        icp = self.icp_strategist.load_strategy(segment)

        # 2 — pre-score every candidate
        prescored: list[tuple[CompanyCandidate, int, dict]] = []
        auto_dropped: list[dict] = []
        baseline_bonus_count = 0

        for c in candidates:
            pre_score, sub = self._pre_score(c, icp)
            if sub.get("baseline_bonus_applied"):
                baseline_bonus_count += 1
            if pre_score == 0 and sub.get("auto_drop_reason"):
                auto_dropped.append({
                    "candidate_name": c.name,
                    "total_score": 0,
                    "drop_reason": sub["auto_drop_reason"],
                })
                continue
            if pre_score < _PRE_SCORE_THRESHOLD:
                auto_dropped.append({
                    "candidate_name": c.name,
                    "total_score": pre_score,
                    "drop_reason": f"pre_score {pre_score} < threshold {_PRE_SCORE_THRESHOLD}",
                })
                continue
            prescored.append((c, pre_score, sub))

        stats["pre_score_filtered"] = len(auto_dropped)
        stats["prescore_baseline_bonus_applied"] = baseline_bonus_count
        self.log.info(
            "Qualifier[%s] %d candidates, %d passed pre-score (>=40), %d auto-dropped",
            segment, len(candidates), len(prescored), len(auto_dropped),
        )

        # 3 — resolve .unknown domains for high-scorers
        resolved_count = 0
        resolved_candidates: list[tuple[CompanyCandidate, int, dict, bool]] = []
        # (candidate, pre_score, sub_scores_dict, domain_was_resolved)

        for c, pre_score, sub in prescored:
            was_resolved = False
            if (
                c.domain.endswith(".unknown")
                and pre_score >= _RESOLUTION_PRESCORE_MIN
                and resolved_count < domain_resolution_cap
            ):
                resolved_domain = await self._resolve_domain(c)
                api_credits["companies_api"] += 1
                resolved_count += 1
                if resolved_domain:
                    c = c.model_copy(update={"domain": resolved_domain})
                    was_resolved = True
                    self.log.info(
                        "Qualifier[%s] resolved domain for %s → %s",
                        segment, c.name, resolved_domain,
                    )
                else:
                    # Drop unresolved .unknown with high pre_score
                    self.log.warning(
                        "Qualifier[%s] dropped %s — domain unresolved",
                        segment, c.name,
                    )
                    auto_dropped.append({
                        "candidate_name": c.name,
                        "total_score": pre_score,
                        "drop_reason": "domain_unresolved",
                    })
                    continue
            resolved_candidates.append((c, pre_score, sub, was_resolved))

        stats["domains_resolved"] = resolved_count

        if not resolved_candidates:
            completed_at = utcnow()
            await self.lead_store.update_run(
                run_id,
                qualified_count=0,
                api_credits_used=api_credits,
            )
            return QualifiedResult(
                segment=segment,
                run_id=run_id,
                qualified=[],
                dropped=auto_dropped,
                stats=stats,
                api_credits_used=api_credits,
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=time.perf_counter() - start_ts,
            )

        # 4 — Gemini batch scoring
        survivor_candidates = [t[0] for t in resolved_candidates]
        gemini_results = await self._gemini_score_all(survivor_candidates, icp)
        n_batches = math.ceil(len(survivor_candidates) / _GEMINI_BATCH_SIZE)
        api_credits["gemini"] = n_batches
        stats["gemini_calls"] = n_batches

        if n_batches > _GEMINI_CALL_WARNING:
            self.log.warning(
                "Qualifier[%s] made %d Gemini calls this run (budget guidance: <= %d)",
                segment, n_batches, _GEMINI_CALL_WARNING,
            )

        # 5 — combine scores, assign tiers
        qualified: list[QualifiedCandidate] = []
        thresholds = icp.scoring_thresholds

        for idx, (c, pre_score, sub, was_resolved) in enumerate(resolved_candidates):
            g = gemini_results[idx]

            # Gemini disqualifiers override everything
            if g.disqualifiers:
                auto_dropped.append({
                    "candidate_name": c.name,
                    "total_score": pre_score,
                    "drop_reason": f"gemini_disqualifier: {'; '.join(g.disqualifiers)}",
                })
                continue

            total = min(100, pre_score + g.segment_fit_score + g.buying_signal_score)

            # Tier assignment
            if total < thresholds.auto_drop_below:
                auto_dropped.append({
                    "candidate_name": c.name,
                    "total_score": total,
                    "drop_reason": f"score {total} below auto_drop_below {thresholds.auto_drop_below}",
                })
                continue

            if total >= thresholds.tier_1_above:
                tier = "tier_1"
            else:
                tier = "tier_2"

            sub_scores = QualifierSubScores(
                funding_recency_score=sub.get("funding_recency", 0),
                reachability_score=sub.get("reachability", 0),
                geography_score=sub.get("geography", 0),
                size_match_score=sub.get("size_match", 0),
                segment_fit_score=g.segment_fit_score,
                buying_signal_score=g.buying_signal_score,
            )

            qualified.append(
                QualifiedCandidate(
                    candidate=c,
                    total_score=total,
                    pre_score=pre_score,
                    sub_scores=sub_scores,
                    reasoning=g.reasoning,
                    disqualifiers=[],
                    tier=tier,
                    domain_was_resolved=was_resolved,
                )
            )

        # Sort tier_1 first, then by score desc
        qualified.sort(key=lambda q: (0 if q.tier == "tier_1" else 1, -q.total_score))

        stats["tier_1_count"] = sum(1 for q in qualified if q.tier == "tier_1")
        stats["tier_2_count"] = sum(1 for q in qualified if q.tier == "tier_2")

        self.log.info(
            "Qualifier[%s] → tier_1=%d tier_2=%d dropped=%d | gemini_calls=%d",
            segment,
            stats["tier_1_count"],
            stats["tier_2_count"],
            len(auto_dropped),
            n_batches,
        )

        completed_at = utcnow()
        duration = time.perf_counter() - start_ts

        await self.lead_store.update_run(
            run_id,
            qualified_count=len(qualified),
            api_credits_used=api_credits,
            completed_at=completed_at,
            status="qualified",
        )

        return QualifiedResult(
            segment=segment,
            run_id=run_id,
            qualified=qualified,
            dropped=auto_dropped,
            stats=stats,
            api_credits_used=api_credits,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
        )

    # =========================================================================
    # Pre-scoring
    # =========================================================================

    def _pre_score(
        self, candidate: CompanyCandidate, icp: IcpStrategy
    ) -> Tuple[int, dict]:
        """Return (pre_score, component_dict). Pre-score is capped at 70."""
        sub: dict = {}
        neg_keywords = [sig.lower() for sig in icp.negative_signals]

        # Negative signal hard check — immediate zero
        haystack = " ".join(filter(None, [candidate.description, candidate.name])).lower()
        if haystack and any(kw in haystack for kw in neg_keywords):
            return 0, {"auto_drop_reason": "negative_signal_match"}

        # --- Funding recency (0-40) ---
        if candidate.funding_date is None:
            fr = 0
        else:
            days_ago = (date.today() - candidate.funding_date).days
            if days_ago <= 90:
                fr = 40
            elif days_ago <= 180:
                fr = 30
            elif days_ago <= 240:
                fr = 20
            else:
                fr = 0
        sub["funding_recency"] = fr

        # --- Reachability (0-10) ---
        reach = 0
        if candidate.domain and not candidate.domain.endswith(".unknown"):
            reach += 5
        if candidate.website:
            reach += 3
        if candidate.description and len(candidate.description) > 20:
            reach += 2
        sub["reachability"] = reach

        # --- Geography (0-10) ---
        allowed = [c.lower() for c in icp.target_company_profile.geographies.countries]
        if not allowed:
            geo = 10  # no restriction → full marks
        elif candidate.hq_country is None:
            geo = 5  # benefit of the doubt
        else:
            country_lower = candidate.hq_country.lower().strip()
            if _geo_matches(country_lower, allowed):
                geo = 10
            else:
                geo = 0
        sub["geography"] = geo

        # --- Company size match (0-10) ---
        if candidate.size_range is None:
            size = 5  # benefit of the doubt
        elif candidate.size_range in icp.target_company_profile.size_ranges:
            size = 10
        else:
            size = 0
        sub["size_match"] = size

        base_score = fr + reach + geo + size

        # --- Phase 13: minimum-viable-candidate baseline bonus ---
        # RSS-extracted candidates usually lack size_range/revenue_range, so they
        # cap out ~35-40 even when they're strong prospects. If a candidate has
        # the three signals we genuinely care about — a real (resolvable) domain,
        # a recent funding date, and an acceptable geography — give +10 so it can
        # clear the 40 gate and reach Gemini for the real judgment.
        from agents._constants import domain_is_news_source

        has_real_domain = bool(
            candidate.domain
            and not candidate.domain.endswith(".unknown")
            and not domain_is_news_source(candidate.domain)
        )
        has_funding_date = candidate.funding_date is not None and fr > 0
        geography_ok = geo > 0  # matched ICP geo, or no restriction, or benefit-of-doubt

        baseline_applied = False
        if has_real_domain and has_funding_date and geography_ok:
            base_score += 10
            baseline_applied = True
        sub["baseline_bonus_applied"] = baseline_applied

        total = min(70, base_score)
        return total, sub

    # =========================================================================
    # Domain resolution
    # =========================================================================

    async def _resolve_domain(self, candidate: CompanyCandidate) -> Optional[str]:
        """Try to find the real domain for a company with a .unknown placeholder.

        Calls CompaniesAPI search_by_filters; returns the first plausible domain
        found, or None on failure / no match.
        """
        from sources._utils import normalize_domain as _nd

        self.log.warning(
            "Qualifier: attempting domain resolution for %r (pre_score >= 60)",
            candidate.name,
        )
        try:
            results = await self.companies_api_client.search_by_filters(
                industries=None,
                countries=None,
                employee_range=None,
                limit=3,
            )
            if not results:
                return None

            # Prefer results whose name closely matches our target
            for r in results:
                d = _nd(r.domain) if r.domain else ""
                if d and ".unknown" not in d:
                    if candidate.name.lower() in r.name.lower() or \
                            r.name.lower() in candidate.name.lower():
                        return d

            # Fall back to first valid domain even without name match
            for r in results:
                d = _nd(r.domain) if r.domain else ""
                if d and ".unknown" not in d:
                    return d

            return None

        except Exception as exc:  # noqa: BLE001
            self.log.warning(
                "Domain resolution failed for %s: %s", candidate.name, exc
            )
            return None

    # =========================================================================
    # Gemini batch scoring
    # =========================================================================

    async def _gemini_score_all(
        self,
        candidates: list[CompanyCandidate],
        icp: IcpStrategy,
    ) -> list[GeminiScoringResult]:
        """Score all candidates in batches of GEMINI_BATCH_SIZE."""
        all_results: list[GeminiScoringResult] = []
        for i in range(0, len(candidates), _GEMINI_BATCH_SIZE):
            batch = candidates[i: i + _GEMINI_BATCH_SIZE]
            batch_results = await self._gemini_score_batch(batch, icp, offset=i)
            all_results.extend(batch_results)
        return all_results

    async def _gemini_score_batch(
        self,
        candidates: list[CompanyCandidate],
        icp: IcpStrategy,
        offset: int = 0,
    ) -> list[GeminiScoringResult]:
        """Score one batch of ≤10 candidates. Returns safe zeros on failure."""
        if not candidates:
            return []

        positive_signals = "\n".join(f"  - {s}" for s in icp.positive_signals)
        negative_signals = "\n".join(f"  - {s}" for s in icp.negative_signals)
        funding_stages = ", ".join(icp.target_company_profile.funding_stages)

        candidate_lines = []
        for local_idx, c in enumerate(candidates):
            global_idx = offset + local_idx + 1  # 1-based
            candidate_lines.append(
                f"[{global_idx}] Name: {c.name} | "
                f"Domain: {c.domain} | "
                f"Description: {c.description or 'N/A'} | "
                f"Stage: {c.funding_stage or 'N/A'} | "
                f"Date: {c.funding_date or 'N/A'} | "
                f"Industry: {c.industry or 'N/A'} | "
                f"Country: {c.hq_country or 'N/A'}"
            )
        candidates_block = "\n".join(candidate_lines)

        prompt = f"""You are evaluating B2B sales leads against this Ideal Customer Profile.

Segment: {icp.segment_name}
What we offer: {icp.what_we_offer}
Pain hypothesis: {icp.outreach_angle.pain_hypothesis}
Target funding stages: {funding_stages}

Positive signals to look for:
{positive_signals}

Negative signals (auto-disqualifiers if clearly present):
{negative_signals}

For EACH candidate below, score:
- segment_fit_score (0-15): how well does their product/stage match the ICP?
- buying_signal_score (0-15): do they show buying intent — hiring, launching, public pain admissions, funding recency?
- reasoning: one sentence why (be specific, not generic)
- disqualifiers: list any clear showstoppers from the negative signals (empty list if none)

Candidates:
{candidates_block}

Return STRICT JSON: {{"results": [{{"candidate_index": <1-based int>, "segment_fit_score": <int>, "buying_signal_score": <int>, "reasoning": "<string>", "disqualifiers": [<strings>]}}]}}
Include one entry per candidate. No commentary outside the JSON."""

        response = await self.gemini.generate_json(
            prompt, BatchScoringResponse, temperature=0.1
        )

        if response is None:
            self.log.warning(
                "Gemini batch scoring failed for batch at offset %d — using zeros",
                offset,
            )
            return [
                GeminiScoringResult(
                    candidate_index=offset + i + 1,
                    segment_fit_score=0,
                    buying_signal_score=0,
                    reasoning="gemini_unavailable",
                    disqualifiers=[],
                )
                for i in range(len(candidates))
            ]

        # Map by candidate_index back to list position
        result_map: dict[int, GeminiScoringResult] = {
            r.candidate_index: r for r in response.results
        }

        out: list[GeminiScoringResult] = []
        for local_idx in range(len(candidates)):
            global_idx = offset + local_idx + 1
            if global_idx in result_map:
                out.append(result_map[global_idx])
            else:
                # Gemini omitted this candidate — use safe zeros
                out.append(
                    GeminiScoringResult(
                        candidate_index=global_idx,
                        segment_fit_score=0,
                        buying_signal_score=0,
                        reasoning="not_scored_by_gemini",
                        disqualifiers=[],
                    )
                )
        return out

"""Pydantic models for the strategy (ICP) and agent layers.

ICP models mirror the structure of ``config/icp_configs.json`` exactly.

Hunter models define the HuntResult and extraction schemas for Phase 3.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class TargetIndustries(BaseModel):
    naics_codes: List[str] = Field(min_length=1)
    linkedin_categories: List[str] = Field(min_length=1)
    industry_keywords: List[str] = Field(min_length=1)


class Geographies(BaseModel):
    countries: List[str] = Field(default_factory=list)
    regions: List[str] = Field(default_factory=list)


class TargetCompanyProfile(BaseModel):
    size_ranges: List[str] = Field(min_length=1)
    revenue_ranges: List[str] = Field(default_factory=list)
    funding_stages: List[str] = Field(min_length=1)
    funding_recency_days: int = Field(gt=0)
    geographies: Geographies
    founded_after_year: int = Field(gt=1900)


class IcpScoringWeights(BaseModel):
    funding_recency: int = Field(ge=0, le=100)
    segment_fit: int = Field(ge=0, le=100)
    buying_signal: int = Field(ge=0, le=100)
    reachability: int = Field(ge=0, le=100)

    @model_validator(mode="after")
    def _must_sum_to_100(self) -> "IcpScoringWeights":
        total = (
            self.funding_recency
            + self.segment_fit
            + self.buying_signal
            + self.reachability
        )
        if total != 100:
            raise ValueError(f"scoring_weights must sum to 100 (got {total})")
        return self


class IcpScoringThresholds(BaseModel):
    auto_drop_below: int = Field(ge=0, le=100)
    tier_1_above: int = Field(ge=0, le=100)
    tier_2_above: int = Field(ge=0, le=100)

    @model_validator(mode="after")
    def _must_be_ordered(self) -> "IcpScoringThresholds":
        # auto_drop <= tier_2 <= tier_1.
        # Equality between auto_drop_below and tier_2_above is intentional and
        # valid: it means the drop boundary and the tier-2 floor are the same
        # score (e.g. 70), so a lead at exactly that score is kept as tier-2 and
        # anything below is dropped — no gap, no overlap.
        if not (self.auto_drop_below <= self.tier_2_above <= self.tier_1_above):
            raise ValueError(
                "thresholds must satisfy: auto_drop_below <= tier_2_above <= "
                f"tier_1_above (got auto_drop={self.auto_drop_below}, "
                f"tier_2={self.tier_2_above}, tier_1={self.tier_1_above})"
            )
        return self


class IcpOutreachAngle(BaseModel):
    pain_hypothesis: str = Field(min_length=1)
    value_framing: str = Field(min_length=1)
    primary_cta: str = Field(min_length=1)
    fallback_cta: str = Field(min_length=1)


class IcpStrategy(BaseModel):
    """Full ICP definition for one segment."""

    segment_name: str = Field(min_length=1)
    value_prop_one_liner: str = Field(min_length=1)
    what_we_offer: str = Field(min_length=1)
    target_industries: TargetIndustries
    target_company_profile: TargetCompanyProfile
    target_titles: List[str] = Field(min_length=1)
    target_departments: List[str] = Field(min_length=1)
    target_levels: List[str] = Field(min_length=1)
    positive_signals: List[str] = Field(min_length=1)
    negative_signals: List[str] = Field(min_length=3)
    scoring_weights: IcpScoringWeights
    scoring_thresholds: IcpScoringThresholds
    outreach_angle: IcpOutreachAngle

    @field_validator("target_titles")
    @classmethod
    def _titles_unique_case_insensitive(cls, v: List[str]) -> List[str]:
        seen = set()
        for title in v:
            key = title.strip().lower()
            if key in seen:
                raise ValueError(f"duplicate target title (case-insensitive): {title!r}")
            seen.add(key)
        return v


# ---------------------------------------------------------------------------
# Phase 3 — Company Hunter models
# ---------------------------------------------------------------------------

class ExtractedFromSearch(BaseModel):
    """Company data extracted from a search result or news headline via Gemini."""

    company_name: Optional[str] = None
    domain: Optional[str] = None
    funding_amount_usd: Optional[float] = None
    funding_stage: Optional[str] = None
    announcement_date: Optional[date] = None
    description: Optional[str] = None


class HuntResult(BaseModel):
    """The output of one CompanyHunter.hunt() run."""

    segment: str
    run_id: str
    candidates: List  # list[CompanyCandidate] — typed at runtime to avoid circular
    source_counts: Dict[str, int]      # {"rss": 23, "serpapi": 12, "newsdata": 8}
    merged_count: int
    after_filter: int
    after_dedupe: int
    enriched_count: int
    api_credits_used: Dict[str, int]   # {"serpapi": 1, "newsdata": 2, ...}
    errors: List[str]
    started_at: datetime
    completed_at: datetime
    duration_seconds: float


# ---------------------------------------------------------------------------
# Phase 4 — Qualifier models
# ---------------------------------------------------------------------------

from typing import Literal  # noqa: E402 — after existing imports


class QualifierSubScores(BaseModel):
    funding_recency_score: int
    reachability_score: int
    geography_score: int
    size_match_score: int
    segment_fit_score: int   # from Gemini (0-15)
    buying_signal_score: int  # from Gemini (0-15)


class QualifiedCandidate(BaseModel):
    candidate: object          # CompanyCandidate — avoided circular import
    total_score: int
    pre_score: int
    sub_scores: QualifierSubScores
    reasoning: str
    disqualifiers: List[str]
    tier: Literal["tier_1", "tier_2"]
    domain_was_resolved: bool


class QualifiedResult(BaseModel):
    segment: str
    run_id: str
    qualified: List[QualifiedCandidate]   # tier_1 and tier_2 only
    dropped: List[Dict]                   # {candidate_name, total_score, drop_reason}
    stats: Dict                           # counts, gemini_calls, domains_resolved, ...
    api_credits_used: Dict[str, int]
    started_at: datetime
    completed_at: datetime
    duration_seconds: float


class GeminiScoringResult(BaseModel):
    """One candidate's scores from the Gemini batch-scoring call."""
    candidate_index: int
    segment_fit_score: int   # 0-15
    buying_signal_score: int  # 0-15
    reasoning: str
    disqualifiers: List[str] = Field(default_factory=list)


class BatchScoringResponse(BaseModel):
    """Wrapper for the full Gemini batch-scoring response."""
    results: List[GeminiScoringResult]


# ---------------------------------------------------------------------------
# Phase 5 — Decision-Maker Finder models
# ---------------------------------------------------------------------------




class DecisionMaker(BaseModel):
    """A resolved decision-maker at a qualified company."""

    full_name: str
    title: str
    linkedin_url: Optional[str] = None
    source: Literal["scrapegraph", "apify_linkedin", "explorium"]
    seniority_score: int
    email: Optional[str] = None      # Phase 6
    phone: Optional[str] = None      # Phase 6


class QualifiedCandidateWithPeople(BaseModel):
    """A qualified company candidate enriched with decision-maker lookups."""

    qualified: QualifiedCandidate
    decision_makers: List[DecisionMaker]
    lookup_status: Literal["found", "no_decision_maker", "needs_manual_lookup"]
    # Maps source name → outcome string, e.g. {"scrapegraph": "no_results", "apify": "not_attempted"}
    lookup_attempts: Dict[str, str]


class EnhancedQualifiedResult(BaseModel):
    """Output of DecisionMakerFinder.find_for_qualified()."""

    segment: str
    run_id: str
    candidates_with_people: List[QualifiedCandidateWithPeople]
    needs_manual_lookup: List[QualifiedCandidate]
    stats: Dict
    api_credits_used: Dict[str, int]
    started_at: datetime
    completed_at: datetime
    duration_seconds: float

# ---------------------------------------------------------------------------
# Phase 6 — Contact Enricher models
# ---------------------------------------------------------------------------


class DomainPattern(BaseModel):
    """Cached result of a Hunter domain_search for a company domain."""

    domain: str
    pattern: Optional[str] = None          # e.g. "{first}.{last}"
    organization_name: Optional[str] = None
    known_emails: List[str] = Field(default_factory=list)
    fetched_at: datetime


class EmailResult(BaseModel):
    """Outcome of the email cascade for one decision-maker."""

    email: Optional[str] = None
    confidence: float = 0.0                # 0.0 – 1.0
    source: str = "not_found"              # "hunter_finder", "hunter_known_email",
                                           #   "pattern+smtp", "common_prefix", etc.
    smtp_verified: bool = False
    catchall_detected: bool = False
    deliverability: Optional[str] = None   # from AbstractAPI


class EnrichedDecisionMaker(BaseModel):
    """A decision-maker with an attached email result and optional phone."""

    decision_maker: DecisionMaker
    email_result: EmailResult
    phone: Optional[str] = None
    phone_source: Optional[str] = None


class EnrichedCandidate(BaseModel):
    """A qualified company candidate with enriched decision-maker contacts."""

    candidate_with_people: QualifiedCandidateWithPeople
    enriched_dms: List[EnrichedDecisionMaker]
    company_contact_email: Optional[EmailResult] = None  # common-prefix fallback
    enrichment_status: Literal["full", "partial", "no_emails", "skipped"]


class EnrichedResult(BaseModel):
    """Output of ContactEnricher.enrich()."""

    segment: str
    run_id: str
    enriched_candidates: List[EnrichedCandidate]
    stats: Dict
    api_credits_used: Dict[str, int]
    started_at: datetime
    completed_at: datetime
    duration_seconds: float


# ---------------------------------------------------------------------------
# Phase 7 — Personalizer + Message Writer models
# ---------------------------------------------------------------------------


class PersonalizationContext(BaseModel):
    """Synthesised hook context for one company domain."""

    domain: str
    company_one_liner: str
    recent_milestone: Optional[str] = None
    pain_hypothesis_specific: str
    why_now_hook: str
    personalization_quality: Literal["high", "medium", "low"]
    built_at: datetime


class GeneratedMessages(BaseModel):
    """All outreach copy for one decision-maker, as output by Gemini Pro."""

    email_subject_a: str
    email_subject_b: str
    email_body: str
    linkedin_dm: str
    reply_likelihood: int         # 0-10 (post-validation adjusted)
    quality_flags: List[str] = Field(default_factory=list)


class MessagedDecisionMaker(BaseModel):
    """A decision-maker with attached generated messages (or a skip reason)."""

    enriched_dm: object           # EnrichedDecisionMaker — avoid circular at model level
    messages: Optional[GeneratedMessages] = None
    skipped_reason: Optional[str] = None  # "no_email", "low_confidence", etc.


class MessagedCandidate(BaseModel):
    """A company candidate with personalisation context + messaged DMs."""

    enriched_candidate: object    # EnrichedCandidate
    personalization: Optional[PersonalizationContext] = None
    messaged_dms: List[MessagedDecisionMaker]


class MessagedResult(BaseModel):
    """Output of MessageWriter.write_for_enriched()."""

    segment: str
    run_id: str
    messaged_candidates: List[MessagedCandidate]
    stats: Dict
    api_credits_used: Dict[str, int]
    started_at: datetime
    completed_at: datetime
    duration_seconds: float


# ---------------------------------------------------------------------------
# Phase 8 — Validator + Sinks models
# ---------------------------------------------------------------------------


class ValidatedDecisionMaker(BaseModel):
    """A messaged DM with its validation outcome."""

    messaged_dm: object                  # MessagedDecisionMaker
    status: Literal["ready_to_send", "needs_review", "rejected"]
    validation_reasons: List[str]        # empty = passed all checks
    lead_hash: str


class ValidatedCandidate(BaseModel):
    """A messaged company candidate with validated DMs."""

    messaged_candidate: object           # MessagedCandidate
    validated_dms: List[ValidatedDecisionMaker]


class ValidatedResult(BaseModel):
    """Output of Validator.validate()."""

    segment: str
    run_id: str
    validated_candidates: List[ValidatedCandidate]
    stats: Dict          # {ready_to_send: N, needs_review: N, rejected: N}
    api_credits_used: Dict[str, int]
    started_at: datetime
    completed_at: datetime
    duration_seconds: float


class SentResult(BaseModel):
    """Output of SinkOrchestrator.dispatch()."""

    segment: str
    run_id: str
    sqlite_inserted: int
    sqlite_skipped: int
    sheets_appended: int
    sheets_errors: List[str]
    telegram_message_id: Optional[int] = None
    telegram_error: Optional[str] = None
    duration_seconds: float

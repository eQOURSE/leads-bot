"""Message Writer agent — Phase 7.

Takes an EnrichedResult + a personalization map (domain → PersonalizationContext)
and generates cold outreach for every DM that has a sendable email address.

One Gemini Pro call per DM returns all four outputs:
  - email_subject_a  (curiosity-driven)
  - email_subject_b  (value-driven)
  - email_body       (90-130 words, 3 paragraphs, one CTA)
  - linkedin_dm      (<280 chars)
  - reply_likelihood (0-10)
  - quality_flags    (self-reported rule violations)

Post-processing validates hard rules and adjusts reply_likelihood.

Budget: one Gemini Pro call per DM with email. Concurrent calls limited by
max_concurrent_calls (default 5).
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import TYPE_CHECKING, Optional

from agents._models import (
    EnrichedCandidate,
    EnrichedDecisionMaker,
    EnrichedResult,
    GeneratedMessages,
    IcpStrategy,
    MessagedCandidate,
    MessagedDecisionMaker,
    MessagedResult,
    PersonalizationContext,
    QualifiedCandidate,
)
from config.logging_config import setup_logging
from config.logging_config import setup_logging
from config.settings import Settings
from sources._utils import utcnow

if TYPE_CHECKING:
    from agents.icp_strategist import IcpStrategist
    from agents._gemini_wrapper import GeminiAgent
    from sinks.sqlite_store import LeadStore

# Banned phrases — any match in body or subject adds a quality flag
_BANNED_PHRASES = [
    "i hope this email finds you well",
    "i came across",
    "i noticed",
    "synergies",
    "leverage",
    "circle back",
    "touch base",
    "low-hanging fruit",
    "ping me",
    "reach out",
    "wanted to connect",
]

# Patterns that suggest multiple CTAs in the body
_CTA_PATTERNS = [
    r"happy to\b",
    r"let me know\b",
    r"open to\b",
    r"would you\b",
    r"interested in\b",
    r"want to\b",
]

_SIGNOFF_PATTERNS = ["best,", "cheers,", "regards,", "thanks,", "sincerely,", "warm regards,"]


class MessageWriter:
    """Generate personalised cold outreach messages for enriched decision-makers."""

    def __init__(
        self,
        settings: Settings,
        icp_strategist: "IcpStrategist",
        gemini_agent: "GeminiAgent",     # should use GEMINI_MODEL_WRITER (Pro)
        lead_store: "LeadStore",
    ) -> None:
        self.settings = settings
        self.icp_strategist = icp_strategist
        self.gemini = gemini_agent
        self.lead_store = lead_store
        self.log = setup_logging("agent.message_writer")

    # =========================================================================
    # Public API
    # =========================================================================

    async def write_for_enriched(
        self,
        enriched_result: EnrichedResult,
        personalization_map: dict[str, PersonalizationContext],
        min_email_confidence: float = 0.3,
        max_concurrent_calls: int = 5,
    ) -> MessagedResult:
        started_at = utcnow()
        start_ts = time.perf_counter()
        segment = enriched_result.segment
        run_id = enriched_result.run_id

        icp = self.icp_strategist.load_strategy(segment)

        # Collect all (ec, edm) pairs that have a sendable email
        work_items: list[tuple[EnrichedCandidate, EnrichedDecisionMaker]] = []
        for ec in enriched_result.enriched_candidates:
            for edm in ec.enriched_dms:
                email = edm.email_result.email
                conf = edm.email_result.confidence
                if email and conf >= min_email_confidence:
                    work_items.append((ec, edm))

        # Process with semaphore
        semaphore = asyncio.Semaphore(max_concurrent_calls)

        async def _process(ec: EnrichedCandidate, edm: EnrichedDecisionMaker):
            async with semaphore:
                return await self._process_one(ec, edm, personalization_map, icp)

        pairs = await asyncio.gather(*[_process(ec, edm) for ec, edm in work_items])

        # Build a map: ec_id → list of (edm, result) so we can regroup by candidate
        from collections import defaultdict
        ec_to_results: dict[int, list[tuple[EnrichedDecisionMaker, MessagedDecisionMaker]]] = defaultdict(list)
        for (ec, edm), mdm in zip(work_items, pairs):
            ec_to_results[id(ec)].append((edm, mdm))

        # Build MessagedCandidate list preserving order
        messaged_candidates: list[MessagedCandidate] = []
        for ec in enriched_result.enriched_candidates:
            domain: str = getattr(
                ec.candidate_with_people.qualified.candidate, "domain", ""
            ) or ""

            # DMs that were in work_items for this ec
            worked_edms = {id(edm): mdm for edm, mdm in ec_to_results.get(id(ec), [])}

            messaged_dms: list[MessagedDecisionMaker] = []
            for edm in ec.enriched_dms:
                if id(edm) in worked_edms:
                    messaged_dms.append(worked_edms[id(edm)])
                else:
                    # DM was skipped (no email or below confidence)
                    email = edm.email_result.email
                    if not email:
                        reason = "no_email"
                    elif edm.email_result.confidence < min_email_confidence:
                        reason = "low_confidence"
                    else:
                        reason = "skipped"
                    messaged_dms.append(
                        MessagedDecisionMaker(
                            enriched_dm=edm,
                            messages=None,
                            skipped_reason=reason,
                        )
                    )

            messaged_candidates.append(
                MessagedCandidate(
                    enriched_candidate=ec,
                    personalization=personalization_map.get(domain),
                    messaged_dms=messaged_dms,
                )
            )

        # Stats
        total_dms = sum(len(mc.messaged_dms) for mc in messaged_candidates)
        generated = sum(
            1 for mc in messaged_candidates
            for mdm in mc.messaged_dms
            if mdm.messages is not None
        )
        skipped_no_email = sum(
            1 for mc in messaged_candidates
            for mdm in mc.messaged_dms
            if mdm.skipped_reason == "no_email"
        )
        all_likelihoods = [
            mdm.messages.reply_likelihood
            for mc in messaged_candidates
            for mdm in mc.messaged_dms
            if mdm.messages is not None
        ]
        all_flag_counts = [
            len(mdm.messages.quality_flags)
            for mc in messaged_candidates
            for mdm in mc.messaged_dms
            if mdm.messages is not None
        ]
        avg_likelihood = sum(all_likelihoods) / len(all_likelihoods) if all_likelihoods else 0.0
        avg_flags = sum(all_flag_counts) / len(all_flag_counts) if all_flag_counts else 0.0

        stats = {
            "total_dms": total_dms,
            "messages_generated": generated,
            "skipped_no_email": skipped_no_email,
            "avg_reply_likelihood": round(avg_likelihood, 2),
            "avg_quality_flags": round(avg_flags, 2),
        }

        self.log.info(
            "MessageWriter[%s]: dms=%d generated=%d skipped=%d "
            "avg_likelihood=%.1f avg_flags=%.1f",
            run_id,
            total_dms,
            generated,
            total_dms - generated,
            avg_likelihood,
            avg_flags,
        )

        try:
            await self.lead_store.update_run(
                run_id,
                api_credits_used={"gemini_writer": generated},
                status="messages_written",
            )
        except Exception as exc:  # noqa: BLE001
            self.log.warning("Failed to update run record: %s", exc)

        completed_at = utcnow()
        return MessagedResult(
            segment=segment,
            run_id=run_id,
            messaged_candidates=messaged_candidates,
            stats=stats,
            api_credits_used={"gemini_writer": generated},
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=time.perf_counter() - start_ts,
        )

    # =========================================================================
    # Private: process one (candidate, DM) pair
    # =========================================================================

    async def _process_one(
        self,
        ec: EnrichedCandidate,
        edm: EnrichedDecisionMaker,
        personalization_map: dict[str, PersonalizationContext],
        icp: IcpStrategy,
    ) -> MessagedDecisionMaker:
        qc: QualifiedCandidate = ec.candidate_with_people.qualified
        domain: str = getattr(qc.candidate, "domain", "") or ""  # type: ignore[attr-defined]
        hook = personalization_map.get(domain)

        if hook:
            messages = await self._write_messages_for_dm(qc, edm, hook, icp)
            skipped_reason = None
        else:
            messages = await self._write_fallback_messages(qc, edm, icp)
            skipped_reason = None  # still generated, just flagged

        if messages is None:
            return MessagedDecisionMaker(
                enriched_dm=edm,
                messages=None,
                skipped_reason="gemini_invalid_output",
            )

        return MessagedDecisionMaker(
            enriched_dm=edm,
            messages=messages,
            skipped_reason=skipped_reason,
        )

    async def _write_messages_for_dm(
        self,
        qualified: QualifiedCandidate,
        edm: EnrichedDecisionMaker,
        hook: PersonalizationContext,
        icp: IcpStrategy,
    ) -> Optional[GeneratedMessages]:
        dm = edm.decision_maker
        candidate = qualified.candidate  # type: ignore[attr-defined]

        prompt = f"""You are writing cold outreach for a B2B sales rep. NEVER write generic spam.

# Sender context
We are {icp.segment_name}: {icp.value_prop_one_liner}.
What we offer: {icp.what_we_offer}
Our pain hypothesis: {icp.outreach_angle.pain_hypothesis}
Value framing: {icp.outreach_angle.value_framing}
Primary CTA: {icp.outreach_angle.primary_cta}
Fallback CTA: {icp.outreach_angle.fallback_cta}

# Recipient context
Name: {dm.full_name}
Title: {dm.title}
Company: {getattr(candidate, "name", "")} ({getattr(candidate, "domain", "")})
Tier: {qualified.tier}
Why-now hook (USE THIS in the opener): {hook.why_now_hook}
Company one-liner: {hook.company_one_liner}
Specific pain hypothesis: {hook.pain_hypothesis_specific}

# Task
Write ALL FOUR outputs in a single response:

1. email_subject_a — curiosity-driven, under 50 chars, no clickbait, no "Re:" or "FW:"
2. email_subject_b — value-driven, under 50 chars, mentions a concrete number/role/product if possible
3. email_body — 90 to 130 words, structured as:
   Paragraph 1 (1-2 sentences): Open with the why_now_hook. Be specific.
   Paragraph 2 (1-2 sentences): Bridge to the pain — apply pain_hypothesis_specific to them.
   Paragraph 3 (1-2 sentences): The offer — one concrete sentence on what we'd do for them. Then ONE clear CTA from the ICP options.
   Sign-off: "Best, [Your Name] | eQOURSE x TUTRAIN"
4. linkedin_dm — under 280 chars, casual, references the milestone, soft ask, NO LINK
5. reply_likelihood — 0-10, your honest assessment of how likely a reply is
6. quality_flags — array of any concerns you have about this draft

# HARD RULES — violate any and quality_flags must include the violation
- BANNED PHRASES: "I hope this email finds you well", "I came across", "I noticed", "synergies", "leverage", "circle back", "touch base", "low-hanging fruit", "ping me", "reach out", "wanted to connect"
- ONE CTA ONLY in email_body. If you have an urge to write "happy to chat OR send info OR jump on a call", pick ONE.
- The opener MUST reference something specific about THIS company. No generic "noticed you are in the space" energy.
- email_body word count must be 90-130. Count strictly.
- Subjects under 50 chars including spaces.
- No exclamation marks in subjects.

Return strict JSON only:
{{
  "email_subject_a": "...",
  "email_subject_b": "...",
  "email_body": "...",
  "linkedin_dm": "...",
  "reply_likelihood": <int 0-10>,
  "quality_flags": ["...", "..."]
}}"""

        messages = await self.gemini.generate_json(
            prompt, GeneratedMessages, temperature=0.6
        )
        if messages is None:
            return None

        return self._post_process_validate(messages)

    async def _write_fallback_messages(
        self,
        qualified: QualifiedCandidate,
        edm: EnrichedDecisionMaker,
        icp: IcpStrategy,
    ) -> Optional[GeneratedMessages]:
        """Fallback when personalization context is unavailable."""
        dm = edm.decision_maker
        candidate = qualified.candidate  # type: ignore[attr-defined]

        prompt = f"""You are writing cold outreach for a B2B sales rep.

# Sender context
We are {icp.segment_name}: {icp.value_prop_one_liner}.
What we offer: {icp.what_we_offer}
Value framing: {icp.outreach_angle.value_framing}
Primary CTA: {icp.outreach_angle.primary_cta}

# Recipient context
Name: {dm.full_name}
Title: {dm.title}
Company: {getattr(candidate, "name", "")} ({getattr(candidate, "domain", "")})
Tier: {qualified.tier}

Note: We do not have specific recent news for this company. Write a value-focused
email without a time-specific hook. Keep it honest and professional.

# Task
Write ALL FOUR outputs. Same format as a normal outreach email.
Follow all the same rules. Tag quality_flags with "fallback_no_personalization".
Max reply_likelihood = 5 (no hook available).

Return strict JSON only:
{{
  "email_subject_a": "...",
  "email_subject_b": "...",
  "email_body": "...",
  "linkedin_dm": "...",
  "reply_likelihood": <int 0-5>,
  "quality_flags": ["fallback_no_personalization"]
}}"""

        messages = await self.gemini.generate_json(
            prompt, GeneratedMessages, temperature=0.5
        )
        if messages is None:
            return None

        messages = self._post_process_validate(messages)
        # Enforce fallback cap
        messages.reply_likelihood = min(messages.reply_likelihood, 5)
        if "fallback_no_personalization" not in messages.quality_flags:
            messages.quality_flags.append("fallback_no_personalization")
        return messages

    # =========================================================================
    # Post-processing validation
    # =========================================================================

    def _post_process_validate(self, messages: GeneratedMessages) -> GeneratedMessages:
        """Check hard rules, add quality_flags, penalise reply_likelihood."""
        flags = list(messages.quality_flags)

        # Subject length and exclamation checks
        if len(messages.email_subject_a) > 50:
            flags.append("subject_a_too_long")
        if len(messages.email_subject_b) > 50:
            flags.append("subject_b_too_long")
        if "!" in messages.email_subject_a or "!" in messages.email_subject_b:
            flags.append("subject_has_exclamation")

        # Body word count
        word_count = len(messages.email_body.split())
        if word_count < 80:
            flags.append("body_word_count_low")
        elif word_count > 145:
            flags.append("body_word_count_high")

        # LinkedIn DM length
        if len(messages.linkedin_dm) > 280:
            flags.append("linkedin_dm_too_long")

        # Sign-off check
        body_lower = messages.email_body.lower()
        has_signoff = any(p in body_lower for p in _SIGNOFF_PATTERNS)
        if not has_signoff:
            flags.append("missing_signoff")

        # Banned phrases (check body + both subjects)
        full_text = (
            messages.email_body + " " +
            messages.email_subject_a + " " +
            messages.email_subject_b
        ).lower()
        for phrase in _BANNED_PHRASES:
            if phrase in full_text:
                safe_key = phrase.replace(" ", "_")[:30]
                flags.append(f"banned_phrase_{safe_key}")

        # Multiple CTAs detection
        cta_matches = sum(
            1 for pattern in _CTA_PATTERNS
            if re.search(pattern, body_lower)
        )
        if cta_matches >= 2:
            flags.append("multiple_ctas")

        # Deduplicate flags
        seen: set[str] = set()
        deduped_flags: list[str] = []
        for f in flags:
            if f not in seen:
                seen.add(f)
                deduped_flags.append(f)

        # Penalise reply_likelihood: each flag costs 1 point
        penalty = len(deduped_flags)
        new_likelihood = max(0, messages.reply_likelihood - penalty)

        messages.quality_flags = deduped_flags
        messages.reply_likelihood = new_likelihood
        return messages


# ---------------------------------------------------------------------------
# Standalone validator for direct unit-testing
# ---------------------------------------------------------------------------

def _post_process_validate_standalone(messages: "GeneratedMessages") -> "GeneratedMessages":
    """Module-level alias so tests can call the validator directly."""
    # Instantiate a throwaway writer with minimal deps
    _writer = object.__new__(MessageWriter)
    _writer.log = setup_logging("agent.message_writer")
    return _writer._post_process_validate(messages)

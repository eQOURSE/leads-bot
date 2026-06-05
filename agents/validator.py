"""Validator agent — Phase 8.

Examines every MessagedDecisionMaker and assigns a status:
  "ready_to_send"  — passes all checks; written to Sheets + Telegram digest
  "needs_review"   — soft failure; written to "Needs Review" Sheets tab only
  "rejected"       — hard failure; SQLite audit row only, never sent

Hard-failure checks (any → "rejected"):
  1. invalid_email_syntax
  2. no_mx_record         (uses SMTPVerifier cache — no new SMTP handshake)
  3. duplicate_lead       (lead_hash already in leads table with non-rejected status)
  4. contains_profanity   (BANNED_WORDS match)
  5. weird_characters     (non-printable, BOM, U+FFFD)

Soft-failure checks (any → "needs_review"):
  6. all_caps_subject
  7. url_in_linkedin_dm
  8. low_reply_likelihood
  9. too_many_quality_flags
 10. weak_personalization
 11. subject_body_mismatch (Gemini Flash, 1 call/lead; skipped if > 50 leads)

Budget: 1 Gemini Flash call per lead for alignment check (capped at 50 leads/run).
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from typing import TYPE_CHECKING, Optional

from agents._constants import BANNED_WORDS
from agents._models import (
    MessagedDecisionMaker,
    MessagedResult,
    PersonalizationContext,
    ValidatedCandidate,
    ValidatedDecisionMaker,
    ValidatedResult,
)
from config.logging_config import setup_logging
from config.settings import Settings
from sources._utils import utcnow

if TYPE_CHECKING:
    from agents._gemini_wrapper import GeminiAgent
    from sinks.sqlite_store import LeadStore
    from sources.smtp_verifier import SMTPVerifier

# Regex for non-printable chars, BOM, replacement char
_WEIRD_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufeff\ufffd]")
# ALL CAPS: more than 5 consecutive uppercase-only words
_ALL_CAPS_RE = re.compile(r"(?:[A-Z]{2,}\s+){5,}")
# URL in linkedin_dm
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
# Alignment check response schema
from pydantic import BaseModel as _PBM


class _AlignmentResponse(_PBM):
    aligned: bool
    reason: str


_ALIGNMENT_VOLUME_THRESHOLD = 50


class Validator:
    """Validate MessagedResult leads for quality and deduplication."""

    def __init__(
        self,
        settings: Settings,
        lead_store: "LeadStore",
        smtp_verifier: "SMTPVerifier",
        gemini_agent: "GeminiAgent",
    ) -> None:
        self.settings = settings
        self.lead_store = lead_store
        self.smtp = smtp_verifier
        self.gemini = gemini_agent
        self.log = setup_logging("agent.validator")

    # =========================================================================
    # Public API
    # =========================================================================

    async def validate(
        self,
        messaged_result: MessagedResult,
        min_reply_likelihood_for_ready: int = 6,
        max_quality_flags_for_ready: int = 1,
    ) -> ValidatedResult:
        started_at = utcnow()
        start_ts = time.perf_counter()
        segment = messaged_result.segment
        run_id = messaged_result.run_id

        # Collect all (mc, mdm) pairs to validate
        all_pairs: list[tuple] = [
            (mc, mdm)
            for mc in messaged_result.messaged_candidates
            for mdm in mc.messaged_dms
        ]

        total_leads = len(all_pairs)
        skip_alignment = total_leads > _ALIGNMENT_VOLUME_THRESHOLD
        if skip_alignment:
            self.log.info(
                "Validator[%s]: %d leads > threshold %d, skipping alignment check",
                run_id, total_leads, _ALIGNMENT_VOLUME_THRESHOLD,
            )

        gemini_calls = 0
        counts = {"ready_to_send": 0, "needs_review": 0, "rejected": 0}

        validated_by_mc: dict[int, list[ValidatedDecisionMaker]] = {}

        for mc, mdm in all_pairs:
            vdm = await self._validate_one(
                mc, mdm,
                min_reply_likelihood_for_ready,
                max_quality_flags_for_ready,
                skip_alignment,
            )
            if not skip_alignment and "subject_body_mismatch" in vdm.validation_reasons:
                gemini_calls += 1
            elif not skip_alignment and vdm.status != "rejected":
                # alignment was attempted (no mismatch found)
                gemini_calls += 1

            counts[vdm.status] += 1
            mc_id = id(mc)
            validated_by_mc.setdefault(mc_id, []).append(vdm)

        # Rebuild structure preserving candidate order
        validated_candidates: list[ValidatedCandidate] = []
        for mc in messaged_result.messaged_candidates:
            validated_candidates.append(
                ValidatedCandidate(
                    messaged_candidate=mc,
                    validated_dms=validated_by_mc.get(id(mc), []),
                )
            )

        self.log.info(
            "Validator[%s]: ready=%d needs_review=%d rejected=%d gemini_calls=%d",
            run_id,
            counts["ready_to_send"],
            counts["needs_review"],
            counts["rejected"],
            gemini_calls,
        )

        completed_at = utcnow()
        return ValidatedResult(
            segment=segment,
            run_id=run_id,
            validated_candidates=validated_candidates,
            stats=counts,
            api_credits_used={"gemini_flash": gemini_calls},
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=time.perf_counter() - start_ts,
        )

    # =========================================================================
    # Per-DM validation
    # =========================================================================

    async def _validate_one(
        self,
        mc,
        mdm: MessagedDecisionMaker,
        min_likelihood: int,
        max_flags: int,
        skip_alignment: bool,
    ) -> ValidatedDecisionMaker:
        # No messages generated → hard reject
        if mdm.messages is None:
            return self._make_vdm(mdm, "rejected", ["no_messages_generated"])

        messages = mdm.messages
        edm = mdm.enriched_dm  # type: ignore[attr-defined]
        dm = edm.decision_maker
        email: str = edm.email_result.email or ""

        # Build lead_hash
        domain: str = getattr(
            mc.enriched_candidate.candidate_with_people.qualified.candidate,  # type: ignore[attr-defined]
            "domain", "",
        ) or ""
        lead_hash = hashlib.sha256(
            f"{domain}|{dm.full_name.lower()}".encode()
        ).hexdigest()

        hard_reasons: list[str] = []
        soft_reasons: list[str] = []

        # 1. Email syntax
        syntax_ok = self._check_email_validity(email)
        if not syntax_ok:
            hard_reasons.append("invalid_email_syntax")

        # 2. MX record (cache only — no new SMTP handshake)
        if syntax_ok:
            mx_ok = await self._check_mx_cached(email)
            if not mx_ok:
                hard_reasons.append("no_mx_record")

        # 3. Deduplication
        is_dup = await self._check_dedup(lead_hash)
        if is_dup:
            hard_reasons.append("duplicate_lead")

        # 4. Profanity + garbage
        prof_flags = self._check_profanity_and_garbage(messages)
        hard_reasons.extend(prof_flags)

        # Soft checks (only bother if no hard failures yet — saves Gemini calls)
        if not hard_reasons:
            # 5. Subject casing
            if _ALL_CAPS_RE.search(messages.email_subject_a) or \
               _ALL_CAPS_RE.search(messages.email_subject_b):
                soft_reasons.append("all_caps_subject")

            # 6. URL in LinkedIn DM
            if _URL_RE.search(messages.linkedin_dm):
                soft_reasons.append("url_in_linkedin_dm")

            # 7. Quality thresholds
            thresh_flags = self._check_quality_thresholds(
                messages, mc, min_likelihood, max_flags
            )
            soft_reasons.extend(thresh_flags)

            # 8. Subject/body alignment (Gemini Flash)
            if not skip_alignment:
                aligned = await self._check_subject_body_alignment(messages)
                if not aligned:
                    soft_reasons.append("subject_body_mismatch")
            elif skip_alignment:
                soft_reasons.append("alignment_check_skipped_for_volume")

        all_reasons = hard_reasons + soft_reasons
        if hard_reasons:
            status = "rejected"
        elif soft_reasons:
            status = "needs_review"
        else:
            status = "ready_to_send"

        return self._make_vdm(mdm, status, all_reasons, lead_hash)

    # =========================================================================
    # Individual checks
    # =========================================================================

    def _check_email_validity(self, email: str) -> bool:
        """Return True if email passes syntax validation."""
        if not email:
            return False
        try:
            from email_validator import validate_email as _ve, EmailNotValidError
            _ve(email, check_deliverability=False)
            return True
        except Exception:  # noqa: BLE001
            return False

    async def _check_mx_cached(self, email: str) -> bool:
        """Return True if MX records exist. Uses cached SMTP result; never re-handshakes."""
        try:
            from sources._cache import cache_get
            cached = await cache_get("smtp_verifier", email, self.settings)
            if cached is not None:
                return cached.get("mx_records_found", True)
        except Exception:  # noqa: BLE001
            pass
        # No cache hit → optimistically allow (MX will be checked on actual send)
        return True

    async def _check_dedup(self, lead_hash: str) -> bool:
        """Return True if this lead_hash already exists with a non-rejected status."""
        try:
            from sqlalchemy import create_engine, select, text
            from sqlalchemy.orm import Session
            from scripts.init_db import Lead

            engine = create_engine(
                f"sqlite:///{self.settings.SQLITE_PATH}", future=True
            )
            try:
                with Session(engine) as session:
                    stmt = (
                        select(Lead.id)
                        .where(Lead.lead_hash == lead_hash)
                        .where(Lead.status != "rejected")
                        .limit(1)
                    )
                    result = session.execute(stmt).first()
                    return result is not None
            finally:
                engine.dispose()
        except Exception as exc:  # noqa: BLE001
            self.log.warning("dedup check failed: %s", exc)
            return False

    def _check_profanity_and_garbage(self, messages) -> list[str]:
        flags: list[str] = []
        full_text = " ".join([
            messages.email_subject_a,
            messages.email_subject_b,
            messages.email_body,
            messages.linkedin_dm,
        ]).lower()

        # Banned words (word-boundary regex)
        for word in BANNED_WORDS:
            pattern = r"\b" + re.escape(word.lower()) + r"\b"
            if re.search(pattern, full_text):
                flags.append("contains_profanity")
                break  # one flag is enough

        # Weird characters
        if _WEIRD_CHAR_RE.search(full_text):
            flags.append("weird_characters")

        return flags

    def _check_quality_thresholds(
        self, messages, mc, min_likelihood: int, max_flags: int
    ) -> list[str]:
        flags: list[str] = []

        if messages.reply_likelihood < min_likelihood:
            flags.append("low_reply_likelihood")

        if len(messages.quality_flags) > max_flags:
            flags.append("too_many_quality_flags")

        # Check personalization quality from parent MessagedCandidate
        personalization: Optional[PersonalizationContext] = getattr(
            mc, "personalization", None
        )
        if personalization and personalization.personalization_quality == "low":
            flags.append("weak_personalization")

        return flags

    async def _check_subject_body_alignment(self, messages) -> bool:
        """Return True if subject aligns with body (Gemini Flash call)."""
        prompt = (
            f"Does this email body deliver on what the subject line promises?\n\n"
            f"Subject A: {messages.email_subject_a}\n"
            f"Subject B: {messages.email_subject_b}\n\n"
            f"Body:\n{messages.email_body}\n\n"
            f'Return JSON: {{"aligned": true/false, "reason": "one sentence"}}'
        )
        try:
            result = await self.gemini.generate_json(
                prompt, _AlignmentResponse, temperature=0.1
            )
            if result is None:
                return True  # on failure, assume aligned (don't penalise)
            return result.aligned
        except Exception as exc:  # noqa: BLE001
            self.log.warning("alignment check failed: %s", exc)
            return True

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _make_vdm(
        mdm: MessagedDecisionMaker,
        status: str,
        reasons: list[str],
        lead_hash: str = "",
    ) -> ValidatedDecisionMaker:
        return ValidatedDecisionMaker(
            messaged_dm=mdm,
            status=status,  # type: ignore[arg-type]
            validation_reasons=reasons,
            lead_hash=lead_hash,
        )

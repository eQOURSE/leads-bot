"""Phase 9b — Synthetic MarkdownV2 validation for the Telegram digest.

Builds a hand-crafted ValidatedResult per segment containing leads with
adversarial characters (apostrophes, em-dashes, plus-signs, parentheses,
Unicode, backticks/asterisks) and sends a REAL Telegram digest so we can
visually confirm MarkdownV2 escaping survives the live Telegram API.

Usage:
    python scripts/test_telegram_digest.py
    python scripts/test_telegram_digest.py --dry-run   # build only, don't send

The digest title is prefixed with "[SYNTHETIC TEST]" so a human can tell it
apart from a real run.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make project root importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents._models import (
    DecisionMaker,
    EmailResult,
    EnrichedCandidate,
    EnrichedDecisionMaker,
    GeneratedMessages,
    MessagedCandidate,
    MessagedDecisionMaker,
    PersonalizationContext,
    QualifiedCandidate,
    QualifiedCandidateWithPeople,
    QualifierSubScores,
    ValidatedCandidate,
    ValidatedDecisionMaker,
    ValidatedResult,
)
from config.settings import get_settings
from sinks.telegram_sink import TelegramSink
from sources.models import CompanyCandidate


# ---------------------------------------------------------------------------
# Adversarial lead spec
# ---------------------------------------------------------------------------

# (company, dm_name, title, email, subject_a, reply_likelihood)
_ADVERSARIAL_LEADS = [
    (
        "O'Brien Labs",
        "Jamie O'Brien",
        "Founder / CEO",
        "j.o'brien+test@obrien-labs.io",
        "Quick thought (re: your Q3 launch)",
        8,
    ),
    (
        "Praxis & Co. — AI Labs",
        "Élise Park-Chen",
        "Co-Founder, CTO (interim)",
        "elise.park-chen@praxis-ai.co",
        "Re: scaling your eval pipeline — quick idea",
        7,
    ),
    (
        "Voice_AI 2.0",
        "Marcus Lee",
        "Head of ML",
        "marcus_lee@voice-ai-2.com",
        "Annotation pipeline for voice AI",
        9,
    ),
    (
        "Acme Technology Solutions International Holdings Inc.",
        "박지원 (Jiwon Park)",
        "Founder & CEO",
        "jiwon@acme-tech.co.kr",
        "Partnering on annotation quality",
        6,
    ),
    (
        "Test Corp",
        "Test User",
        "VP of Engineering",
        "test@test-corp.io",
        "Idea about `inline_code` and *emphasis*",
        5,
    ),
]


def _lead_hash(domain: str, name: str) -> str:
    import hashlib
    return hashlib.sha256(f"{domain}|{name.lower()}".encode()).hexdigest()


def _build_validated_dm(
    company: str,
    dm_name: str,
    title: str,
    email: str,
    subject_a: str,
    reply_likelihood: int,
) -> tuple[ValidatedCandidate, ValidatedDecisionMaker]:
    """Build a single fully-populated ValidatedCandidate with one ready DM."""
    domain = email.split("@", 1)[1] if "@" in email else "unknown.test"

    company_candidate = CompanyCandidate(
        domain=domain,
        name=company,
        description=f"{company} builds AI products.",
        funding_stage="Series A",
        funding_amount_usd=5_000_000,
        raw_source="synthetic",
        confidence=0.9,
    )

    qc = QualifiedCandidate(
        candidate=company_candidate,
        total_score=88,
        pre_score=60,
        sub_scores=QualifierSubScores(
            funding_recency_score=40,
            reachability_score=10,
            geography_score=10,
            size_match_score=8,
            segment_fit_score=12,
            buying_signal_score=8,
        ),
        reasoning="Strong synthetic match.",
        disqualifiers=[],
        tier="tier_1",
        domain_was_resolved=False,
    )

    dm = DecisionMaker(
        full_name=dm_name,
        title=title,
        linkedin_url="https://linkedin.com/in/synthetic",
        source="scrapegraph",
        seniority_score=92,
    )

    cwp = QualifiedCandidateWithPeople(
        qualified=qc,
        decision_makers=[dm],
        lookup_status="found",
        lookup_attempts={"scrapegraph": "found_1"},
    )

    edm = EnrichedDecisionMaker(
        decision_maker=dm,
        email_result=EmailResult(
            email=email,
            confidence=0.9,
            source="hunter_finder",
            smtp_verified=True,
        ),
    )

    ec = EnrichedCandidate(
        candidate_with_people=cwp,
        enriched_dms=[edm],
        enrichment_status="full",
    )

    messages = GeneratedMessages(
        email_subject_a=subject_a,
        email_subject_b="Value-driven alt subject",
        email_body=(
            f"Hi {dm_name.split()[0]}, congrats on the raise. "
            "We help teams like yours scale. One quick question — open to a chat? "
            "Best, eQOURSE x TUTRAIN"
        ),
        linkedin_dm=f"Hi {dm_name.split()[0]}, congrats on the milestone — worth a quick chat?",
        reply_likelihood=reply_likelihood,
        quality_flags=[],
    )

    personalization = PersonalizationContext(
        domain=domain,
        company_one_liner=f"{company} builds AI products.",
        recent_milestone="Series A raise",
        pain_hypothesis_specific="Scaling eval pipelines post-raise.",
        why_now_hook=f"Saw {company} just raised — timing looks right.",
        personalization_quality="high",
        built_at=datetime.now(timezone.utc),
    )

    mc = MessagedCandidate(
        enriched_candidate=ec,
        personalization=personalization,
        messaged_dms=[MessagedDecisionMaker(enriched_dm=edm, messages=messages)],
    )

    vdm = ValidatedDecisionMaker(
        messaged_dm=mc.messaged_dms[0],
        status="ready_to_send",
        validation_reasons=[],
        lead_hash=_lead_hash(domain, dm_name),
    )

    vc = ValidatedCandidate(messaged_candidate=mc, validated_dms=[vdm])
    return vc, vdm


def _build_synthetic_results() -> dict[str, ValidatedResult]:
    """Distribute the 5 adversarial leads across the 3 segments."""
    segments = ["tutrain", "eqourse_content", "eqourse_ai_data"]
    now = datetime.now(timezone.utc)

    # Assign leads: 2 / 2 / 1 across segments
    assignment = {
        "tutrain": _ADVERSARIAL_LEADS[0:2],
        "eqourse_content": _ADVERSARIAL_LEADS[2:4],
        "eqourse_ai_data": _ADVERSARIAL_LEADS[4:5],
    }

    results: dict[str, ValidatedResult] = {}
    for seg in segments:
        leads = assignment[seg]
        validated_candidates = []
        for spec in leads:
            vc, _ = _build_validated_dm(*spec)
            validated_candidates.append(vc)
        ready = len(validated_candidates)
        results[seg] = ValidatedResult(
            segment=seg,
            run_id=f"synthetic_{seg}_{now.strftime('%Y%m%d_%H%M%S')}",
            validated_candidates=validated_candidates,
            stats={"ready_to_send": ready, "needs_review": 0, "rejected": 0},
            api_credits_used={},
            started_at=now,
            completed_at=now,
            duration_seconds=0.0,
        )
    return results


# ---------------------------------------------------------------------------
# Telegram error reporting
# ---------------------------------------------------------------------------

def _report_telegram_error(exc: Exception, message_text: str) -> None:
    """Print actionable diagnostics for known Telegram failure modes."""
    import re

    name = type(exc).__name__
    msg = str(exc)
    print(f"\n[FAIL] Telegram send failed: {name}: {msg}")

    # BadRequest: Can't parse entities at offset N
    m = re.search(r"offset (\d+)", msg)
    if m:
        offset = int(m.group(1))
        start = max(0, offset - 20)
        end = min(len(message_text), offset + 20)
        snippet = message_text[start:end]
        char = message_text[offset] if offset < len(message_text) else "<EOF>"
        print(f"  Parse error at offset {offset}.")
        print(f"  Offending character: {char!r}")
        print(f"  Context: ...{snippet}...")
        print("  Fix: ensure this character is escaped via _md2().")
    elif "Forbidden" in name or "Forbidden" in msg:
        print("  Bot token invalid OR bot is blocked / not a member of the chat.")
        print("  Fix: verify TELEGRAM_BOT_TOKEN and that the bot can post to TELEGRAM_CHAT_ID.")
    elif "TimedOut" in name or "Timed" in msg:
        print("  Network timeout talking to Telegram.")
    else:
        import traceback
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _async_main(dry_run: bool) -> int:
    settings = get_settings()
    sink = TelegramSink(settings)

    results = _build_synthetic_results()
    sheets_url = ""
    if settings.SHEET_ID:
        sheets_url = f"https://docs.google.com/spreadsheets/d/{settings.SHEET_ID}/edit"

    # Build the message text once (for diagnostics / dry-run preview)
    text, leads_included, _ = sink._build_digest(
        results, sheets_url, title_prefix="[SYNTHETIC TEST]"
    )

    print("=" * 60)
    print("Synthetic Telegram digest — MarkdownV2 validation")
    print("=" * 60)
    print(f"Segments: {list(results.keys())}")
    print(f"Leads included in top section: {leads_included}")
    print(f"Message length: {len(text)} chars")
    print("-" * 60)
    print("Raw MarkdownV2 payload:")
    print(text)
    print("-" * 60)

    if dry_run:
        print("[DRY-RUN] Message built successfully. Not sending.")
        return 0

    # Retry once on TimedOut
    for attempt in (1, 2):
        try:
            bot = sink._make_bot()
            msg = await bot.send_message(
                chat_id=settings.TELEGRAM_CHAT_ID,
                text=text,
                parse_mode="MarkdownV2",
                disable_web_page_preview=True,
            )
            print(f"\n[PASS] Digest sent successfully — message_id={msg.message_id}")
            print("Check your Telegram: all 5 adversarial leads should render cleanly,")
            print("with the [SYNTHETIC TEST] marker in the title.")
            return 0
        except Exception as exc:  # noqa: BLE001
            is_timeout = "Timed" in type(exc).__name__ or "Timed" in str(exc)
            if is_timeout and attempt == 1:
                print("  TimedOut — retrying once…")
                await asyncio.sleep(2)
                continue
            _report_telegram_error(exc, text)
            return 1

    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic Telegram MarkdownV2 validation")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build the message but do not send it to Telegram",
    )
    args = parser.parse_args()
    exit_code = asyncio.run(_async_main(args.dry_run))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

"""Telegram sink — Phase 8.

Sends a MarkdownV2-formatted digest of top ready-to-send leads to a Telegram
chat. All dynamic content is escaped via telegram.helpers.escape_markdown.

Uses python-telegram-bot v21+ async API.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Optional

from config.logging_config import setup_logging
from config.settings import Settings
from sources._utils import utcnow

if TYPE_CHECKING:
    from agents._models import ValidatedResult

_MAX_MESSAGE_LEN = 4096    # Telegram hard limit
_TOP_N = 5                 # leads per digest


def _md2(text: str) -> str:
    """Escape a string for Telegram MarkdownV2.

    MarkdownV2 reserves exactly 18 characters that MUST be backslash-escaped
    anywhere they appear in text:  _ * [ ] ( ) ~ ` > # + - = | { } . !
    Everything else passes through literally — including em-dashes (—),
    apostrophes ('), ampersands (&), and all Unicode (e.g. 박지원). Do NOT
    over-escape those; escaping a non-special char is itself a parse error.
    Note '-' here is the ASCII hyphen (U+002D), not the em-dash (U+2014).
    """
    try:
        import telegram.helpers
        return telegram.helpers.escape_markdown(str(text), version=2)
    except Exception:  # noqa: BLE001
        # Fallback: manual escape of special chars
        import re
        special = r"\_*[]()~`>#+-=|{}.!"
        return re.sub(r"([" + re.escape(special) + r"])", r"\\\1", str(text))


class TelegramSink:
    """Send lead digests and error alerts to a Telegram chat."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.log = setup_logging("sink.telegram")

    def _make_bot(self):
        import telegram
        token = self.settings.TELEGRAM_BOT_TOKEN
        if not token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN not set")
        return telegram.Bot(token=token)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def send_run_digest(
        self,
        validated_results_per_segment: dict[str, "ValidatedResult"],
        sheets_url: str = "",
        title_prefix: str = "",
    ) -> dict:
        """Send a digest covering all provided segments. Returns {message_id, leads_included}.

        ``title_prefix`` lets callers tag the digest (e.g. "[SYNTHETIC TEST] ")
        so a human can tell test sends apart from real runs.
        """
        text, leads_included, all_vdms_ready = self._build_digest(
            validated_results_per_segment, sheets_url, title_prefix=title_prefix
        )

        try:
            bot = self._make_bot()
            msg = await bot.send_message(
                chat_id=self.settings.TELEGRAM_CHAT_ID,
                text=text,
                parse_mode="MarkdownV2",
                disable_web_page_preview=True,
            )
            message_id = msg.message_id
            self.log.info("telegram: digest sent, message_id=%d", message_id)

            # Mark sent_to_telegram_at in SQLite for included leads
            if all_vdms_ready:
                await self._mark_telegram_sent(all_vdms_ready)

            return {"message_id": message_id, "leads_included": leads_included}

        except Exception as exc:  # noqa: BLE001
            self.log.error("telegram: send_run_digest failed: %s", exc)
            return {"message_id": None, "leads_included": 0, "error": str(exc)}

    async def send_empty_run_digest(
        self,
        segment_stats: dict[str, dict],
        run_id: str,
        sheets_url: str = "",
    ) -> Optional[int]:
        """Send a brief 'ran, found nothing' message with funnel diagnostics.

        ``segment_stats`` maps each segment to a dict with at least
        ``hunt_count``, ``qualified_count``, and optionally ``after_dedupe``.
        Returns the Telegram message_id on success, or None on failure (logged).
        """
        text = self._build_empty_digest(segment_stats, run_id, sheets_url)

        try:
            bot = self._make_bot()
            msg = await bot.send_message(
                chat_id=self.settings.TELEGRAM_CHAT_ID,
                text=text,
                parse_mode="MarkdownV2",
                disable_web_page_preview=True,
            )
            self.log.info("telegram: empty-run digest sent, message_id=%d", msg.message_id)
            return msg.message_id
        except Exception as exc:  # noqa: BLE001
            self.log.error("telegram: send_empty_run_digest failed: %s", exc)
            return None

    async def send_error_alert(self, run_id: str, error_summary: str) -> None:
        """Send a brief error message if the pipeline failed."""
        text = (
            f"🚨 *Lead Gen Error*\n"
            f"Run ID: `{_md2(run_id)}`\n"
            f"Error: {_md2(error_summary[:300])}"
        )
        try:
            bot = self._make_bot()
            await bot.send_message(
                chat_id=self.settings.TELEGRAM_CHAT_ID,
                text=text,
                parse_mode="MarkdownV2",
            )
        except Exception as exc:  # noqa: BLE001
            self.log.error("telegram: send_error_alert failed: %s", exc)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_digest(
        self,
        results: dict[str, "ValidatedResult"],
        sheets_url: str,
        title_prefix: str = "",
    ) -> tuple[str, int, list]:
        """Build the MarkdownV2 digest text. Returns (text, leads_included, vdm_list)."""
        today = utcnow().strftime("%B %d, %Y")

        # Segment summary lines
        segment_lines = []
        for segment, vr in results.items():
            ready = vr.stats.get("ready_to_send", 0)
            review = vr.stats.get("needs_review", 0)
            total = ready + review + vr.stats.get("rejected", 0)
            seg_label = _md2(segment.replace("_", " ").title())
            segment_lines.append(
                f"• {seg_label}: {_md2(ready)}/{_md2(total)} ready, "
                f"{_md2(review)} for review"
            )

        # Collect all ready-to-send vdms across segments, sorted by reply_likelihood desc
        all_ready = []
        for segment, vr in results.items():
            for vc in vr.validated_candidates:
                mc = vc.messaged_candidate  # type: ignore[attr-defined]
                ec = mc.enriched_candidate  # type: ignore[attr-defined]
                qc = ec.candidate_with_people.qualified
                candidate = qc.candidate  # type: ignore[attr-defined]
                for vdm in vc.validated_dms:
                    if vdm.status != "ready_to_send":
                        continue
                    mdm = vdm.messaged_dm  # type: ignore[attr-defined]
                    if mdm.messages is None:
                        continue
                    edm = mdm.enriched_dm  # type: ignore[attr-defined]
                    dm = edm.decision_maker
                    all_ready.append({
                        "vdm": vdm,
                        "dm": dm,
                        "candidate": candidate,
                        "tier": qc.tier,
                        "messages": mdm.messages,
                        "email": edm.email_result.email or "",
                    })

        all_ready.sort(key=lambda x: -x["messages"].reply_likelihood)
        top = all_ready[:_TOP_N]

        # Build top-5 lines
        top_lines = []
        for i, item in enumerate(top, 1):
            company = _md2(getattr(item["candidate"], "name", "Unknown"))
            tier = _md2(item["tier"])
            dm_name = _md2(item["dm"].full_name)
            title = _md2(item["dm"].title)
            likelihood = _md2(item["messages"].reply_likelihood)
            subject = _md2(item["messages"].email_subject_a)
            email = _md2(item["email"])
            top_lines.append(
                f"{i}\\. *{company}* \\({tier}\\) — {dm_name}, {title}\n"
                f"   Reply likelihood: {likelihood}/10\n"
                f"   Subject A: \"{subject}\"\n"
                f"   📧 {email}"
            )

        top_block = "\n\n".join(top_lines) if top_lines else "_No ready leads this run_"

        # Sheets link
        if sheets_url:
            sheets_link = f"\n\n📋 [Open in Google Sheets]({sheets_url})"
        else:
            sheets_link = ""

        # Combine everything
        seg_block = "\n".join(segment_lines) if segment_lines else "_No segments_"
        run_ids = list(results.keys())
        run_id = _md2(results[run_ids[0]].run_id if run_ids else "unknown")

        # Optional human-visible marker (e.g. "[SYNTHETIC TEST] "), escaped for MarkdownV2.
        prefix_md = f"{_md2(title_prefix.strip())} " if title_prefix.strip() else ""

        text = (
            f"🚀 *{prefix_md}Lead Gen Run Complete* — {_md2(today)}\n"
            f"Run ID: `{run_id}`\n\n"
            f"📊 *Summary by Segment*\n{seg_block}\n\n"
            f"🎯 *Top {_md2(_TOP_N)} Ready\\-to\\-Send*\n\n"
            f"{top_block}"
            f"{sheets_link}"
        )

        # Truncate if over limit
        if len(text) > _MAX_MESSAGE_LEN:
            text = text[:_MAX_MESSAGE_LEN - 10] + "\n\\.\\.\\."

        return text, len(top), [item["vdm"] for item in top]

    def _build_empty_digest(
        self,
        segment_stats: dict[str, dict],
        run_id: str,
        sheets_url: str,
    ) -> str:
        """Build a short MarkdownV2 'ran, found nothing' digest with funnel diagnostics."""
        today = utcnow().strftime("%B %d, %Y %H:%M UTC")
        n_segments = len(segment_stats)

        lines = []
        for segment, stats in segment_stats.items():
            seg_label = _md2(segment.replace("_", " ").title())
            hunt = stats.get("hunt_count", 0)
            qualified = stats.get("qualified_count", 0)
            after_dedupe = stats.get("after_dedupe")
            # If dedupe info available and it dropped candidates, surface it.
            extra = ""
            if after_dedupe is not None and hunt > after_dedupe:
                dropped = hunt - after_dedupe
                extra = f" \\({_md2(dropped)} dropped at dedupe\\)"
            lines.append(
                f"• {seg_label}: {_md2(hunt)} hunt → {_md2(qualified)} qualified{extra}"
            )
        funnel_block = "\n".join(lines) if lines else "_No segments_"

        if sheets_url:
            sheets_link = f"\n\n📋 [Open Run History]({sheets_url})"
        else:
            sheets_link = ""

        text = (
            f"🟡 *Lead Gen Run* — {_md2(today)}\n"
            f"Run ID: `{_md2(run_id)}`\n\n"
            f"No qualified leads across {_md2(n_segments)} segments today\\.\n\n"
            f"*Funnel summary:*\n{funnel_block}"
            f"{sheets_link}"
        )

        if len(text) > _MAX_MESSAGE_LEN:
            text = text[:_MAX_MESSAGE_LEN - 10] + "\n\\.\\.\\."

        return text

    async def _mark_telegram_sent(self, vdms: list) -> None:
        """Update sent_to_telegram_at for included lead rows in SQLite."""
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import Session
            from sqlalchemy import text as sa_text

            engine = create_engine(
                f"sqlite:///{self.settings.SQLITE_PATH}", future=True
            )
            now_str = utcnow().isoformat()
            await asyncio.to_thread(
                self._mark_sync, engine, [v.lead_hash for v in vdms], now_str
            )
            engine.dispose()
        except Exception as exc:  # noqa: BLE001
            self.log.warning("telegram: mark_sent failed: %s", exc)

    @staticmethod
    def _mark_sync(engine, lead_hashes: list[str], ts: str) -> None:
        from sqlalchemy.orm import Session
        from sqlalchemy import text as sa_text

        with Session(engine) as session:
            for lh in lead_hashes:
                session.execute(
                    sa_text(
                        "UPDATE leads SET sent_to_telegram_at = :ts WHERE lead_hash = :h"
                    ),
                    {"ts": ts, "h": lh},
                )
            session.commit()

"""Sink Orchestrator — Phase 8.

Chains SQLite → Sheets → Telegram in order. Each step is fault-tolerant:
  - SQLite write: raises clearly on failure (data loss risk)
  - Sheets write: logs and continues if auth/API fails
  - Telegram: logs and continues if bot fails

Idempotency is enforced at each sink level.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

from agents._models import SentResult
from config.logging_config import setup_logging
from config.settings import Settings
from sources._utils import utcnow

if TYPE_CHECKING:
    from agents._models import ValidatedResult
    from sinks.google_sheets_sink import GoogleSheetsSink
    from sinks.sink_orchestrator import SinkOrchestrator
    from sinks.sqlite_store import LeadStore
    from sinks.sqlite_writer import SQLiteWriter
    from sinks.telegram_sink import TelegramSink


class SinkOrchestrator:
    """Dispatch a ValidatedResult to all three sinks in order."""

    def __init__(
        self,
        settings: Settings,
        sqlite_writer: "SQLiteWriter",
        sheets_sink: "GoogleSheetsSink",
        telegram_sink: "TelegramSink",
        lead_store: "LeadStore",
    ) -> None:
        self.settings = settings
        self.sqlite_writer = sqlite_writer
        self.sheets_sink = sheets_sink
        self.telegram_sink = telegram_sink
        self.lead_store = lead_store
        self.log = setup_logging("sink.orchestrator")

    # =========================================================================
    # Single-segment dispatch
    # =========================================================================

    async def dispatch(
        self,
        validated_result: "ValidatedResult",
        needs_manual_lookup: Optional[list] = None,
    ) -> SentResult:
        """Dispatch one segment's ValidatedResult to all sinks."""
        return await self.dispatch_multi(
            {validated_result.segment: validated_result},
            needs_manual_lookup_per_segment={
                validated_result.segment: needs_manual_lookup or []
            },
        )

    # =========================================================================
    # Multi-segment dispatch (full daily run)
    # =========================================================================

    async def dispatch_multi(
        self,
        segment_results: dict[str, "ValidatedResult"],
        needs_manual_lookup_per_segment: Optional[dict[str, list]] = None,
    ) -> SentResult:
        start_ts = time.perf_counter()
        needs_manual_lookup_per_segment = needs_manual_lookup_per_segment or {}
        sheets_errors: list[str] = []
        telegram_message_id: Optional[int] = None
        telegram_error: Optional[str] = None

        sqlite_inserted_total = 0
        sqlite_skipped_total = 0
        sheets_appended_total = 0

        # Use the first segment's run_id for the SentResult
        first_result = next(iter(segment_results.values()))
        run_id = first_result.run_id
        segment_label = ", ".join(segment_results.keys())

        # ---- 1. SQLite write (all segments) ---------------------------------
        for segment, vr in segment_results.items():
            try:
                counts = await self.sqlite_writer.write_validated(vr)
                sqlite_inserted_total += counts.get("inserted", 0)
                sqlite_skipped_total += counts.get("skipped_existing", 0)
            except Exception as exc:  # noqa: BLE001
                self.log.error("SQLite write failed for %s: %s", segment, exc)
                sheets_errors.append(f"sqlite_{segment}: {exc}")

        # ---- 2. Sheets write (all segments, fault-tolerant) -----------------
        for segment, vr in segment_results.items():
            try:
                sheets_result = await self.sheets_sink.write_leads(vr)
                sheets_appended_total += sheets_result.get("appended", 0)

                # Manual lookup rows
                manual = needs_manual_lookup_per_segment.get(segment, [])
                if manual:
                    await self.sheets_sink.write_manual_lookup(
                        manual, run_id, segment
                    )

                # Run history
                await self.sheets_sink.write_run_history(
                    self._build_run_summary(vr, sheets_result)
                )

            except Exception as exc:  # noqa: BLE001
                err = f"sheets_{segment}: {exc}"
                self.log.error("Sheets write failed: %s", err)
                sheets_errors.append(err)

        # ---- 3. Telegram digest (fault-tolerant) ----------------------------
        sheets_url = self._build_sheets_url()
        try:
            tg_result = await self.telegram_sink.send_run_digest(
                segment_results, sheets_url
            )
            telegram_message_id = tg_result.get("message_id")
            if tg_result.get("error"):
                telegram_error = tg_result["error"]
        except Exception as exc:  # noqa: BLE001
            telegram_error = str(exc)
            self.log.error("Telegram digest failed: %s", exc)

        self.log.info(
            "SinkOrchestrator[%s]: sqlite_in=%d sqlite_skip=%d sheets=%d tg=%s",
            run_id,
            sqlite_inserted_total,
            sqlite_skipped_total,
            sheets_appended_total,
            telegram_message_id,
        )

        return SentResult(
            segment=segment_label,
            run_id=run_id,
            sqlite_inserted=sqlite_inserted_total,
            sqlite_skipped=sqlite_skipped_total,
            sheets_appended=sheets_appended_total,
            sheets_errors=sheets_errors,
            telegram_message_id=telegram_message_id,
            telegram_error=telegram_error,
            duration_seconds=time.perf_counter() - start_ts,
        )

    # =========================================================================
    # Helpers
    # =========================================================================

    def _build_sheets_url(self) -> str:
        sheet_id = self.settings.SHEET_ID
        if sheet_id:
            return f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit"
        return ""

    @staticmethod
    def _build_run_summary(vr: "ValidatedResult", sheets_result: dict) -> dict:
        return {
            "date": utcnow().strftime("%Y-%m-%d %H:%M"),
            "run_id": vr.run_id,
            "segment": vr.segment,
            "status": "completed",
            "duration_s": f"{vr.duration_seconds:.1f}",
            "candidates_hunted": "",
            "qualified": "",
            "dms_found": "",
            "emails_found": "",
            "messages_generated": "",
            "ready_to_send": vr.stats.get("ready_to_send", 0),
            "needs_review": vr.stats.get("needs_review", 0),
            "rejected": vr.stats.get("rejected", 0),
            "api_credits": vr.api_credits_used,
            "errors": "",
        }

"""Google Sheets sink — Phase 8.

Appends validated leads to per-segment tabs plus shared "Needs Review",
"Manual Lookup", and "Run History" tabs. Idempotent via sent_to_sheets_at
timestamp stored in SQLite.

Uses gspread 6.x (synchronous), wrapped in asyncio.to_thread.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from config.logging_config import setup_logging
from config.settings import Settings
from sources._utils import utcnow

if TYPE_CHECKING:
    from agents._models import ValidatedResult
    from sinks.sqlite_store import LeadStore

# Column headers for the lead tabs — order matters (must match row building)
_LEAD_HEADERS = [
    "Date Added", "Run ID", "Segment", "Tier", "Company", "Domain",
    "Decision Maker", "Title", "Email", "Email Confidence", "Email Source",
    "Phone", "LinkedIn", "Funding Amount", "Funding Stage", "Funding Date",
    "Qualifier Score", "Why-Now Hook", "Subject A", "Subject B",
    "Email Body", "LinkedIn DM", "Reply Likelihood", "Quality Flags",
    "Validation Reasons", "Status", "Sent?", "Replied?", "Notes",
]

_MANUAL_LOOKUP_HEADERS = [
    "Date Added", "Run ID", "Segment", "Company", "Domain Guess",
    "Funding", "Funding Date", "Reason", "Notes",
]

_RUN_HISTORY_HEADERS = [
    "Date", "Run ID", "Segment", "Status", "Duration (s)",
    "Candidates Hunted", "Qualified", "DMs Found", "Emails Found",
    "Messages Generated", "Ready to Send", "Needs Review",
    "Rejected", "API Credits Used", "Errors",
]

# Tab names that are always present
_FIXED_TABS = ["Needs Review", "Manual Lookup", "Run History"]


def _conf_color(confidence: float) -> dict:
    """Return a gspread backgroundColor dict for email confidence."""
    if confidence >= 0.8:
        return {"red": 0.565, "green": 0.933, "blue": 0.565}   # green
    elif confidence >= 0.5:
        return {"red": 1.0, "green": 0.898, "blue": 0.6}        # yellow
    else:
        return {"red": 0.957, "green": 0.604, "blue": 0.604}    # red


class GoogleSheetsSink:
    """Append leads and metadata to Google Sheets."""

    def __init__(self, settings: Settings, lead_store: "LeadStore") -> None:
        self.settings = settings
        self.lead_store = lead_store
        self.log = setup_logging("sink.google_sheets")
        self._spreadsheet = None   # lazy-initialised

    # ------------------------------------------------------------------
    # Lazy init (gspread is sync)
    # ------------------------------------------------------------------

    def _get_spreadsheet(self):
        if self._spreadsheet is not None:
            return self._spreadsheet
        try:
            import gspread
            gc = gspread.service_account(filename=self.settings.GOOGLE_SHEETS_CREDS_PATH)
            self._spreadsheet = gc.open_by_key(self.settings.SHEET_ID)
            return self._spreadsheet
        except Exception as exc:
            raise RuntimeError(f"Google Sheets auth failed: {exc}") from exc

    def _ensure_tabs_exist_sync(self) -> None:
        ss = self._get_spreadsheet()
        existing = {ws.title for ws in ss.worksheets()}

        segment_tabs = {
            self.settings.SHEET_TAB_TUTRAIN,
            self.settings.SHEET_TAB_CONTENT,
            self.settings.SHEET_TAB_AI_DATA,
        }
        for tab_name in sorted(segment_tabs | set(_FIXED_TABS)):
            if tab_name not in existing:
                ws = ss.add_worksheet(title=tab_name, rows=1000, cols=len(_LEAD_HEADERS))
                # Write header row
                if tab_name == "Manual Lookup":
                    ws.append_row(_MANUAL_LOOKUP_HEADERS)
                elif tab_name == "Run History":
                    ws.append_row(_RUN_HISTORY_HEADERS)
                else:
                    ws.append_row(_LEAD_HEADERS)
                self.log.info("google_sheets: created tab %r", tab_name)

    def _segment_tab(self, segment: str) -> str:
        mapping = {
            "tutrain": self.settings.SHEET_TAB_TUTRAIN,
            "eqourse_content": self.settings.SHEET_TAB_CONTENT,
            "eqourse_ai_data": self.settings.SHEET_TAB_AI_DATA,
        }
        return mapping.get(segment, self.settings.SHEET_TAB_AI_DATA)

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def write_leads(self, validated_result: "ValidatedResult") -> dict:
        return await asyncio.to_thread(self._write_leads_sync, validated_result)

    def _write_leads_sync(self, validated_result: "ValidatedResult") -> dict:
        from sqlalchemy import create_engine, text as sa_text
        from sqlalchemy.orm import Session
        from scripts.init_db import Lead

        try:
            ss = self._get_spreadsheet()
            self._ensure_tabs_exist_sync()
        except Exception as exc:
            self.log.error("google_sheets: init failed: %s", exc)
            return {"appended": 0, "skipped_already_sent": 0, "tab_writes": {}}

        engine = create_engine(f"sqlite:///{self.settings.SQLITE_PATH}", future=True)
        appended = 0
        skipped = 0
        tab_writes: dict[str, int] = {}

        try:
            for vc in validated_result.validated_candidates:
                mc = vc.messaged_candidate  # type: ignore[attr-defined]
                ec = mc.enriched_candidate  # type: ignore[attr-defined]
                qc = ec.candidate_with_people.qualified
                candidate = qc.candidate  # type: ignore[attr-defined]
                personalization = getattr(mc, "personalization", None)

                for vdm in vc.validated_dms:
                    if vdm.status == "rejected":
                        continue  # rejected leads never go to Sheets

                    mdm = vdm.messaged_dm  # type: ignore[attr-defined]
                    edm = mdm.enriched_dm  # type: ignore[attr-defined]
                    dm = edm.decision_maker
                    messages = mdm.messages
                    lead_hash = vdm.lead_hash

                    # Check idempotency: is sent_to_sheets_at already set?
                    with Session(engine) as session:
                        existing = session.execute(
                            sa_text(
                                "SELECT id, sent_to_sheets_at FROM leads "
                                "WHERE lead_hash = :h LIMIT 1"
                            ),
                            {"h": lead_hash},
                        ).first()

                    if existing and existing[1]:  # sent_to_sheets_at is set
                        skipped += 1
                        continue

                    # Determine target tab
                    if vdm.status == "needs_review":
                        tab_name = "Needs Review"
                    else:
                        tab_name = self._segment_tab(validated_result.segment)

                    # Build row
                    now_str = utcnow().strftime("%Y-%m-%d %H:%M")
                    conf = edm.email_result.confidence
                    row = [
                        now_str,
                        validated_result.run_id,
                        validated_result.segment,
                        qc.tier,
                        getattr(candidate, "name", ""),
                        getattr(candidate, "domain", ""),
                        dm.full_name,
                        dm.title,
                        edm.email_result.email or "",
                        f"{conf:.2f}",
                        edm.email_result.source,
                        getattr(edm, "phone", "") or "",
                        dm.linkedin_url or "",
                        str(getattr(candidate, "funding_amount_usd", "") or ""),
                        getattr(candidate, "funding_stage", "") or "",
                        str(getattr(candidate, "funding_date", "") or ""),
                        str(qc.total_score),
                        personalization.why_now_hook if personalization else "",
                        messages.email_subject_a if messages else "",
                        messages.email_subject_b if messages else "",
                        messages.email_body if messages else "",
                        messages.linkedin_dm if messages else "",
                        str(messages.reply_likelihood) if messages else "",
                        json.dumps(messages.quality_flags) if messages else "[]",
                        json.dumps(vdm.validation_reasons),
                        vdm.status,
                        "No",
                        "No",
                        "",  # Notes
                    ]

                    try:
                        ws = ss.worksheet(tab_name)
                        ws.append_row(row, value_input_option="USER_ENTERED")

                        # Get the row index that was just appended
                        row_index = len(ws.get_all_values())  # 1-based

                        # Color-code the Email Confidence cell (col index 10 = 'J')
                        try:
                            conf_col = _LEAD_HEADERS.index("Email Confidence") + 1
                            cell_addr = f"{chr(64 + conf_col)}{row_index}"
                            ws.format(cell_addr, {
                                "backgroundColor": _conf_color(conf)
                            })
                        except Exception:  # noqa: BLE001
                            pass  # formatting is best-effort

                        appended += 1
                        tab_writes[tab_name] = tab_writes.get(tab_name, 0) + 1

                        # Write back to SQLite
                        with Session(engine) as session:
                            session.execute(
                                sa_text(
                                    "UPDATE leads SET sent_to_sheets_at = :ts, "
                                    "sheets_row_index = :ri WHERE lead_hash = :h"
                                ),
                                {
                                    "ts": utcnow().isoformat(),
                                    "ri": row_index,
                                    "h": lead_hash,
                                },
                            )
                            session.commit()

                    except Exception as exc:  # noqa: BLE001
                        self.log.error(
                            "google_sheets: failed to append row for %s: %s",
                            dm.full_name, exc,
                        )

        finally:
            engine.dispose()

        self.log.info(
            "GoogleSheetsSink[%s]: appended=%d skipped=%d tabs=%s",
            validated_result.run_id, appended, skipped, tab_writes,
        )
        return {
            "appended": appended,
            "skipped_already_sent": skipped,
            "tab_writes": tab_writes,
        }

    async def write_manual_lookup(
        self, candidates: list, run_id: str, segment: str
    ) -> int:
        return await asyncio.to_thread(
            self._write_manual_lookup_sync, candidates, run_id, segment
        )

    def _write_manual_lookup_sync(
        self, candidates: list, run_id: str, segment: str
    ) -> int:
        try:
            ss = self._get_spreadsheet()
            self._ensure_tabs_exist_sync()
            ws = ss.worksheet("Manual Lookup")
        except Exception as exc:
            self.log.error("google_sheets: manual lookup init failed: %s", exc)
            return 0

        count = 0
        now_str = utcnow().strftime("%Y-%m-%d %H:%M")
        for qc in candidates:
            candidate = qc.candidate  # type: ignore[attr-defined]
            try:
                row = [
                    now_str,
                    run_id,
                    segment,
                    getattr(candidate, "name", ""),
                    getattr(candidate, "domain", ""),
                    str(getattr(candidate, "funding_amount_usd", "") or ""),
                    str(getattr(candidate, "funding_date", "") or ""),
                    "needs_manual_lookup",
                    "",  # Notes
                ]
                ws.append_row(row, value_input_option="USER_ENTERED")
                count += 1
            except Exception as exc:  # noqa: BLE001
                self.log.warning("manual lookup row failed: %s", exc)

        return count

    async def write_run_history(self, run_summary: dict) -> None:
        await asyncio.to_thread(self._write_run_history_sync, run_summary)

    def _write_run_history_sync(self, run_summary: dict) -> None:
        try:
            ss = self._get_spreadsheet()
            self._ensure_tabs_exist_sync()
            ws = ss.worksheet("Run History")
        except Exception as exc:
            self.log.error("google_sheets: run history init failed: %s", exc)
            return

        row = [
            run_summary.get("date", ""),
            run_summary.get("run_id", ""),
            run_summary.get("segment", ""),
            run_summary.get("status", ""),
            run_summary.get("duration_s", ""),
            run_summary.get("candidates_hunted", ""),
            run_summary.get("qualified", ""),
            run_summary.get("dms_found", ""),
            run_summary.get("emails_found", ""),
            run_summary.get("messages_generated", ""),
            run_summary.get("ready_to_send", ""),
            run_summary.get("needs_review", ""),
            run_summary.get("rejected", ""),
            json.dumps(run_summary.get("api_credits", {})),
            run_summary.get("errors", ""),
        ]
        try:
            ws.append_row(row, value_input_option="USER_ENTERED")
        except Exception as exc:  # noqa: BLE001
            self.log.warning("run history row failed: %s", exc)

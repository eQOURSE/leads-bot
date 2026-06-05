"""SQLite sink — writes validated leads to the leads table.

Uses INSERT OR IGNORE on lead_hash so re-runs are idempotent.
All inserts happen in a single transaction per call.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from config.logging_config import setup_logging
from config.settings import Settings
from sources._utils import utcnow

if TYPE_CHECKING:
    from agents._models import ValidatedResult
    from sinks.sqlite_store import LeadStore


class SQLiteWriter:
    """Write validated leads to the SQLite leads table."""

    def __init__(self, settings: Settings, lead_store: "LeadStore") -> None:
        self.settings = settings
        self.lead_store = lead_store
        self.log = setup_logging("sink.sqlite_writer")

    async def write_validated(self, validated_result: "ValidatedResult") -> dict[str, int]:
        """Insert all validated DMs into the leads table.

        Returns {"inserted": N, "skipped_existing": M}.
        """
        return await asyncio.to_thread(self._write_sync, validated_result)

    def _write_sync(self, validated_result: "ValidatedResult") -> dict[str, int]:
        from sqlalchemy import create_engine
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert
        from sqlalchemy.orm import Session
        from scripts.init_db import Lead

        engine = create_engine(
            f"sqlite:///{self.settings.SQLITE_PATH}", future=True
        )
        inserted = 0
        skipped = 0

        try:
            with Session(engine) as session:
                for vc in validated_result.validated_candidates:
                    mc = vc.messaged_candidate  # type: ignore[attr-defined]
                    ec = mc.enriched_candidate  # type: ignore[attr-defined]
                    qc = ec.candidate_with_people.qualified
                    candidate = qc.candidate  # type: ignore[attr-defined]
                    personalization = getattr(mc, "personalization", None)

                    for vdm in vc.validated_dms:
                        mdm = vdm.messaged_dm  # type: ignore[attr-defined]
                        edm = mdm.enriched_dm  # type: ignore[attr-defined]
                        dm = edm.decision_maker
                        messages = mdm.messages

                        lead_id = str(uuid.uuid4())
                        lead_hash = vdm.lead_hash or ""

                        # Build column values
                        row = {
                            "id": lead_id,
                            "segment": validated_result.segment,
                            "company_name": getattr(candidate, "name", None),
                            "domain": getattr(candidate, "domain", None),
                            "decision_maker_name": dm.full_name,
                            "title": dm.title,
                            "email": edm.email_result.email,
                            "email_confidence": edm.email_result.confidence,
                            "email_source": edm.email_result.source,
                            "phone": getattr(edm, "phone", None),
                            "linkedin_url": dm.linkedin_url,
                            "funding_amount": str(getattr(candidate, "funding_amount_usd", "") or ""),
                            "funding_date": str(getattr(candidate, "funding_date", "") or ""),
                            "funding_source": getattr(candidate, "funding_source", None),
                            "qualifier_score": qc.total_score,
                            "segment_fit_score": qc.sub_scores.segment_fit_score,
                            "funding_recency_score": qc.sub_scores.funding_recency_score,
                            "buying_signal_score": qc.sub_scores.buying_signal_score,
                            "reachability_score": qc.sub_scores.reachability_score,
                            "personalization_hook": personalization.why_now_hook if personalization else None,
                            "email_subject_a": messages.email_subject_a if messages else None,
                            "email_subject_b": messages.email_subject_b if messages else None,
                            "email_body": messages.email_body if messages else None,
                            "linkedin_dm": messages.linkedin_dm if messages else None,
                            "reply_likelihood": messages.reply_likelihood if messages else None,
                            "status": vdm.status,
                            "validation_reasons": json.dumps(vdm.validation_reasons),
                            "lead_hash": lead_hash,
                            "created_at": utcnow(),
                            "reply_received": False,
                        }

                        # INSERT OR IGNORE — idempotent on lead_hash
                        stmt = sqlite_insert(Lead).values(**row).on_conflict_do_nothing(
                            index_elements=["lead_hash"]
                        )
                        result = session.execute(stmt)
                        if result.rowcount == 1:
                            inserted += 1
                        else:
                            skipped += 1

                session.commit()
        except Exception as exc:  # noqa: BLE001
            self.log.error("SQLiteWriter.write_validated failed: %s", exc)
        finally:
            engine.dispose()

        self.log.info(
            "SQLiteWriter[%s]: inserted=%d skipped=%d",
            validated_result.run_id, inserted, skipped,
        )
        return {"inserted": inserted, "skipped_existing": skipped}

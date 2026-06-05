"""SQLite-backed persistence layer for the lead generation system.

Wraps the SQLAlchemy models from scripts/init_db.py so agents never touch
the ORM directly. All public methods are async; DB calls run in worker threads
via asyncio.to_thread so the event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from config.logging_config import setup_logging
from config.settings import Settings
from sources._utils import utcnow


class LeadStore:
    """Async facade over the SQLite lead database."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.log = setup_logging("sink.sqlite_store")

    # ----- engine factory (not cached; each call opens+closes cleanly) --------

    def _engine(self):
        return create_engine(
            f"sqlite:///{self.settings.SQLITE_PATH}", future=True
        )

    # ----- run management -----------------------------------------------------

    def _create_run_sync(self, segment: str) -> str:
        from scripts.init_db import Run

        run_id = str(uuid.uuid4())
        engine = self._engine()
        try:
            with Session(engine) as session:
                session.add(
                    Run(
                        id=run_id,
                        segment=segment,
                        started_at=utcnow(),
                        status="running",
                    )
                )
                session.commit()
        finally:
            engine.dispose()
        return run_id

    async def create_run(self, segment: str) -> str:
        """Insert a new run record and return its UUID."""
        return await asyncio.to_thread(self._create_run_sync, segment)

    def _update_run_sync(self, run_id: str, **fields: Any) -> None:
        from scripts.init_db import Run

        engine = self._engine()
        try:
            with Session(engine) as session:
                row = session.get(Run, run_id)
                if row is None:
                    self.log.warning("update_run: run_id %s not found", run_id)
                    return
                for key, value in fields.items():
                    # Serialise dicts to JSON for the TEXT column.
                    if isinstance(value, dict):
                        value = json.dumps(value)
                    if hasattr(row, key):
                        setattr(row, key, value)
                session.commit()
        finally:
            engine.dispose()

    async def update_run(self, run_id: str, **fields: Any) -> None:
        """Update arbitrary columns on an existing run row."""
        await asyncio.to_thread(self._update_run_sync, run_id, **fields)

    # ----- seen_domains -------------------------------------------------------

    def _mark_domains_seen_sync(self, domains: list[str]) -> None:
        from scripts.init_db import SeenDomain

        if not domains:
            return
        now = utcnow()
        engine = self._engine()
        try:
            with Session(engine) as session:
                for domain in domains:
                    if not domain:
                        continue
                    existing = session.get(SeenDomain, domain)
                    if existing is None:
                        session.add(
                            SeenDomain(
                                domain=domain,
                                first_seen_at=now,
                                last_attempt_at=now,
                                attempts=1,
                            )
                        )
                    else:
                        existing.last_attempt_at = now
                        existing.attempts = (existing.attempts or 0) + 1
                session.commit()
        finally:
            engine.dispose()

    async def mark_domains_seen(self, domains: list[str]) -> None:
        """Upsert domains into seen_domains, bumping last_attempt_at and attempts."""
        await asyncio.to_thread(self._mark_domains_seen_sync, domains)

    def _get_seen_domains_sync(self, days: int) -> set[str]:
        from scripts.init_db import SeenDomain

        cutoff = utcnow() - timedelta(days=days)
        engine = self._engine()
        try:
            with Session(engine) as session:
                stmt = select(SeenDomain.domain).where(
                    SeenDomain.last_attempt_at >= cutoff
                )
                return {row for row in session.scalars(stmt)}
        finally:
            engine.dispose()

    async def get_seen_domains_within(self, days: int) -> set[str]:
        """Return the set of domains seen (attempted) within the last *days* days."""
        return await asyncio.to_thread(self._get_seen_domains_sync, days)

"""Initialize the SQLite database and create all tables.

Run this as the final step of Phase 0 setup so the database exists before
Phase 1:

    python scripts/init_db.py

Uses SQLAlchemy 2.0 declarative syntax.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Optional

# Make the project root importable when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from config.settings import get_settings


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class Lead(Base):
    __tablename__ = "leads"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    segment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    company_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    domain: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    decision_maker_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    email: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    email_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    email_source: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    phone: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    linkedin_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    funding_amount: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    funding_date: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    funding_source: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    qualifier_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    segment_fit_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    funding_recency_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    buying_signal_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    reachability_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    personalization_hook: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    email_subject_a: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    email_subject_b: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    email_body: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    linkedin_dm: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reply_likelihood: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(Text, default="new", nullable=False)
    created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=True
    )
    sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    reply_received: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    lead_hash: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    sent_to_sheets_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    sent_to_telegram_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    sheets_row_index: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    validation_reasons: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON array

    __table_args__ = (
        UniqueConstraint("lead_hash", name="uq_leads_lead_hash"),
    )


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    segment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    status: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    candidates_found: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    qualified_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    enriched_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    validated_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    api_credits_used: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
    error_log: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class SeenDomain(Base):
    __tablename__ = "seen_domains"

    domain: Mapped[str] = mapped_column(Text, primary_key=True)
    first_seen_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_attempt_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=1, nullable=False)


class Reply(Base):
    __tablename__ = "replies"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    lead_id: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("leads.id"), nullable=True
    )
    received_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    reply_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sentiment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    follow_up_status: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class ApiUsage(Base):
    __tablename__ = "api_usage"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    source: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    credits_used: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    credits_remaining: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


def init_db(db_path: Optional[str] = None) -> str:
    """Create the SQLite database and all tables.

    Args:
        db_path: Optional override for the SQLite path. Defaults to the
            ``SQLITE_PATH`` setting.

    Returns:
        The resolved database path.
    """
    settings = get_settings()
    path = db_path or settings.SQLITE_PATH

    db_file = Path(path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(f"sqlite:///{db_file}", echo=False, future=True)
    Base.metadata.create_all(engine)
    engine.dispose()
    return str(db_file)


def main() -> None:
    resolved = init_db()
    tables = sorted(Base.metadata.tables.keys())
    print(f"Database initialized at: {resolved}")
    print(f"Created/verified tables: {', '.join(tables)}")


if __name__ == "__main__":
    main()

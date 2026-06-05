"""Shared utilities for the data source layer.

- ``track_usage``    : record API consumption in the ``api_usage`` table.
- ``make_lead_hash`` : deterministic sha256 of domain + name.
- ``normalize_domain``: reduce a URL or messy domain to a bare registrable domain.
- ``month_call_count``: count rows in ``api_usage`` for a source this month.
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from datetime import date, datetime
from typing import Optional

import tldextract
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from config.settings import Settings, get_settings

# tldextract instance that avoids live suffix-list fetches at runtime
# (uses the bundled snapshot); keeps tests offline.
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())


def make_lead_hash(domain: str, name: str) -> str:
    """Return a stable sha256 hex digest of ``domain`` + ``name``.

    Inputs are normalized (lowercased, stripped) so trivial variations map to
    the same hash.
    """
    norm_domain = normalize_domain(domain) if domain else ""
    norm_name = (name or "").strip().lower()
    payload = f"{norm_domain}|{norm_name}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_domain(url_or_domain: str) -> str:
    """Strip protocol, ``www``, paths and ports — return the bare domain.

    Uses tldextract so subdomains collapse to the registrable domain
    (e.g. ``https://blog.acme.co.uk/x`` -> ``acme.co.uk``).
    """
    if not url_or_domain:
        return ""
    value = url_or_domain.strip().lower()
    extracted = _EXTRACT(value)
    if extracted.domain and extracted.suffix:
        return f"{extracted.domain}.{extracted.suffix}"
    # Fallback: best-effort manual strip when tldextract can't parse.
    value = value.split("://")[-1]
    if value.startswith("www."):
        value = value[4:]
    return value.split("/")[0].split(":")[0]


def _usage_engine(settings: Optional[Settings] = None):
    settings = settings or get_settings()
    return create_engine(f"sqlite:///{settings.SQLITE_PATH}", future=True)


def _track_usage_sync(
    source: str,
    credits: int,
    remaining: Optional[int],
    settings: Optional[Settings] = None,
) -> None:
    """Synchronous implementation of usage tracking (runs in a worker thread)."""
    from scripts.init_db import ApiUsage  # local import avoids cycles

    engine = _usage_engine(settings)
    try:
        with Session(engine) as session:
            row = ApiUsage(
                id=str(uuid.uuid4()),
                source=source,
                date=date.today(),
                credits_used=credits,
                credits_remaining=remaining,
            )
            session.add(row)
            session.commit()
    finally:
        engine.dispose()


async def track_usage(
    source: str,
    credits: int,
    remaining: Optional[int],
    settings: Optional[Settings] = None,
) -> None:
    """Write a usage row to the ``api_usage`` table.

    Executed in a thread so the synchronous SQLAlchemy call does not block the
    event loop.
    """
    await asyncio.to_thread(_track_usage_sync, source, credits, remaining, settings)


def _month_call_count_sync(source: str, settings: Optional[Settings] = None) -> int:
    from scripts.init_db import ApiUsage

    engine = _usage_engine(settings)
    today = date.today()
    month_start = today.replace(day=1)
    try:
        with Session(engine) as session:
            stmt = (
                select(func.count())
                .select_from(ApiUsage)
                .where(ApiUsage.source == source)
                .where(ApiUsage.date >= month_start)
            )
            return int(session.execute(stmt).scalar_one())
    finally:
        engine.dispose()


async def month_call_count(source: str, settings: Optional[Settings] = None) -> int:
    """Return the number of ``api_usage`` rows for ``source`` in the current month."""
    return await asyncio.to_thread(_month_call_count_sync, source, settings)


def utcnow() -> datetime:
    """Timezone-naive UTC now (matches the DB's naive timestamps)."""
    return datetime.utcnow()

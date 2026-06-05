"""Lightweight SQLite-backed response cache for source clients.

Creates its own ``source_cache`` table on demand in the project SQLite DB so it
does not require changes to the Phase 0 schema. Entries are keyed by
``(method, cache_key)`` and expire after a TTL.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import (
    DateTime,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from config.settings import Settings, get_settings


class _CacheBase(DeclarativeBase):
    pass


class SourceCache(_CacheBase):
    __tablename__ = "source_cache"

    method: Mapped[str] = mapped_column(String, primary_key=True)
    cache_key: Mapped[str] = mapped_column(String, primary_key=True)
    payload: Mapped[str] = mapped_column(Text)
    expires_at: Mapped[datetime] = mapped_column(DateTime)


def _engine(settings: Optional[Settings] = None):
    settings = settings or get_settings()
    engine = create_engine(f"sqlite:///{settings.SQLITE_PATH}", future=True)
    try:
        _CacheBase.metadata.create_all(engine, checkfirst=True)
    except Exception:  # noqa: BLE001 — table may already exist (concurrent threads)
        pass
    return engine


def _get_sync(method: str, key: str, settings: Optional[Settings]) -> Optional[dict]:
    engine = _engine(settings)
    try:
        with Session(engine) as session:
            row = session.get(SourceCache, (method, key))
            if row is None:
                return None
            if row.expires_at < datetime.utcnow():
                session.delete(row)
                session.commit()
                return None
            return json.loads(row.payload)
    finally:
        engine.dispose()


def _set_sync(
    method: str,
    key: str,
    value: dict,
    ttl_days: int,
    settings: Optional[Settings],
) -> None:
    engine = _engine(settings)
    expires = datetime.utcnow() + timedelta(days=ttl_days)
    try:
        with Session(engine) as session:
            row = session.get(SourceCache, (method, key))
            if row is None:
                row = SourceCache(method=method, cache_key=key)
                session.add(row)
            row.payload = json.dumps(value)
            row.expires_at = expires
            session.commit()
    finally:
        engine.dispose()


async def cache_get(
    method: str, key: str, settings: Optional[Settings] = None
) -> Optional[dict]:
    return await asyncio.to_thread(_get_sync, method, key, settings)


async def cache_set(
    method: str,
    key: str,
    value: dict,
    ttl_days: int,
    settings: Optional[Settings] = None,
) -> None:
    await asyncio.to_thread(_set_sync, method, key, value, ttl_days, settings)

"""Shared pytest fixtures for the source-client test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import Settings
from scripts.init_db import init_db


@pytest.fixture
def test_settings(tmp_path, monkeypatch) -> Settings:
    """A Settings instance pointing at an isolated temp SQLite DB.

    All Tier-1 keys are populated with dummy values and base URLs use example
    hosts so respx can match them deterministically.
    """
    db_path = tmp_path / "leads.db"
    init_db(str(db_path))

    # Avoid leaking real environment / .env values into tests.
    for key in (
        "GEMINI_API_KEY",
        "VIBE_PROSPECTING_API_KEY",
        "HUNTER_API_KEY",
        "SCRAPEGRAPH_API_KEY",
        "SERPAPI_KEY",
        "NEWSDATA_API_KEY",
        "COMPANIES_API_TOKEN",
        "APIFY_TOKEN_1",
        "APIFY_TOKEN_2",
        "APIFY_TOKEN_3",
        "APIFY_TOKEN_4",
    ):
        monkeypatch.delenv(key, raising=False)

    return Settings(
        _env_file=None,
        GEMINI_API_KEY="test-gemini-key",
        VIBE_PROSPECTING_API_KEY="test-vibe-key",
        HUNTER_API_KEY="test-hunter-key",
        SCRAPEGRAPH_API_KEY="test-sgai-key",
        SERPAPI_KEY="test-serpapi-key",
        NEWSDATA_API_KEY="test-newsdata-key",
        COMPANIES_API_TOKEN="test-companies-token",
        APIFY_TOKEN_1="test-apify-1",
        APIFY_TOKEN_2="test-apify-2",
        APIFY_TOKEN_3="test-apify-3",
        APIFY_TOKEN_4="test-apify-4",
        VIBE_PROSPECTING_BASE_URL="https://api.vibeprospecting.test/v1",
        HUNTER_BASE_URL="https://api.hunter.test/v2",
        SCRAPEGRAPH_BASE_URL="https://api.scrapegraph.test/v1",
        APIFY_BASE_URL="https://api.apify.test/v2",
        SERPAPI_BASE_URL="https://serpapi.test",
        NEWSDATA_BASE_URL="https://newsdata.test/api/1",
        COMPANIES_API_BASE_URL="https://api.companies.test/v2",
        SQLITE_PATH=str(db_path),
    )


def count_usage(settings: Settings, source: str) -> int:
    """Helper: count api_usage rows for a source in the test DB."""
    from sqlalchemy import create_engine, func, select
    from sqlalchemy.orm import Session

    from scripts.init_db import ApiUsage

    engine = create_engine(f"sqlite:///{settings.SQLITE_PATH}", future=True)
    try:
        with Session(engine) as session:
            stmt = (
                select(func.count())
                .select_from(ApiUsage)
                .where(ApiUsage.source == source)
            )
            return int(session.execute(stmt).scalar_one())
    finally:
        engine.dispose()

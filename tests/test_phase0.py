"""Phase 0 tests — none of these require external API access."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from sqlalchemy import create_engine, inspect

# Make the project root importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.logging_config import setup_logging
from config.settings import Settings
from scripts.init_db import init_db

_EXPECTED_TABLES = {"leads", "runs", "seen_domains", "replies", "api_usage"}


def _make_settings(env_file, monkeypatch, values: dict) -> Settings:
    """Build a Settings instance from an explicit env file, isolated from the
    real process environment."""
    # Ensure stray environment variables don't leak into the test.
    for key in values:
        monkeypatch.delenv(key, raising=False)
    env_file.write_text(
        "\n".join(f"{k}={v}" for k, v in values.items()),
        encoding="utf-8",
    )
    return Settings(_env_file=str(env_file))


def test_settings_loads_from_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    settings = _make_settings(
        env_file,
        monkeypatch,
        {
            "GCP_PROJECT_ID": "my-project",
            "HUNTER_API_KEY": "hunter-123",
            "DAILY_LEAD_TARGET_PER_SEGMENT": "20",
            "LOG_LEVEL": "DEBUG",
        },
    )

    assert settings.GCP_PROJECT_ID == "my-project"
    assert settings.HUNTER_API_KEY == "hunter-123"
    # Typed coercion from string to int.
    assert settings.DAILY_LEAD_TARGET_PER_SEGMENT == 20
    assert settings.LOG_LEVEL == "DEBUG"
    # Default values fill in for unset fields.
    assert settings.GCP_REGION == "us-central1"


def test_logging_writes_to_file(tmp_path, monkeypatch):
    # Run inside a temp dir so the logs/ folder is created there.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LOG_LEVEL", "INFO")

    logger = setup_logging("phase0test")
    logger.info("hello phase 0")

    for handler in logger.handlers:
        handler.flush()

    log_files = list((tmp_path / "logs").glob("phase0test_*.log"))
    assert len(log_files) == 1

    contents = log_files[0].read_text(encoding="utf-8")
    assert "hello phase 0" in contents
    assert "INFO" in contents

    # urllib3 / httpx should be pinned to WARNING.
    assert logging.getLogger("urllib3").level == logging.WARNING
    assert logging.getLogger("httpx").level == logging.WARNING


def test_db_schema_has_all_tables(tmp_path):
    db_path = tmp_path / "leads.db"
    resolved = init_db(str(db_path))

    assert Path(resolved).is_file()

    engine = create_engine(f"sqlite:///{resolved}", future=True)
    try:
        tables = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()

    assert _EXPECTED_TABLES.issubset(tables)


def test_apify_tokens_property_filters_empty(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    settings = _make_settings(
        env_file,
        monkeypatch,
        {
            "APIFY_TOKEN_1": "token-a",
            "APIFY_TOKEN_2": "",
            "APIFY_TOKEN_3": "token-c",
            "APIFY_TOKEN_4": "   ",
        },
    )

    tokens = settings.apify_tokens
    assert tokens == ["token-a", "token-c"]
    assert "" not in tokens

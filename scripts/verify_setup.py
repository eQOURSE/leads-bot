"""Verify the Phase 0 setup without making any external API calls.

Checks performed:
  1. Loads settings and reports which Tier-1 keys are present / missing.
  2. Imports every dependency declared in requirements.txt.
  3. Connects to the SQLite database and verifies all expected tables exist.
  4. Prints a clear PASS / FAIL summary.

Run:
    python scripts/verify_setup.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Make the project root importable when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sqlalchemy import create_engine, inspect

from config.settings import get_settings

# Maps the importable module name -> the distribution (pip) name.
# Covers every dependency listed in requirements.txt.
_DEPENDENCY_IMPORTS = {
    "google.cloud.aiplatform": "google-cloud-aiplatform",
    "google.generativeai": "google-generativeai",
    "google.auth": "google-auth",
    "gspread": "gspread",
    "oauth2client": "oauth2client",
    "langgraph": "langgraph",
    "langchain_core": "langchain-core",
    "pydantic": "pydantic",
    "pydantic_settings": "pydantic-settings",
    "dotenv": "python-dotenv",
    "httpx": "httpx",
    "requests": "requests",
    "tenacity": "tenacity",
    "feedparser": "feedparser",
    "bs4": "beautifulsoup4",
    "email_validator": "email-validator",
    "dns": "dnspython",
    "tldextract": "tldextract",
    "sqlalchemy": "sqlalchemy",
    "apify_client": "apify-client",
    "scrapegraph_py": "scrapegraph-py",
    "telegram": "python-telegram-bot",
    "streamlit": "streamlit",
    "yaml": "pyyaml",
    "pytest": "pytest",
    "pytest_asyncio": "pytest-asyncio",
    "respx": "respx",
}

_EXPECTED_TABLES = {
    "leads",
    "runs",
    "seen_domains",
    "replies",
    "api_usage",
}

# Tier-1 keys that should be present for Phase 1.
_TIER1_KEYS = [
    "HUNTER_API_KEY",
    "SCRAPEGRAPH_API_KEY",
    "SERPAPI_KEY",
    "NEWSDATA_API_KEY",
]


def check_settings() -> bool:
    """Report Tier-1 key presence. Returns True if all required keys are set."""
    print("\n[1] Settings / environment")
    print("-" * 50)
    settings = get_settings()

    all_present = True

    # Gemini auth: AI Studio key OR Vertex.
    mode = settings.gemini_auth_mode
    if mode == "ai_studio":
        print(f"  {'PRESENT':>8}  Gemini auth (AI Studio API key)")
    elif mode == "vertex":
        print(f"  {'PRESENT':>8}  Gemini auth (Vertex AI / GCP project)")
    else:
        all_present = False
        print(f"  {'MISSING':>8}  Gemini auth (set GEMINI_API_KEY or configure Vertex)")

    for key in _TIER1_KEYS:
        value = getattr(settings, key, None)
        present = bool(value and str(value).strip())
        all_present = all_present and present
        print(f"  {'PRESENT' if present else 'MISSING':>8}  {key}")

    tokens = settings.apify_tokens
    apify_ok = len(tokens) > 0
    all_present = all_present and apify_ok
    print(f"  {'PRESENT' if apify_ok else 'MISSING':>8}  APIFY_TOKEN(s) ({len(tokens)} found)")

    try:
        settings.validate_required()
        print("\n  validate_required(): OK")
    except ValueError as exc:
        print(f"\n  validate_required() reported missing keys:\n{exc}")

    return all_present


def check_imports() -> bool:
    """Import every declared dependency. Returns True if all succeed."""
    print("\n[2] Dependency imports")
    print("-" * 50)
    all_ok = True
    for module_name, dist_name in _DEPENDENCY_IMPORTS.items():
        try:
            importlib.import_module(module_name)
            print(f"  OK    {dist_name} ({module_name})")
        except Exception as exc:  # noqa: BLE001 - report any import failure
            all_ok = False
            print(f"  FAIL  {dist_name} ({module_name}): {exc}")
    return all_ok


def check_database() -> bool:
    """Verify the SQLite DB exists and contains all expected tables."""
    print("\n[3] Database schema")
    print("-" * 50)
    settings = get_settings()
    db_file = Path(settings.SQLITE_PATH)

    if not db_file.is_file():
        print(f"  FAIL  Database not found at {db_file}")
        print("        Run: python scripts/init_db.py")
        return False

    engine = create_engine(f"sqlite:///{db_file}", future=True)
    try:
        inspector = inspect(engine)
        existing = set(inspector.get_table_names())
    finally:
        engine.dispose()

    all_ok = True
    for table in sorted(_EXPECTED_TABLES):
        present = table in existing
        all_ok = all_ok and present
        print(f"  {'OK' if present else 'MISSING':>7}  {table}")

    return all_ok


def main() -> int:
    print("=" * 50)
    print("Phase 0 setup verification (no external API calls)")
    print("=" * 50)

    settings_ok = check_settings()
    imports_ok = check_imports()
    db_ok = check_database()

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  Settings (Tier-1 keys present) : {'PASS' if settings_ok else 'FAIL'}")
    print(f"  Dependency imports             : {'PASS' if imports_ok else 'FAIL'}")
    print(f"  Database schema                : {'PASS' if db_ok else 'FAIL'}")

    # Imports and DB are hard requirements for setup; missing API keys are a
    # soft warning (keys may legitimately not be filled in yet).
    setup_ok = imports_ok and db_ok
    print("-" * 50)
    print(f"  OVERALL SETUP : {'PASS' if setup_ok else 'FAIL'}")
    if setup_ok and not settings_ok:
        print("  (Note: some Tier-1 API keys are not yet set in .env)")
    print("=" * 50)

    return 0 if setup_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

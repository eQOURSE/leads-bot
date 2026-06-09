"""Typed application settings loaded from the environment / .env file.

Uses pydantic-settings to provide a single, validated source of truth for
configuration across the lead generation system.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # === Gemini auth ===
    # Two ways to authenticate to Gemini:
    #   1. AI Studio (Gemini Developer API): set GEMINI_API_KEY only.
    #   2. Vertex AI: set GCP_PROJECT_ID + GCP_REGION + a service account JSON.
    GEMINI_API_KEY: Optional[str] = None

    # === GCP / Vertex AI (optional — only needed for Vertex or Google Sheets) ===
    GCP_PROJECT_ID: Optional[str] = None
    GCP_REGION: str = "us-central1"
    GOOGLE_APPLICATION_CREDENTIALS: str = "./secrets/gcp-service-account.json"
    GEMINI_MODEL_QUALIFIER: str = "gemini-2.5-flash-lite"
    GEMINI_MODEL_RSS_PARSER: str = "gemini-2.5-flash-lite"
    GEMINI_MODEL_PERSONALIZER: str = "gemini-2.5-flash"
    GEMINI_MODEL_WRITER: str = "gemini-2.5-pro"
    GEMINI_MODEL_STRATEGIST: str = "gemini-2.5-pro"
    GEMINI_MODEL_VALIDATOR: str = "gemini-2.5-flash-lite"
    GEMINI_MODEL_FALLBACK: str = "gemini-2.5-flash"

    # === Primary Data Sources (Tier-1) ===
    VIBE_PROSPECTING_API_KEY: Optional[str] = None
    HUNTER_API_KEY: Optional[str] = None
    SCRAPEGRAPH_API_KEY: Optional[str] = None
    APIFY_TOKEN_1: Optional[str] = None
    APIFY_TOKEN_2: Optional[str] = None
    APIFY_TOKEN_3: Optional[str] = None
    APIFY_TOKEN_4: Optional[str] = None
    SERPAPI_KEY: Optional[str] = None
    NEWSDATA_API_KEY: Optional[str] = None
    COMPANIES_API_TOKEN: Optional[str] = None

    # === Phase 11 — Apify discovery actors (overridable) ===
    CRUNCHBASE_APIFY_ACTOR: Optional[str] = None
    WELLFOUND_APIFY_ACTOR: Optional[str] = None
    ENABLE_CRUNCHBASE_DISCOVERY: bool = True
    ENABLE_WELLFOUND_DISCOVERY: bool = True

    # === Parked (not wired in Phase 1) ===
    ABSTRACT_EMAIL_API_KEY: Optional[str] = None
    SMTP_HELO_DOMAIN: str = "verify.eqourse.com"
    ROCKETREACH_API_KEY: Optional[str] = None
    PHANTOMBUSTER_API_KEY: Optional[str] = None
    CLAY_API_KEY: Optional[str] = None
    OPENALEX_POLITE_EMAIL: Optional[str] = None

    # === Outputs (wired in Phase 7) ===
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_ID: Optional[str] = None
    TELEGRAM_SEND_EMPTY_DIGEST: bool = True  # send a "ran, found nothing" digest on 0-lead runs
    GOOGLE_SHEETS_CREDS_PATH: str = "./secrets/gcp-service-account.json"
    SHEET_ID: Optional[str] = None
    SHEET_TAB_TUTRAIN: str = "TUTRAIN_Leads"
    SHEET_TAB_CONTENT: str = "eQOURSE_Content_Leads"
    SHEET_TAB_AI_DATA: str = "eQOURSE_AI_Data_Leads"

    # === App config ===
    LOG_LEVEL: str = "INFO"
    DAILY_LEAD_TARGET_PER_SEGMENT: int = 15
    SQLITE_PATH: str = "./data/leads.db"
    STREAMLIT_PASSWORD: Optional[str] = None

    # === Source API base URLs (overridable so endpoints can be swapped) ===
    VIBE_PROSPECTING_BASE_URL: str = "https://api.vibeprospecting.ai/v1"
    HUNTER_BASE_URL: str = "https://api.hunter.io/v2"
    SCRAPEGRAPH_BASE_URL: str = "https://api.scrapegraphai.com/v1"
    APIFY_BASE_URL: str = "https://api.apify.com/v2"
    SERPAPI_BASE_URL: str = "https://serpapi.com"
    NEWSDATA_BASE_URL: str = "https://newsdata.io/api/1"
    COMPANIES_API_BASE_URL: str = "https://api.thecompaniesapi.com/v2"

    # === Source usage guardrails ===
    HUNTER_MONTHLY_CALL_LIMIT: int = 50
    SERPAPI_MONTHLY_LIMIT: int = 90  # leave a 10-call buffer under the 100 cap
    APIFY_INITIAL_CREDITS_USD: float = 5.0

    # --- Tier-1 keys required for Phase 1 to run ---
    # Maps the field name to a human-readable label for error messages.
    # Gemini auth is validated separately (AI Studio key OR Vertex), so it is
    # not listed here.
    _TIER1_REQUIRED = {
        "HUNTER_API_KEY": "Hunter.io API key (HUNTER_API_KEY)",
        "SCRAPEGRAPH_API_KEY": "ScrapeGraph API key (SCRAPEGRAPH_API_KEY)",
        "SERPAPI_KEY": "SerpAPI key (SERPAPI_KEY)",
        "NEWSDATA_API_KEY": "NewsData API key (NEWSDATA_API_KEY)",
    }

    @property
    def apify_tokens(self) -> List[str]:
        """Return the list of non-empty Apify tokens in numeric order."""
        candidates = [
            self.APIFY_TOKEN_1,
            self.APIFY_TOKEN_2,
            self.APIFY_TOKEN_3,
            self.APIFY_TOKEN_4,
        ]
        return [t.strip() for t in candidates if t and t.strip()]

    @property
    def is_gcp_configured(self) -> bool:
        """True if a GCP project id is set and the credentials file exists."""
        if not (self.GCP_PROJECT_ID and self.GCP_PROJECT_ID.strip()):
            return False
        return os.path.isfile(self.GOOGLE_APPLICATION_CREDENTIALS)

    @property
    def gemini_auth_mode(self) -> Optional[str]:
        """How Gemini will authenticate.

        Returns ``"ai_studio"`` if a Gemini Developer API key is set,
        ``"vertex"`` if GCP/Vertex is fully configured, else ``None``.
        AI Studio takes precedence since it is the simpler path.
        """
        if self.GEMINI_API_KEY and self.GEMINI_API_KEY.strip():
            return "ai_studio"
        if self.is_gcp_configured:
            return "vertex"
        return None

    def validate_required(self) -> None:
        """Raise a clear error listing any missing Tier-1 keys.

        Requires (a) a working Gemini auth path, (b) all Tier-1 source keys,
        and (c) at least one Apify token.
        """
        missing: List[str] = []

        if self.gemini_auth_mode is None:
            missing.append(
                "Gemini auth — set GEMINI_API_KEY (AI Studio) OR configure "
                "Vertex (GCP_PROJECT_ID + service account JSON)"
            )

        for field_name, label in self._TIER1_REQUIRED.items():
            value = getattr(self, field_name, None)
            if not (value and str(value).strip()):
                missing.append(label)

        if not self.apify_tokens:
            missing.append("at least one Apify token (APIFY_TOKEN_1..4)")

        if missing:
            bullets = "\n".join(f"  - {item}" for item in missing)
            raise ValueError(
                "Missing required Tier-1 configuration key(s):\n"
                f"{bullets}\n"
                "Set these in your .env file before running Phase 1."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()

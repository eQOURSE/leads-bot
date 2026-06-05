"""ScrapeGraph AI client — v2 SDK (scrapegraph-py >= 2.1.0).

Uses AsyncScrapeGraphAI for all extraction. Every call returns ApiResult[T];
status is checked before accessing .data. Results are cached in SQLite for a
configurable TTL (default 7 days), keyed by (method_name, url).

URL fallback strategy for extract_team_page:
  1. <base_url>/team
  2. <base_url>/about
  3. <base_url>/
Returns on first response that yields ≥1 member.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel
from scrapegraph_py import AsyncScrapeGraphAI  # type: ignore[import]  # requires Python 3.12+

from config.logging_config import setup_logging
from config.settings import Settings
from sources._cache import cache_get, cache_set
from sources._utils import track_usage
from sources.models import ProspectCandidate

# ---------------------------------------------------------------------------
# Local Pydantic schemas passed to sgai.extract(schema=...)
# ---------------------------------------------------------------------------

class _TeamMember(BaseModel):
    full_name: str
    title: str
    linkedin_url: Optional[str] = None


class _TeamPage(BaseModel):
    members: list[_TeamMember]


class _CompanySummary(BaseModel):
    one_liner: Optional[str] = None
    stage: Optional[str] = None
    founded_year: Optional[int] = None
    employee_estimate: Optional[str] = None
    target_market: Optional[str] = None


class _RecentNews(BaseModel):
    class _Announcement(BaseModel):
        type: Optional[str] = None
        date: Optional[str] = None
        summary: Optional[str] = None

    announcements: list[_Announcement] = []


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_TEAM_PROMPT = (
    "Find all founders, co-founders, C-level executives, VPs, and Heads of "
    "departments. For each return full_name, title, and linkedin_url if visible. "
    "Skip junior/intern/associate roles."
)

_SUMMARY_PROMPT = (
    "Return a JSON summary with: one_liner (2-sentence description), "
    "stage (pre-seed/seed/series-a/series-b/later), founded_year (int or null), "
    "employee_estimate (range string), target_market (string)."
)

_NEWS_PROMPT = (
    "Find announcements from the last 6 months: funding, product launches, "
    "hires, partnerships. Return JSON: {announcements: [{type, date, summary}]}."
)

_TEAM_URL_VARIANTS = ["/team", "/about", "/about-us", ""]

_DEFAULT_TTL_DAYS = 7


class ScrapeGraphClient:
    """Async ScrapeGraph v2 client with caching and usage tracking."""

    source_name = "scrapegraph"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.log = setup_logging(f"source.{self.source_name}")
        self._scrapegraph_available = True  # set False on first auth error

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    async def _extract(
        self,
        method_name: str,
        url: str,
        prompt: str,
        schema_cls: type[BaseModel],
        cache_ttl_days: int = _DEFAULT_TTL_DAYS,
    ) -> Optional[dict]:
        """Run one extract call with caching. Returns parsed json_data dict or None."""
        if not self._scrapegraph_available:
            self.log.warning("scrapegraph unavailable for this run, skipping %s", url)
            return None

        # Cache check
        cached = await cache_get(method_name, url, self.settings)
        if cached is not None:
            self.log.info("scrapegraph.%s cache hit for %s", method_name, url)
            return cached

        try:
            async with AsyncScrapeGraphAI(
                api_key=self.settings.SCRAPEGRAPH_API_KEY or ""
            ) as sgai:
                result = await sgai.extract(
                    prompt=prompt,
                    url=url,
                    schema=schema_cls.model_json_schema(),
                )
        except Exception as exc:  # noqa: BLE001
            self.log.error(
                "scrapegraph.%s network/import error for %s: %s", method_name, url, exc
            )
            return None

        if result.status != "success":
            err = getattr(result, "error", "unknown")
            # Auth errors disable scrapegraph for the whole run
            if isinstance(err, str) and any(
                kw in err.lower() for kw in ("auth", "invalid", "unauthorized", "key")
            ):
                self.log.error(
                    "scrapegraph auth error — disabling for this run: %s", err
                )
                self._scrapegraph_available = False
            else:
                self.log.warning(
                    "scrapegraph.%s API error for %s: %s", method_name, url, err
                )
            return None

        # Extract json_data from the ApiResult
        json_data = None
        try:
            json_data = result.data.json_data  # type: ignore[union-attr]
        except AttributeError:
            pass

        if json_data is None:
            self.log.warning(
                "scrapegraph.%s returned success but no json_data for %s",
                method_name,
                url,
            )
            return None

        # Track usage and cache
        await self._track_usage()
        await cache_set(method_name, url, json_data, cache_ttl_days, self.settings)
        return json_data

    async def _track_usage(self) -> None:
        try:
            await track_usage(self.source_name, 1, None, self.settings)
        except Exception as exc:  # noqa: BLE001
            self.log.warning("Failed to track scrapegraph usage: %s", exc)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    async def extract_team_page(
        self, company_url: str, cache_ttl_days: int = _DEFAULT_TTL_DAYS
    ) -> list[ProspectCandidate]:
        """Return decision-maker candidates from the company's team/about page.

        Tries /team → /about → /about-us → / in order; returns on first hit
        with ≥1 member. Uses the company_url base (strips any trailing path).
        """
        # Normalise base URL — strip trailing slash and any path beyond the origin
        base = company_url.rstrip("/")
        # Keep only scheme+host if a path was given that isn't a variant
        from urllib.parse import urlparse
        parsed = urlparse(base)
        origin = f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else base

        for variant in _TEAM_URL_VARIANTS:
            target_url = f"{origin}{variant}"
            cache_key = f"extract_team_page:{target_url}"

            data = await self._extract(
                "extract_team_page",
                target_url,
                _TEAM_PROMPT,
                _TeamPage,
                cache_ttl_days,
            )
            if data is None:
                continue

            # Parse into _TeamPage model
            try:
                page = _TeamPage.model_validate(data)
                members = page.members
            except Exception:
                # Fallback: look for a "members" key or treat as list
                if isinstance(data, list):
                    raw_members = data
                else:
                    raw_members = (
                        data.get("members")
                        or data.get("team")
                        or data.get("people")
                        or []
                    )
                members = []
                for m in raw_members:
                    if isinstance(m, dict):
                        name = m.get("full_name") or m.get("name")
                        if name:
                            members.append(
                                _TeamMember(
                                    full_name=name,
                                    title=m.get("title") or "",
                                    linkedin_url=m.get("linkedin_url"),
                                )
                            )

            if not members:
                continue  # try next variant

            prospects: list[ProspectCandidate] = []
            for m in members:
                try:
                    prospects.append(
                        ProspectCandidate(
                            full_name=m.full_name,
                            title=m.title,
                            company_domain=origin,
                            linkedin_url=m.linkedin_url,
                            source="scrapegraph",
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    self.log.warning("scrapegraph team member skipped: %s", exc)

            if prospects:
                self.log.info(
                    "scrapegraph.extract_team_page found %d members at %s",
                    len(prospects),
                    target_url,
                )
                return prospects

        self.log.info(
            "scrapegraph.extract_team_page found no members for %s", company_url
        )
        return []

    async def extract_company_summary(
        self, company_url: str, cache_ttl_days: int = _DEFAULT_TTL_DAYS
    ) -> dict:
        """Extract a structured company summary. Returns {} on failure."""
        data = await self._extract(
            "extract_company_summary",
            company_url,
            _SUMMARY_PROMPT,
            _CompanySummary,
            cache_ttl_days,
        )
        return data if data is not None else {}

    async def extract_recent_news(
        self, company_url: str, cache_ttl_days: int = _DEFAULT_TTL_DAYS
    ) -> dict:
        """Extract recent company announcements. Returns {} on failure."""
        data = await self._extract(
            "extract_recent_news",
            company_url,
            _NEWS_PROMPT,
            _RecentNews,
            cache_ttl_days,
        )
        return data if data is not None else {}

    async def get_credits_remaining(self) -> int:
        """Return remaining ScrapeGraph credits, or -1 on failure."""
        try:
            async with AsyncScrapeGraphAI(
                api_key=self.settings.SCRAPEGRAPH_API_KEY or ""
            ) as sgai:
                result = await sgai.credits()

            if result.status == "success" and result.data:
                remaining = getattr(result.data, "remaining", None)
                if remaining is not None:
                    return int(remaining)
        except Exception as exc:  # noqa: BLE001
            self.log.warning("scrapegraph.get_credits_remaining failed: %s", exc)
        return -1

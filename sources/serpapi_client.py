"""SerpAPI client for Google organic + news search.

Free tier is 100 searches/month. A hard guardrail stops at
``SERPAPI_MONTHLY_LIMIT`` (default 90) to leave a buffer. Usage is tracked in
the ``api_usage`` table.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import httpx

from config.settings import Settings
from sources._utils import month_call_count, utcnow
from sources.base import BaseSourceClient
from sources.models import NewsItem, SearchResult

_DEFAULT_TIMEOUT = 30.0


class SerpAPIClient(BaseSourceClient):
    source_name = "serpapi"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.api_key = settings.SERPAPI_KEY
        self.base_url = settings.SERPAPI_BASE_URL.rstrip("/")
        self.monthly_limit = settings.SERPAPI_MONTHLY_LIMIT

    async def _over_limit(self, force: bool = False) -> bool:
        if force:
            return False
        used = await month_call_count(self.source_name, self.settings)
        if used >= self.monthly_limit:
            self.log.warning(
                "SerpAPI monthly budget reached (%s/%s) — skipping.",
                used,
                self.monthly_limit,
            )
            return True
        return False

    async def search(
        self, query: str, location: str = "United States", num: int = 10
    ) -> list[SearchResult]:
        if await self._over_limit():
            return []

        url = f"{self.base_url}/search"
        params = {
            "engine": "google",
            "q": query,
            "location": location,
            "num": num,
            "api_key": self.api_key,
        }
        try:
            async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
                resp = await self._request(client, "GET", url, params=params)
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            self.log.error("serpapi.search failed: %s", exc)
            return []

        await self._track(1, self._remaining_from(data))

        results: list[SearchResult] = []
        for i, r in enumerate(data.get("organic_results", []), start=1):
            try:
                results.append(
                    SearchResult(
                        title=r.get("title") or "",
                        url=r.get("link") or "",
                        snippet=r.get("snippet") or "",
                        position=r.get("position") or i,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                self.log.warning("serpapi result skipped: %s", exc)
        return results

    async def search_news(
        self, query: str, days_back: int = 30
    ) -> list[NewsItem]:
        if await self._over_limit():
            return []

        url = f"{self.base_url}/search"
        params = {
            "engine": "google_news",
            "q": query,
            "api_key": self.api_key,
        }
        try:
            async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
                resp = await self._request(client, "GET", url, params=params)
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            self.log.error("serpapi.search_news failed: %s", exc)
            return []

        await self._track(1, self._remaining_from(data))

        cutoff = utcnow() - timedelta(days=days_back)
        items: list[NewsItem] = []
        for r in data.get("news_results", []):
            published = self._parse_date(r.get("date"))
            if published and published < cutoff:
                continue
            try:
                items.append(
                    NewsItem(
                        title=r.get("title") or "",
                        url=r.get("link") or "",
                        published_at=published or utcnow(),
                        source_name=(r.get("source") or {}).get("name", "google_news")
                        if isinstance(r.get("source"), dict)
                        else (r.get("source") or "google_news"),
                        snippet=r.get("snippet") or "",
                    )
                )
            except Exception as exc:  # noqa: BLE001
                self.log.warning("serpapi news item skipped: %s", exc)
        return items

    async def get_remaining_searches(self) -> int:
        url = f"{self.base_url}/account"
        params = {"api_key": self.api_key}
        try:
            async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
                resp = await self._request(client, "GET", url, params=params)
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            self.log.error("serpapi.get_remaining_searches failed: %s", exc)
            return 0
        total = data.get("total_searches_left")
        return int(total) if total is not None else 0

    @staticmethod
    def _remaining_from(data: dict) -> Optional[int]:
        meta = data.get("search_metadata") or {}
        # SerpAPI does not always echo remaining; left as None when absent.
        return meta.get("total_searches_left")

    @staticmethod
    def _parse_date(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        for fmt in ("%m/%d/%Y, %I:%M %p, %z", "%m/%d/%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(value, fmt).replace(tzinfo=None)
            except (ValueError, TypeError):
                continue
        return None

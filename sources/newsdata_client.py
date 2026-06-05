"""newsdata.io client.

Free tier: 200 requests/day, production-allowed. Base URL from settings
(default https://newsdata.io/api/1).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import httpx

from config.settings import Settings
from sources._utils import utcnow
from sources.base import BaseSourceClient
from sources.models import NewsItem

_DEFAULT_TIMEOUT = 30.0


class NewsDataClient(BaseSourceClient):
    source_name = "newsdata"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.api_key = settings.NEWSDATA_API_KEY
        self.base_url = settings.NEWSDATA_BASE_URL.rstrip("/")

    async def search_funding_news(
        self,
        keywords: list[str],
        days_back: int = 90,
        countries: Optional[list[str]] = None,
        categories: Optional[list[str]] = None,
    ) -> list[NewsItem]:
        if countries is None:
            countries = ["us"]
        if categories is None:
            categories = ["business", "technology"]

        # Build an OR query of the supplied keywords combined with funding terms.
        kw = " OR ".join(keywords) if keywords else "funding"
        query = f"(raised OR seed OR \"series A\") AND ({kw})"

        url = f"{self.base_url}/latest"
        params = {
            "apikey": self.api_key,
            "q": query,
            "language": "en",
        }
        # Only constrain by country/category when explicitly provided. Empty
        # lists mean "search globally / all categories", which avoids
        # over-filtering on the free tier's 48-hour window.
        if countries:
            params["country"] = ",".join(countries)
        if categories:
            params["category"] = ",".join(categories)
        return await self._fetch(url, params, days_back)

    async def search_company_news(
        self, company_name: str, days_back: int = 30
    ) -> list[NewsItem]:
        url = f"{self.base_url}/latest"
        params = {
            "apikey": self.api_key,
            "q": f"\"{company_name}\"",
            "language": "en",
        }
        return await self._fetch(url, params, days_back)

    async def _fetch(
        self, url: str, params: dict, days_back: int
    ) -> list[NewsItem]:
        try:
            async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
                resp = await self._request(client, "GET", url, params=params)
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            self.log.error("newsdata fetch failed: %s", exc)
            return []

        await self._track(1, data.get("totalResults"))

        items: list[NewsItem] = []
        for r in data.get("results", []) or []:
            published = self._parse_date(r.get("pubDate"))
            try:
                items.append(
                    NewsItem(
                        title=r.get("title") or "",
                        url=r.get("link") or "",
                        published_at=published or utcnow(),
                        source_name=r.get("source_id") or "newsdata",
                        snippet=r.get("description") or "",
                    )
                )
            except Exception as exc:  # noqa: BLE001
                self.log.warning("newsdata item skipped: %s", exc)
        return items

    @staticmethod
    def _parse_date(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(value, fmt)
            except (ValueError, TypeError):
                continue
        return None

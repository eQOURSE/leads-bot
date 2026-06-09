"""RSS funding monitor.

No API key needed for fetching feeds (uses feedparser). The
``extract_company_from_headline`` method uses Gemini Flash-Lite (model from
settings.GEMINI_MODEL_RSS_PARSER) to turn a headline + snippet into structured
funding data.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from time import mktime
from typing import Optional

import feedparser

from config.settings import Settings
from sources._gemini import generate_text
from sources._utils import normalize_domain, utcnow
from sources.base import BaseSourceClient
from sources.models import CompanyCandidate, NewsItem

_DEFAULT_FEEDS = [
    "https://techcrunch.com/category/venture/feed/",
    "https://strictlyvc.com/feed/",
    "https://www.edsurge.com/articles_rss",
    "https://tech.eu/feed/",
    "https://news.crunchbase.com/feed/",
]

_DEFAULT_KEYWORDS = ["raises", "raised", "funding", "seed", "series"]

_EXTRACT_PROMPT = (
    "Extract from this funding announcement headline. Return strict JSON: "
    "{company_name, funding_amount_usd: int or null, "
    "funding_stage: pre-seed/seed/series-a/series-b/other, "
    "announcement_date: YYYY-MM-DD or null}. If not a funding announcement, "
    "return {company_name: null}.\n\nHeadline + snippet:\n"
)


class RSSFundingMonitor(BaseSourceClient):
    source_name = "rss_feeds"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.model = settings.GEMINI_MODEL_RSS_PARSER

    async def fetch_recent_funding(
        self,
        feeds: Optional[list[str]] = None,
        since_days: int = 240,
        keywords: Optional[list[str]] = None,
    ) -> list[NewsItem]:
        feeds = feeds or _DEFAULT_FEEDS
        keywords = keywords or _DEFAULT_KEYWORDS
        cutoff = utcnow() - timedelta(days=since_days)

        # feedparser is blocking; run each parse in a thread, concurrently.
        parsed_feeds = await asyncio.gather(
            *(asyncio.to_thread(feedparser.parse, url) for url in feeds),
            return_exceptions=True,
        )

        items: list[NewsItem] = []
        for url, parsed in zip(feeds, parsed_feeds):
            if isinstance(parsed, Exception):
                self.log.warning("rss feed failed (%s): %s", url, parsed)
                continue
            for entry in getattr(parsed, "entries", []):
                title = entry.get("title", "") or ""
                snippet = entry.get("summary", "") or ""
                haystack = f"{title} {snippet}".lower()
                if not any(k.lower() in haystack for k in keywords):
                    continue
                published = self._entry_date(entry)
                if published and published < cutoff:
                    continue
                try:
                    items.append(
                        NewsItem(
                            title=title,
                            url=entry.get("link", "") or "",
                            published_at=published or utcnow(),
                            source_name=self._feed_name(parsed, url),
                            snippet=snippet,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    self.log.warning("rss entry skipped: %s", exc)

        await self._track(len(feeds), None)
        self.log.info("rss_feeds: %s funding-related items from %s feeds", len(items), len(feeds))
        return items

    # Per-run domain-resolution counters (reset each fetch_recent_funding call
    # via reset_resolution_metrics()). Read by the hunter for measurement.
    _resolution_attempts: int = 0
    _resolution_hits: int = 0

    def reset_resolution_metrics(self) -> None:
        self._resolution_attempts = 0
        self._resolution_hits = 0

    @property
    def article_link_resolution_rate(self) -> float:
        if self._resolution_attempts == 0:
            return 0.0
        return self._resolution_hits / self._resolution_attempts

    async def extract_company_from_headline(
        self, news_item: NewsItem
    ) -> Optional[CompanyCandidate]:
        # Step 1 — try to resolve the real company domain from the article body.
        resolved_domain: Optional[str] = None
        if news_item.url:
            self._resolution_attempts += 1
            try:
                from sources._article_extract import resolve_company_domain
                resolved_domain = await resolve_company_domain(news_item.url, self.settings)
            except Exception as exc:  # noqa: BLE001
                self.log.warning("article domain resolution error (%s): %s", news_item.url, exc)
            if resolved_domain:
                self._resolution_hits += 1

        # Step 2 — Gemini extraction. If we have candidate domain(s), let the
        # model confirm/choose; otherwise just extract company facts.
        if resolved_domain:
            prompt = (
                f"{_EXTRACT_PROMPT}{news_item.title}\n{news_item.snippet}\n\n"
                f"Candidate company domain from the article body: {resolved_domain}. "
                "Include a 'domain' field set to this domain if it plausibly "
                "belongs to the funded company, else null."
            )
        else:
            prompt = f"{_EXTRACT_PROMPT}{news_item.title}\n{news_item.snippet}"

        raw = await generate_text(self.settings, self.model, prompt)
        if not raw:
            return None

        parsed = self._parse_json(raw)
        if not parsed:
            return None

        company_name = parsed.get("company_name")
        if not company_name:
            return None

        # Prefer: model-confirmed domain → article-resolved domain → slug.
        model_domain = parsed.get("domain")
        domain = (
            normalize_domain(model_domain) if model_domain else ""
        ) or resolved_domain or self._guess_domain(company_name)

        funding_date = self._parse_iso_date(parsed.get("announcement_date"))
        try:
            return CompanyCandidate(
                domain=domain,
                name=company_name,
                funding_amount_usd=self._to_float(parsed.get("funding_amount_usd")),
                funding_stage=parsed.get("funding_stage"),
                funding_date=funding_date,
                funding_source=news_item.source_name,
                raw_source="rss_feeds",
                # Higher confidence when we resolved a real domain.
                confidence=0.65 if domain and not domain.endswith(".unknown") else 0.5,
            )
        except Exception as exc:  # noqa: BLE001
            self.log.warning("rss company extraction skipped: %s", exc)
            return None

    # --- helpers ---

    @staticmethod
    def _entry_date(entry) -> Optional[datetime]:
        for key in ("published_parsed", "updated_parsed"):
            t = entry.get(key)
            if t:
                try:
                    return datetime.fromtimestamp(mktime(t))
                except (TypeError, ValueError, OverflowError):
                    continue
        return None

    @staticmethod
    def _feed_name(parsed, url: str) -> str:
        feed = getattr(parsed, "feed", {})
        return feed.get("title") or url

    @staticmethod
    def _parse_json(raw: str) -> Optional[dict]:
        text = raw.strip()
        if text.startswith("```"):
            # strip ```json ... ``` fences
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _parse_iso_date(value):
        if not value:
            return None
        try:
            return datetime.strptime(value, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _to_float(value):
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _guess_domain(company_name: str) -> str:
        # Placeholder domain derived from the name; real resolution happens in a
        # later enrichment phase. Keeps the CompanyCandidate.domain non-empty.
        slug = "".join(ch for ch in company_name.lower() if ch.isalnum())
        return f"{slug}.unknown" if slug else "unknown.unknown"

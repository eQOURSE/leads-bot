"""Phase 11 — Resolve a funded company's real domain from a news article.

News/RSS headlines give us a company NAME but the article URL is the
publisher's (techcrunch.com), not the company's. This module fetches the
article page and extracts the most likely subject-company domain from the
outbound links, so downstream enrichment/email steps have a real domain to
work with instead of a ``<slug>.unknown`` placeholder.

The link-filtering/selection logic is pure (no network) and unit-tested.
The fetch is cached 24h in SQLite (sources._cache) keyed by URL.
"""

from __future__ import annotations

import re
from typing import Optional
from urllib.parse import urlparse

import httpx

from agents._constants import NEWS_SOURCE_DOMAINS, domain_is_news_source
from config.logging_config import setup_logging
from sources._cache import cache_get, cache_set
from sources._utils import normalize_domain

_log = setup_logging("source.article_extract")

_CACHE_METHOD = "article_html"
_CACHE_TTL_DAYS = 1
_FETCH_TIMEOUT = 8.0

# Social / tracking / aggregator domains that are never the subject company.
_SOCIAL_DOMAINS = frozenset({
    "linkedin.com", "twitter.com", "x.com", "facebook.com", "instagram.com",
    "tiktok.com", "youtube.com", "youtu.be", "pinterest.com", "reddit.com",
    "threads.net", "mastodon.social",
})
_TRACKING_DOMAINS = frozenset({
    "bit.ly", "buff.ly", "t.co", "lnkd.in", "ow.ly", "hubs.ly", "dlvr.it",
    "trib.al", "ift.tt", "goo.gl", "tinyurl.com",
})
# Path fragments that indicate a non-company internal/utility link.
_BAD_PATH_FRAGMENTS = (
    "/privacy", "/cookie", "/terms", "/about-us", "/about", "/contact",
    "/careers", "/jobs", "/subscribe", "/newsletter", "/advertise",
    "/login", "/signin", "/sign-in", "/author/", "/tag/", "/category/",
    "/feed", "/rss",
)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}


# ---------------------------------------------------------------------------
# Pure logic (no network) — unit-tested
# ---------------------------------------------------------------------------

def _publisher_domain(article_url: str) -> str:
    return normalize_domain(article_url)


def _is_excluded_domain(domain: str, publisher: str) -> bool:
    """True if domain should NOT be considered a candidate company domain."""
    if not domain:
        return True
    d = normalize_domain(domain)
    if not d or "." not in d:
        return True
    if d == publisher:
        return True
    if d in _SOCIAL_DOMAINS or d in _TRACKING_DOMAINS:
        return True
    if domain_is_news_source(d):
        return True
    return False


def _is_excluded_path(href: str, on_news_site: bool) -> bool:
    """True if the link path looks like a utility/internal page."""
    try:
        path = urlparse(href).path.lower()
    except Exception:  # noqa: BLE001
        return False
    if not on_news_site:
        return False
    return any(frag in path for frag in _BAD_PATH_FRAGMENTS)


def extract_candidate_domains(html: str, article_url: str) -> list[str]:
    """Return ordered, de-duplicated candidate company domains from article HTML.

    Order reflects appearance position in the document (body links first, then
    canonical/og:url as backups). Excludes publisher, news, social, tracking,
    and utility links.
    """
    from bs4 import BeautifulSoup

    publisher = _publisher_domain(article_url)
    soup = BeautifulSoup(html, "html.parser")

    # Total text length for "first 60% of body" weighting.
    full_text = soup.get_text(" ", strip=True) or ""
    total_len = max(len(full_text), 1)

    ordered: list[tuple[int, str]] = []  # (position_fraction*1000, domain)
    seen: set[str] = set()

    # Body anchors, in document order.
    anchors = soup.find_all("a", href=True)
    n_anchors = max(len(anchors), 1)
    for idx, a in enumerate(anchors):
        href = a.get("href", "")
        if not href or href.startswith("#") or href.startswith("mailto:"):
            continue
        if not href.startswith("http"):
            continue
        d = normalize_domain(href)
        if _is_excluded_domain(d, publisher):
            continue
        # Path filtering only matters for news-source domains (utility pages on
        # publishers). A company's own /about or /careers link is still a valid
        # signal of the company domain, so do NOT drop it.
        if domain_is_news_source(d) and _is_excluded_path(href, on_news_site=True):
            continue
        if d in seen:
            continue
        seen.add(d)
        # Approximate position by anchor index.
        frac = int((idx / n_anchors) * 1000)
        ordered.append((frac, d))

    ordered.sort(key=lambda t: t[0])
    body_domains = [d for _, d in ordered]

    # Canonical / og:url as backups (only if not the publisher).
    backups: list[str] = []
    for tag, attr, key in (
        ("link", "rel", "canonical"),
        ("meta", "property", "og:url"),
    ):
        for el in soup.find_all(tag):
            if tag == "link" and key not in (el.get("rel") or []):
                continue
            if tag == "meta" and el.get("property") != key:
                continue
            href = el.get("href") or el.get("content") or ""
            d = normalize_domain(href)
            if d and not _is_excluded_domain(d, publisher) and d not in seen:
                seen.add(d)
                backups.append(d)

    return body_domains + backups


def pick_company_domain(html: str, article_url: str) -> Optional[str]:
    """Pick the single most likely company domain (first body candidate)."""
    candidates = extract_candidate_domains(html, article_url)
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Network fetch (cached)
# ---------------------------------------------------------------------------

async def fetch_article_html(url: str, settings) -> Optional[str]:
    """Fetch article HTML with one retry, 8s timeout, 24h SQLite cache."""
    if not url or not url.startswith("http"):
        return None

    cached = await cache_get(_CACHE_METHOD, url, settings)
    if cached is not None:
        return cached.get("html")

    html: Optional[str] = None
    for attempt in (1, 2):
        try:
            async with httpx.AsyncClient(
                timeout=_FETCH_TIMEOUT, follow_redirects=True, headers=_HEADERS
            ) as client:
                resp = await client.get(url)
                if resp.status_code == 200 and resp.text:
                    html = resp.text
                    break
        except Exception as exc:  # noqa: BLE001
            if attempt == 2:
                _log.warning("article fetch failed (%s): %s", url, exc)
        # brief implicit backoff between the two attempts handled by caller loop

    # Cache even an empty result to avoid re-fetching a dead URL all day.
    try:
        await cache_set(_CACHE_METHOD, url, {"html": html or ""}, _CACHE_TTL_DAYS, settings)
    except Exception:  # noqa: BLE001
        pass

    return html


async def resolve_company_domain(url: str, settings) -> Optional[str]:
    """Fetch the article and return the best candidate company domain, or None."""
    html = await fetch_article_html(url, settings)
    if not html:
        return None
    try:
        return pick_company_domain(html, url)
    except Exception as exc:  # noqa: BLE001
        _log.warning("domain extraction failed (%s): %s", url, exc)
        return None

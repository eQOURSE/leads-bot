"""Tests for RSSFundingMonitor.

RSS fetching uses feedparser (not httpx), so feedparser.parse is monkeypatched.
Gemini extraction is monkeypatched too — no real network calls.
"""

from __future__ import annotations

import time
from datetime import datetime

import pytest

import sources.rss_feeds as rss_mod
from sources.models import NewsItem
from sources.rss_feeds import RSSFundingMonitor
from tests.conftest import count_usage


class _FakeParsed:
    def __init__(self, entries, title="Test Feed"):
        self.entries = entries
        self.feed = {"title": title}


def _entry(title, summary, link):
    return {
        "title": title,
        "summary": summary,
        "link": link,
        "published_parsed": time.localtime(),
    }


@pytest.mark.asyncio
async def test_happy_path_fetch_funding(test_settings, monkeypatch):
    def fake_parse(url):
        return _FakeParsed(
            [
                _entry("Acme raises $10M Series A", "edtech startup funding", "https://a.test"),
                _entry("Unrelated product review", "no money here", "https://b.test"),
            ]
        )

    monkeypatch.setattr(rss_mod.feedparser, "parse", fake_parse)

    monitor = RSSFundingMonitor(test_settings)
    items = await monitor.fetch_recent_funding(feeds=["https://feed.test/rss"])

    # Only the funding-keyword entry should match.
    assert len(items) == 1
    assert "raises" in items[0].title.lower()
    assert count_usage(test_settings, "rss_feeds") == 1


@pytest.mark.asyncio
async def test_feed_failure_does_not_raise(test_settings, monkeypatch):
    def boom(url):
        raise RuntimeError("network down")

    monkeypatch.setattr(rss_mod.feedparser, "parse", boom)

    monitor = RSSFundingMonitor(test_settings)
    items = await monitor.fetch_recent_funding(feeds=["https://feed.test/rss"])

    assert items == []


@pytest.mark.asyncio
async def test_extract_company_from_headline(test_settings, monkeypatch):
    async def fake_generate(settings, model, prompt):
        return (
            '{"company_name": "Acme", "funding_amount_usd": 10000000, '
            '"funding_stage": "series-a", "announcement_date": "2026-05-01"}'
        )

    monkeypatch.setattr(rss_mod, "generate_text", fake_generate)

    monitor = RSSFundingMonitor(test_settings)
    news = NewsItem(
        title="Acme raises $10M Series A",
        url="https://a.test",
        published_at=datetime.utcnow(),
        source_name="TestFeed",
        snippet="edtech funding",
    )
    company = await monitor.extract_company_from_headline(news)

    assert company is not None
    assert company.name == "Acme"
    assert company.funding_stage == "series-a"
    assert company.funding_amount_usd == 10000000.0


@pytest.mark.asyncio
async def test_extract_returns_none_for_non_funding(test_settings, monkeypatch):
    async def fake_generate(settings, model, prompt):
        return '{"company_name": null}'

    monkeypatch.setattr(rss_mod, "generate_text", fake_generate)

    monitor = RSSFundingMonitor(test_settings)
    news = NewsItem(
        title="A general tech article",
        url="https://x.test",
        published_at=datetime.utcnow(),
        source_name="TestFeed",
        snippet="nothing about funding",
    )
    company = await monitor.extract_company_from_headline(news)

    assert company is None


@pytest.mark.asyncio
async def test_extract_handles_empty_model_output(test_settings, monkeypatch):
    async def fake_generate(settings, model, prompt):
        return ""

    monkeypatch.setattr(rss_mod, "generate_text", fake_generate)

    monitor = RSSFundingMonitor(test_settings)
    news = NewsItem(
        title="Acme raises $10M",
        url="https://a.test",
        published_at=datetime.utcnow(),
        source_name="TestFeed",
        snippet="funding",
    )
    company = await monitor.extract_company_from_headline(news)

    assert company is None

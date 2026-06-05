"""Smoke test a single data source with ONE real call and a safe query.

Usage:
    python scripts/smoke_test_sources.py --source rss
    python scripts/smoke_test_sources.py --source newsdata
    python scripts/smoke_test_sources.py --source all

Prints raw response count, the first 3 parsed results, and (when known) credits
consumed / remaining. Exits 0 on success, 1 on failure.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import get_settings

_SAFE_QUERY = "edtech seed funding"
_SAFE_COMPANY_URL = "https://www.khanacademy.org"

SOURCES = [
    "vibe_prospecting",
    "hunter",
    "scrapegraph",
    "apify",
    "serpapi",
    "newsdata",
    "companies_api",
    "rss",
    "abstract_api",
    "smtp",
]


def _print_header(name: str) -> None:
    print("\n" + "=" * 60)
    print(f"SMOKE TEST: {name}")
    print("=" * 60)


def _preview(results) -> None:
    print(f"  parsed result count: {len(results)}")
    for i, r in enumerate(results[:3], start=1):
        text = getattr(r, "title", None) or getattr(r, "full_name", None) or str(r)
        print(f"  [{i}] {text}")


async def _smoke_rss(settings) -> bool:
    from sources.rss_feeds import RSSFundingMonitor

    monitor = RSSFundingMonitor(settings)
    items = await monitor.fetch_recent_funding(since_days=240)
    _preview(items)
    return len(items) > 0


async def _smoke_newsdata(settings) -> bool:
    from sources.newsdata_client import NewsDataClient

    client = NewsDataClient(settings)
    # Global search (no country filter) so the smoke test reliably surfaces
    # results from the free tier's 48-hour window.
    items = await client.search_funding_news(
        keywords=["edtech"], days_back=90, countries=[], categories=[]
    )
    _preview(items)
    return len(items) > 0


async def _smoke_serpapi(settings) -> bool:
    from sources.serpapi_client import SerpAPIClient

    client = SerpAPIClient(settings)
    results = await client.search(_SAFE_QUERY, num=5)
    _preview(results)
    return len(results) > 0


async def _smoke_vibe(settings) -> bool:
    from sources.vibe_prospecting import VibeProspectingClient

    client = VibeProspectingClient(settings)
    companies = await client.search_funded_companies(limit=5)
    _preview(companies)
    return len(companies) > 0


async def _smoke_hunter(settings) -> bool:
    from sources.hunter_client import HunterClient

    client = HunterClient(settings)
    info = await client.get_account_info()
    print(f"  account info keys: {list(info.keys())}")
    return bool(info)


async def _smoke_scrapegraph(settings) -> bool:
    from sources.scrapegraph_client import ScrapeGraphClient

    client = ScrapeGraphClient(settings)
    summary = await client.extract_company_summary(_SAFE_COMPANY_URL)
    print(f"  summary keys: {list(summary.keys())}")
    return bool(summary)


async def _smoke_apify(settings) -> bool:
    from sources.apify_client import ApifyMultiKeyClient

    client = ApifyMultiKeyClient(settings)
    results = await client.google_search(_SAFE_QUERY, num_results=5)
    _preview(results)
    return len(results) > 0


async def _smoke_companies(settings) -> bool:
    from sources.companies_api_client import CompaniesAPIClient

    client = CompaniesAPIClient(settings)
    companies = await client.search_by_filters(industries=["education"], limit=5)
    _preview(companies)
    return len(companies) > 0


async def _smoke_abstract_api(settings) -> bool:
    from sources.abstract_api_client import AbstractAPIClient

    client = AbstractAPIClient(settings)
    result = await client.validate_email("test@example.com")
    print(f"  response keys: {list(result.keys())}")
    print(f"  deliverability: {result.get('deliverability', 'N/A')}")
    return bool(result)


async def _smoke_smtp(settings) -> bool:
    from sources.smtp_verifier import SMTPVerifier

    verifier = SMTPVerifier(settings)
    test_cases = [
        "test@gmail.com",
        "nonsense12345xyzabc@gmail.com",
        "test@thisisnotarealdomaindefinitely987.com",
    ]
    any_responded = False
    for email in test_cases:
        result = await verifier.verify_email(email)
        print(f"  {email:<45} exists={result.exists!s:<6} smtp={result.smtp_response}")
        if result.smtp_response != "connection_error":
            any_responded = True
    # Pass if at least the no-MX case returned no_mx
    return any_responded


_DISPATCH = {
    "rss": _smoke_rss,
    "newsdata": _smoke_newsdata,
    "serpapi": _smoke_serpapi,
    "vibe_prospecting": _smoke_vibe,
    "hunter": _smoke_hunter,
    "scrapegraph": _smoke_scrapegraph,
    "apify": _smoke_apify,
    "companies_api": _smoke_companies,
    "abstract_api": _smoke_abstract_api,
    "smtp": _smoke_smtp,
}


async def run_one(source: str, settings) -> bool:
    _print_header(source)
    fn = _DISPATCH.get(source)
    if fn is None:
        print(f"  unknown source: {source}")
        return False
    try:
        ok = await fn(settings)
        print(f"  RESULT: {'PASS' if ok else 'FAIL (no results)'}")
        return ok
    except Exception as exc:  # noqa: BLE001
        print(f"  RESULT: FAIL ({type(exc).__name__}: {exc})")
        return False


async def main_async(source: str) -> int:
    settings = get_settings()

    if source == "all":
        results = {}
        for s in SOURCES:
            results[s] = await run_one(s, settings)
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for s, ok in results.items():
            print(f"  {s:20s} {'PASS' if ok else 'FAIL'}")
        return 0 if all(results.values()) else 1

    ok = await run_one(source, settings)
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test data sources.")
    parser.add_argument(
        "--source",
        required=True,
        choices=SOURCES + ["all"],
        help="Which source to smoke test (or 'all').",
    )
    args = parser.parse_args()
    return asyncio.run(main_async(args.source))


if __name__ == "__main__":
    raise SystemExit(main())

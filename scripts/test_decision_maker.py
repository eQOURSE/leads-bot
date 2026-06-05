"""CLI test script for the Decision-Maker Finder (Phase 5).

Usage:
    python scripts/test_decision_maker.py --segment eqourse_ai_data --target 30
    python scripts/test_decision_maker.py --segment tutrain --target 30

Steps:
  1. Hunt companies (CompanyHunter)
  2. Qualify them (Qualifier)
  3. Find decision-makers (DecisionMakerFinder)
  4. Print rich table + summary
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


async def main_async(segment: str, target: int) -> int:
    from config.settings import get_settings
    from scripts.init_db import init_db

    settings = get_settings()
    init_db(settings.SQLITE_PATH)

    # --- Wire up all dependencies ---
    from agents.company_hunter import CompanyHunter
    from agents.decision_maker_finder import DecisionMakerFinder
    from agents.icp_strategist import IcpStrategist
    from agents.qualifier import Qualifier
    from agents._gemini_wrapper import GeminiAgent
    from sinks.sqlite_store import LeadStore
    from sources.apify_client import ApifyMultiKeyClient
    from sources.companies_api_client import CompaniesAPIClient
    from sources.newsdata_client import NewsDataClient
    from sources.rss_feeds import RSSFundingMonitor
    from sources.scrapegraph_client import ScrapeGraphClient
    from sources.serpapi_client import SerpAPIClient
    from sources.vibe_prospecting import VibeProspectingClient

    lead_store = LeadStore(settings)
    icp_strategist = IcpStrategist(settings)

    hunter = CompanyHunter(
        settings=settings,
        icp_strategist=icp_strategist,
        rss_client=RSSFundingMonitor(settings),
        serpapi_client=SerpAPIClient(settings),
        newsdata_client=NewsDataClient(settings),
        companies_api_client=CompaniesAPIClient(settings),
        lead_store=lead_store,
    )

    qualifier = Qualifier(
        settings=settings,
        icp_strategist=icp_strategist,
        gemini_agent=GeminiAgent(settings.GEMINI_MODEL_QUALIFIER, settings),
        companies_api_client=CompaniesAPIClient(settings),
        lead_store=lead_store,
    )

    dm_finder = DecisionMakerFinder(
        settings=settings,
        icp_strategist=icp_strategist,
        scrapegraph_client=ScrapeGraphClient(settings),
        apify_client=ApifyMultiKeyClient(settings),
        vibe_prospecting_client=VibeProspectingClient(settings),
        lead_store=lead_store,
    )

    # --- Step 1: Hunt ---
    print(f"\n[1/3] Hunting companies for segment={segment!r}, target={target}...")
    hunt_result = await hunter.hunt(segment, target_count=target, bypass_dedupe=True)
    print(f"      Found {len(hunt_result.candidates)} candidates")

    # --- Step 2: Qualify ---
    print("[2/3] Qualifying candidates...")
    qualified_result = await qualifier.qualify(hunt_result)
    t1 = sum(1 for q in qualified_result.qualified if q.tier == "tier_1")
    t2 = sum(1 for q in qualified_result.qualified if q.tier == "tier_2")
    print(f"      Qualified: tier_1={t1} tier_2={t2} (dropped={len(qualified_result.dropped)})")

    if not qualified_result.qualified:
        print("\nNo qualified candidates — nothing to find decision-makers for.")
        print("Segment may be too restrictive or no recent funding news found.")
        return 0

    # --- Step 3: Find decision-makers ---
    print("[3/3] Finding decision-makers...")
    enhanced = await dm_finder.find_for_qualified(qualified_result)

    # --- Step 4: Print results table ---
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title=f"Decision-Maker Results — {segment}", show_lines=True)
        table.add_column("Company", style="cyan", no_wrap=True)
        table.add_column("Tier", style="bold")
        table.add_column("Decision Makers", style="green")
        table.add_column("Source", style="yellow")
        table.add_column("Status", style="magenta")

        for cwp in enhanced.candidates_with_people:
            cand = cwp.qualified.candidate
            company_name = getattr(cand, "name", getattr(cand, "domain", "?"))
            tier = cwp.qualified.tier

            if cwp.decision_makers:
                for dm in cwp.decision_makers:
                    li = f"\n  [blue]{dm.linkedin_url}[/blue]" if dm.linkedin_url else ""
                    dm_text = f"{dm.full_name}\n{dm.title}{li}"
                    table.add_row(company_name, tier, dm_text, dm.source, cwp.lookup_status)
                    company_name = ""  # only show company name on first row
                    tier = ""
            else:
                table.add_row(company_name, tier, "—", "—", cwp.lookup_status)

        console.print(table)

        # Summary
        stats = enhanced.stats
        console.print(f"\n[bold]Summary[/bold]")
        console.print(f"  Qualified:           {len(enhanced.candidates_with_people)}")
        console.print(f"  DMs found:           {stats.get('total_dms_found', 0)}")
        console.print(f"  Needs manual lookup: {len(enhanced.needs_manual_lookup)}")
        console.print(f"  No DM found:         {sum(1 for c in enhanced.candidates_with_people if c.lookup_status == 'no_decision_maker')}")
        console.print(f"\n[bold]API Credits Used[/bold]")
        for src, count in enhanced.api_credits_used.items():
            console.print(f"  {src:20s}: {count}")

        if enhanced.needs_manual_lookup:
            console.print(f"\n[bold yellow]Needs Manual Lookup ({len(enhanced.needs_manual_lookup)})[/bold yellow]")
            for qc in enhanced.needs_manual_lookup:
                cand = qc.candidate
                name = getattr(cand, "name", getattr(cand, "domain", "?"))
                domain = getattr(cand, "domain", "")
                console.print(f"  - {name} ({domain})")

    except ImportError:
        # Fallback without rich
        print("\n=== RESULTS ===")
        for cwp in enhanced.candidates_with_people:
            cand = cwp.qualified.candidate
            name = getattr(cand, "name", "?")
            print(f"\n{name} [{cwp.qualified.tier}] → {cwp.lookup_status}")
            for dm in cwp.decision_makers:
                print(f"  - {dm.full_name}, {dm.title} ({dm.source})")
            if not cwp.decision_makers:
                print("  - No decision-makers found")

        print(f"\nSummary: qualified={len(enhanced.candidates_with_people)} "
              f"DMs={enhanced.stats.get('total_dms_found', 0)} "
              f"manual={len(enhanced.needs_manual_lookup)}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Test the Decision-Maker Finder.")
    parser.add_argument(
        "--segment",
        required=True,
        choices=["tutrain", "eqourse_content", "eqourse_ai_data"],
        help="ICP segment to target.",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=30,
        help="Number of companies to hunt.",
    )
    args = parser.parse_args()
    return asyncio.run(main_async(args.segment, args.target))


if __name__ == "__main__":
    raise SystemExit(main())

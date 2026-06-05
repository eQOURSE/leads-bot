"""CLI test script for the Contact Enricher (Phase 6).

Usage:
    python scripts/test_contact_enricher.py --segment eqourse_ai_data --target 30
    python scripts/test_contact_enricher.py --segment tutrain --target 30

Steps:
  1. Hunt companies (CompanyHunter)
  2. Qualify them (Qualifier)
  3. Find decision-makers (DecisionMakerFinder)
  4. Enrich contacts (ContactEnricher)
  5. Print rich tree-style output + summary table
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
    from agents._gemini_wrapper import GeminiAgent
    from agents.company_hunter import CompanyHunter
    from agents.contact_enricher import ContactEnricher
    from agents.decision_maker_finder import DecisionMakerFinder
    from agents.icp_strategist import IcpStrategist
    from agents.qualifier import Qualifier
    from sinks.sqlite_store import LeadStore
    from sources.abstract_api_client import AbstractAPIClient
    from sources.apify_client import ApifyMultiKeyClient
    from sources.companies_api_client import CompaniesAPIClient
    from sources.hunter_client import HunterClient
    from sources.newsdata_client import NewsDataClient
    from sources.rss_feeds import RSSFundingMonitor
    from sources.scrapegraph_client import ScrapeGraphClient
    from sources.serpapi_client import SerpAPIClient
    from sources.smtp_verifier import SMTPVerifier
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
    enricher = ContactEnricher(
        settings=settings,
        hunter_client=HunterClient(settings),
        abstract_api_client=AbstractAPIClient(settings),
        smtp_verifier=SMTPVerifier(settings),
        vibe_prospecting_client=VibeProspectingClient(settings),
        lead_store=lead_store,
    )

    # --- Step 1: Hunt ---
    print(f"\n[1/4] Hunting companies for segment={segment!r}, target={target}...")
    hunt_result = await hunter.hunt(segment, target_count=target, bypass_dedupe=True)
    print(f"      Found {len(hunt_result.candidates)} candidates")

    # --- Step 2: Qualify ---
    print("[2/4] Qualifying candidates...")
    qualified_result = await qualifier.qualify(hunt_result)
    t1 = sum(1 for q in qualified_result.qualified if q.tier == "tier_1")
    t2 = sum(1 for q in qualified_result.qualified if q.tier == "tier_2")
    print(f"      Qualified: tier_1={t1} tier_2={t2} (dropped={len(qualified_result.dropped)})")

    if not qualified_result.qualified:
        print("\nNo qualified candidates — nothing to enrich.")
        return 0

    # --- Step 3: Find decision-makers ---
    print("[3/4] Finding decision-makers...")
    enhanced = await dm_finder.find_for_qualified(qualified_result)
    total_dms = sum(len(c.decision_makers) for c in enhanced.candidates_with_people)
    print(f"      DMs found: {total_dms}, needs manual lookup: {len(enhanced.needs_manual_lookup)}")

    # --- Step 4: Enrich contacts ---
    print("[4/4] Enriching contact emails...")
    enriched = await enricher.enrich(enhanced)

    # --- Output ---
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.tree import Tree

        console = Console()
        tree = Tree(f"[bold]Enriched Results — {segment}[/bold]")

        for ec in enriched.enriched_candidates:
            cand = ec.candidate_with_people.qualified.candidate
            name = getattr(cand, "name", getattr(cand, "domain", "?"))
            tier = ec.candidate_with_people.qualified.tier
            status_color = {"full": "green", "partial": "yellow", "no_emails": "red", "skipped": "dim"}.get(
                ec.enrichment_status, "white"
            )

            company_node = tree.add(
                f"[cyan]{name}[/cyan]  [bold]{tier}[/bold]  "
                f"[{status_color}]{ec.enrichment_status}[/{status_color}]"
            )

            for edm in ec.enriched_dms:
                dm = edm.decision_maker
                dm_node = company_node.add(
                    f"[white]{dm.full_name}[/white] — [italic]{dm.title}[/italic]"
                )
                email = edm.email_result.email or "—"
                conf = f"{edm.email_result.confidence:.2f}" if edm.email_result.email else "—"
                src = edm.email_result.source if edm.email_result.email else "not_found"
                dm_node.add(f"Email: [green]{email}[/green]  [confidence: {conf}]  source: [yellow]{src}[/yellow]")
                if dm.linkedin_url:
                    dm_node.add(f"LinkedIn: [blue]{dm.linkedin_url}[/blue]")
                if edm.phone:
                    dm_node.add(f"Phone: {edm.phone}  (source: {edm.phone_source})")

            if ec.company_contact_email and ec.company_contact_email.email:
                ce = ec.company_contact_email
                company_node.add(
                    f"[dim]Company contact: {ce.email}  [confidence: {ce.confidence:.2f}]"
                    f"  source: {ce.source}[/dim]"
                )

        console.print(tree)

        # Summary table
        stats = enriched.stats
        console.print("\n[bold]Summary[/bold]")
        console.print(f"  Total DMs processed:  {stats.get('dms_total', 0)}")
        console.print(f"  Emails found:         {stats.get('emails_found', 0)}")
        console.print(f"  By source:")
        for src, cnt in (stats.get("by_source") or {}).items():
            console.print(f"    {src:<35} {cnt}")

        console.print(f"\n[bold]API Credits Used[/bold]")
        for src, cnt in enriched.api_credits_used.items():
            console.print(f"  {src:<25} {cnt}")

        dms_total = stats.get("dms_total", 0)
        emails_found = stats.get("emails_found", 0)
        pct = (emails_found / dms_total * 100) if dms_total else 0
        console.print(f"\n  Email coverage: {emails_found}/{dms_total} ({pct:.0f}%)")

    except ImportError:
        # Fallback without rich
        print("\n=== ENRICHED RESULTS ===")
        for ec in enriched.enriched_candidates:
            cand = ec.candidate_with_people.qualified.candidate
            name = getattr(cand, "name", "?")
            print(f"\n{name} [{ec.candidate_with_people.qualified.tier}] → {ec.enrichment_status}")
            for edm in ec.enriched_dms:
                dm = edm.decision_maker
                email = edm.email_result.email or "—"
                conf = f"{edm.email_result.confidence:.2f}"
                print(f"  {dm.full_name} ({dm.title})")
                print(f"    Email: {email}  confidence: {conf}  source: {edm.email_result.source}")
            if ec.company_contact_email and ec.company_contact_email.email:
                print(f"  Company: {ec.company_contact_email.email} (common_prefix)")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Test the Contact Enricher (Phase 6).")
    parser.add_argument(
        "--segment",
        required=True,
        choices=["tutrain", "eqourse_content", "eqourse_ai_data"],
    )
    parser.add_argument("--target", type=int, default=30)
    args = parser.parse_args()
    return asyncio.run(main_async(args.segment, args.target))


if __name__ == "__main__":
    raise SystemExit(main())

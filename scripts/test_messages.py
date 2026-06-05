"""CLI test script for the Personalizer + Message Writer (Phase 7).

Usage:
    python scripts/test_messages.py --segment eqourse_ai_data --target 30
    python scripts/test_messages.py --segment tutrain --target 30

Steps:
  1. Hunt companies
  2. Qualify
  3. Find decision-makers
  4. Enrich contacts
  5. Build personalization hooks
  6. Write messages
  7. Print rich tree + summary
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


async def main_async(segment: str, target: int) -> int:
    from config.settings import get_settings
    from scripts.init_db import init_db

    settings = get_settings()
    init_db(settings.SQLITE_PATH)

    from agents._gemini_wrapper import GeminiAgent
    from agents.company_hunter import CompanyHunter
    from agents.contact_enricher import ContactEnricher
    from agents.decision_maker_finder import DecisionMakerFinder
    from agents.icp_strategist import IcpStrategist
    from agents.message_writer import MessageWriter
    from agents.personalizer import Personalizer
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
    scrapegraph = ScrapeGraphClient(settings)
    newsdata = NewsDataClient(settings)

    hunter = CompanyHunter(
        settings=settings, icp_strategist=icp_strategist,
        rss_client=RSSFundingMonitor(settings),
        serpapi_client=SerpAPIClient(settings),
        newsdata_client=newsdata,
        companies_api_client=CompaniesAPIClient(settings),
        lead_store=lead_store,
    )
    qualifier = Qualifier(
        settings=settings, icp_strategist=icp_strategist,
        gemini_agent=GeminiAgent(settings.GEMINI_MODEL_QUALIFIER, settings),
        companies_api_client=CompaniesAPIClient(settings),
        lead_store=lead_store,
    )
    dm_finder = DecisionMakerFinder(
        settings=settings, icp_strategist=icp_strategist,
        scrapegraph_client=scrapegraph,
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
    personalizer = Personalizer(
        settings=settings, icp_strategist=icp_strategist,
        gemini_agent=GeminiAgent(settings.GEMINI_MODEL_PERSONALIZER, settings),
        scrapegraph_client=scrapegraph,
        newsdata_client=newsdata,
        lead_store=lead_store,
    )
    writer = MessageWriter(
        settings=settings, icp_strategist=icp_strategist,
        gemini_agent=GeminiAgent(settings.GEMINI_MODEL_WRITER, settings),
        lead_store=lead_store,
    )

    print(f"\n[1/6] Hunting for segment={segment!r}, target={target}...")
    hunt_result = await hunter.hunt(segment, target_count=target, bypass_dedupe=True)
    print(f"      {len(hunt_result.candidates)} candidates found")

    print("[2/6] Qualifying...")
    qualified_result = await qualifier.qualify(hunt_result)
    t1 = sum(1 for q in qualified_result.qualified if q.tier == "tier_1")
    t2 = sum(1 for q in qualified_result.qualified if q.tier == "tier_2")
    print(f"      tier_1={t1} tier_2={t2} dropped={len(qualified_result.dropped)}")

    if not qualified_result.qualified:
        print("\nNo qualified candidates.")
        return 0

    print("[3/6] Finding decision-makers...")
    enhanced = await dm_finder.find_for_qualified(qualified_result)
    total_dms = sum(len(c.decision_makers) for c in enhanced.candidates_with_people)
    print(f"      DMs found={total_dms}")

    print("[4/6] Enriching contacts...")
    enriched = await enricher.enrich(enhanced)
    emails = enriched.stats.get("emails_found", 0)
    print(f"      Emails found={emails}/{enriched.stats.get('dms_total', 0)}")

    print("[5/6] Building personalization hooks...")
    p_map = await personalizer.build_hooks_for_enriched_result(enriched)
    print(f"      Hooks built={len(p_map)}")

    print("[6/6] Writing messages...")
    messaged = await writer.write_for_enriched(enriched, p_map)
    generated = messaged.stats.get("messages_generated", 0)
    print(f"      Messages generated={generated}")

    # Display
    try:
        from rich.console import Console
        from rich.tree import Tree
        from rich.panel import Panel

        console = Console()

        for mc in messaged.messaged_candidates:
            ec = mc.enriched_candidate
            qc = ec.candidate_with_people.qualified
            cand = qc.candidate
            name = getattr(cand, "name", getattr(cand, "domain", "?"))
            domain = getattr(cand, "domain", "")
            tier = qc.tier

            tree = Tree(f"[cyan bold]{name}[/cyan bold]  [{tier}]  [dim]{domain}[/dim]")

            if mc.personalization:
                hook_color = {"high": "green", "medium": "yellow", "low": "red"}.get(
                    mc.personalization.personalization_quality, "white"
                )
                tree.add(
                    f"Hook [{hook_color}]{mc.personalization.personalization_quality}[/{hook_color}]: "
                    f"[italic]{mc.personalization.why_now_hook}[/italic]"
                )

            for mdm in mc.messaged_dms:
                dm = mdm.enriched_dm.decision_maker
                email = mdm.enriched_dm.email_result.email or "—"
                conf = mdm.enriched_dm.email_result.confidence

                dm_node = tree.add(
                    f"[white]{dm.full_name}[/white] — [italic]{dm.title}[/italic]"
                )
                dm_node.add(f"To: [green]{email}[/green]  [confidence: {conf:.2f}]")

                if mdm.messages:
                    m = mdm.messages
                    dm_node.add(f"Subject A: [yellow]{m.email_subject_a}[/yellow]")
                    dm_node.add(f"Subject B: [yellow]{m.email_subject_b}[/yellow]")
                    dm_node.add(f"Reply likelihood: [bold]{m.reply_likelihood}/10[/bold]")
                    if m.quality_flags:
                        dm_node.add(f"Quality flags: [red]{m.quality_flags}[/red]")
                    body_node = dm_node.add("Email body:")
                    for line in m.email_body.split("\n"):
                        if line.strip():
                            body_node.add(f"  {line}")
                    dm_node.add(f"LinkedIn DM: [dim]{m.linkedin_dm}[/dim]")
                elif mdm.skipped_reason:
                    dm_node.add(f"[dim]Skipped: {mdm.skipped_reason}[/dim]")

            console.print(tree)
            console.print()

        # Summary
        all_likelihoods = [
            mdm.messages.reply_likelihood
            for mc in messaged.messaged_candidates
            for mdm in mc.messaged_dms
            if mdm.messages
        ]
        all_flags = [
            f for mc in messaged.messaged_candidates
            for mdm in mc.messaged_dms
            if mdm.messages
            for f in mdm.messages.quality_flags
        ]
        flag_freq = Counter(all_flags)

        console.print(Panel.fit(
            f"[bold]Summary[/bold]\n"
            f"  Messages generated:     {messaged.stats.get('messages_generated', 0)}\n"
            f"  Skipped (no email):     {messaged.stats.get('skipped_no_email', 0)}\n"
            f"  Avg reply likelihood:   {messaged.stats.get('avg_reply_likelihood', 0):.2f}/10\n"
            f"  Avg quality flags:      {messaged.stats.get('avg_quality_flags', 0):.2f}\n"
            + (f"  Top flags:              {dict(flag_freq.most_common(5))}\n" if flag_freq else "")
            + "\n[bold]API Credits[/bold]\n"
            + "\n".join(f"  {k:<25} {v}" for k, v in messaged.api_credits_used.items()),
            title="Phase 7 Results"
        ))

    except ImportError:
        print("\n=== RESULTS (no rich) ===")
        for mc in messaged.messaged_candidates:
            cand = mc.enriched_candidate.candidate_with_people.qualified.candidate
            name = getattr(cand, "name", "?")
            for mdm in mc.messaged_dms:
                if mdm.messages:
                    m = mdm.messages
                    print(f"\n{name}")
                    print(f"  Subject A: {m.email_subject_a}")
                    print(f"  Subject B: {m.email_subject_b}")
                    print(f"  Likelihood: {m.reply_likelihood}/10")
                    print(f"  Flags: {m.quality_flags}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Test the Message Writer (Phase 7).")
    parser.add_argument(
        "--segment", required=True,
        choices=["tutrain", "eqourse_content", "eqourse_ai_data"],
    )
    parser.add_argument("--target", type=int, default=30)
    args = parser.parse_args()
    return asyncio.run(main_async(args.segment, args.target))


if __name__ == "__main__":
    raise SystemExit(main())

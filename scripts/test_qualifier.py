"""Live smoke test for the Qualifier agent.

Usage:
    python scripts/test_qualifier.py --segment tutrain --target 30
    python scripts/test_qualifier.py --segment eqourse_ai_data --target 30

Runs a full live hunt then qualifies the results, printing:
  - Pre-score distribution histogram
  - Qualified candidates rich table
  - Dropped summary
  - API credits and tuning hints
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rich.console import Console
from rich.table import Table

from agents._gemini_wrapper import GeminiAgent
from agents.company_hunter import CompanyHunter
from agents.icp_strategist import IcpStrategist
from agents.qualifier import Qualifier
from config.settings import get_settings
from sinks.sqlite_store import LeadStore
from sources.companies_api_client import CompaniesAPIClient
from sources.newsdata_client import NewsDataClient
from sources.rss_feeds import RSSFundingMonitor
from sources.serpapi_client import SerpAPIClient

console = Console(legacy_windows=False)


def _build_pipeline(settings):
    strategist = IcpStrategist(settings)
    lead_store = LeadStore(settings)

    hunter = CompanyHunter(
        settings=settings,
        icp_strategist=strategist,
        rss_client=RSSFundingMonitor(settings),
        serpapi_client=SerpAPIClient(settings),
        newsdata_client=NewsDataClient(settings),
        companies_api_client=CompaniesAPIClient(settings),
        lead_store=lead_store,
    )

    gemini = GeminiAgent(settings.GEMINI_MODEL_QUALIFIER, settings)
    qualifier = Qualifier(
        settings=settings,
        icp_strategist=strategist,
        gemini_agent=gemini,
        companies_api_client=CompaniesAPIClient(settings),
        lead_store=lead_store,
    )
    return hunter, qualifier


def _histogram(scores: list[int]) -> str:
    buckets = {"<40": 0, "40-49": 0, "50-59": 0, "60-69": 0, "70-79": 0, "80-89": 0, "90+": 0}
    for s in scores:
        if s < 40:
            buckets["<40"] += 1
        elif s < 50:
            buckets["40-49"] += 1
        elif s < 60:
            buckets["50-59"] += 1
        elif s < 70:
            buckets["60-69"] += 1
        elif s < 80:
            buckets["70-79"] += 1
        elif s < 90:
            buckets["80-89"] += 1
        else:
            buckets["90+"] += 1
    bars = []
    for label, count in buckets.items():
        bar = "#" * count
        bars.append(f"  {label:6s} | {bar:<20s} {count}")
    return "\n".join(bars)


async def run(segment: str, target: int, enrichment_top: int) -> int:
    settings = get_settings()
    hunter, qualifier = _build_pipeline(settings)

    console.print(
        f"\n[bold cyan]Pipeline[/bold cyan] | segment=[bold]{segment}[/bold] "
        f"| target={target}\n"
    )

    # Stage 1: Hunt
    console.print("[bold yellow]Stage 1: Hunting...[/bold yellow]")
    hunt_result = await hunter.hunt(
        segment=segment,
        target_count=target,
        enrichment_top_n=enrichment_top,
        bypass_dedupe=True,  # bypass for testing so we always get candidates
    )
    console.print(
        f"  Hunt complete: {len(hunt_result.candidates)} candidates "
        f"({hunt_result.source_counts}) in {hunt_result.duration_seconds:.1f}s\n"
    )

    if not hunt_result.candidates:
        console.print("[yellow]No candidates from hunt — nothing to qualify.[/yellow]")
        return 1

    # Stage 2: Qualify
    console.print("[bold yellow]Stage 2: Qualifying...[/bold yellow]")
    result = await qualifier.qualify(hunt_result, domain_resolution_cap=5)

    # --- Pre-score histogram ---
    console.print("\n[bold yellow]Pre-score distribution[/bold yellow]")
    all_scores = (
        [q.pre_score for q in result.qualified]
        + [d.get("total_score", 0) for d in result.dropped]
    )
    console.print(_histogram(all_scores))

    # --- Stats ---
    console.print("\n[bold yellow]Qualification stats[/bold yellow]")
    stats_rows = [
        ("Input candidates", str(len(hunt_result.candidates))),
        ("Pre-score filtered", str(result.stats.get("pre_score_filtered", "?"))),
        ("Gemini calls", str(result.stats.get("gemini_calls", "?"))),
        ("Domains resolved", str(result.stats.get("domains_resolved", 0))),
        ("Tier 1 qualified", str(result.stats.get("tier_1_count", 0))),
        ("Tier 2 qualified", str(result.stats.get("tier_2_count", 0))),
        ("Total dropped", str(len(result.dropped))),
        ("Duration", f"{result.duration_seconds:.1f}s"),
    ]
    for label, value in stats_rows:
        console.print(f"  [cyan]{label:22s}[/cyan] {value}")

    # --- Credits ---
    console.print("\n[bold yellow]API credits used[/bold yellow]")
    for source, credits in result.api_credits_used.items():
        console.print(f"  [cyan]{source:20s}[/cyan] {credits}")

    # --- Qualified table ---
    if result.qualified:
        console.print(f"\n[bold yellow]Qualified candidates ({len(result.qualified)})[/bold yellow]")
        table = Table(show_header=True, header_style="bold magenta", expand=False)
        table.add_column("Tier", width=6)
        table.add_column("Score", width=6)
        table.add_column("Name", min_width=18)
        table.add_column("Domain", min_width=20)
        table.add_column("Reasoning", min_width=30)
        table.add_column("Disqualifiers", min_width=10)

        for qc in result.qualified:
            c = qc.candidate
            tier_style = "bold green" if qc.tier == "tier_1" else "yellow"
            table.add_row(
                f"[{tier_style}]{qc.tier}[/{tier_style}]",
                str(qc.total_score),
                c.name[:22],  # type: ignore[union-attr]
                c.domain[:25],  # type: ignore[union-attr]
                qc.reasoning[:45] + ("..." if len(qc.reasoning) > 45 else ""),
                ", ".join(qc.disqualifiers) or "-",
            )
        console.print(table)
    else:
        console.print("\n[red]No candidates qualified.[/red]")

    # --- Dropped summary ---
    if result.dropped:
        console.print(f"\n[bold yellow]Dropped ({len(result.dropped)})[/bold yellow]")
        reason_counter: Counter = Counter(d["drop_reason"] for d in result.dropped)
        for reason, count in reason_counter.most_common(5):
            console.print(f"  [red]*[/red] {reason[:70]} ({count}x)")

    # --- Tuning hints ---
    total_in = len(hunt_result.candidates)
    total_qualified = len(result.qualified)
    console.print("\n[bold yellow]Tuning hints[/bold yellow]")
    if total_in > 0:
        qual_pct = total_qualified / total_in * 100
        if qual_pct > 80:
            console.print(
                "  [yellow]![/yellow] 80%+ of candidates qualified — ICP may be too loose. "
                "Consider raising tier_2_above threshold."
            )
        elif total_qualified == 0:
            # Identify the main drop reason
            reasons = Counter(d["drop_reason"] for d in result.dropped)
            top_reason, _ = reasons.most_common(1)[0] if reasons else ("unknown", 0)
            if "funding_recency" in top_reason or "pre_score" in top_reason:
                console.print(
                    "  [yellow]![/yellow] 0 qualified. Most dropped on pre-score. "
                    "Hunter may be finding old funding — check RSS date parsing."
                )
            elif "domain_unresolved" in top_reason:
                console.print(
                    "  [yellow]![/yellow] 0 qualified. Most dropped due to unresolved domains. "
                    "Raise domain_resolution_cap or improve domain extraction."
                )
            elif "negative_signal" in top_reason:
                console.print(
                    "  [yellow]![/yellow] 0 qualified. Most dropped on negative signal match. "
                    "Review negative_signals in icp_configs.json."
                )
            else:
                console.print(
                    "  [yellow]![/yellow] 0 qualified. ICP may be too strict. "
                    "Consider lowering auto_drop_below or relaxing negative_signals."
                )
        else:
            console.print(f"  [green]OK[/green] {qual_pct:.0f}% qualification rate — looks healthy.")

    if result.stats.get("domains_resolved", 0) == 5:
        console.print(
            "  [yellow]![/yellow] Hit domain resolution cap (5). "
            "Increase domain_resolution_cap if more unknowns need resolving."
        )

    console.print()
    return 0 if total_qualified >= 1 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Live smoke test for Qualifier.")
    parser.add_argument(
        "--segment",
        required=True,
        choices=["tutrain", "eqourse_content", "eqourse_ai_data"],
    )
    parser.add_argument("--target", type=int, default=30)
    parser.add_argument("--enrichment-top", type=int, default=3)
    args = parser.parse_args()
    return asyncio.run(run(args.segment, args.target, args.enrichment_top))


if __name__ == "__main__":
    raise SystemExit(main())

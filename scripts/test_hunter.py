"""Live smoke test for the CompanyHunter agent.

Usage:
    python scripts/test_hunter.py --segment tutrain --target 30 --enrichment-top 5
    python scripts/test_hunter.py --segment eqourse_ai_data --target 30
    python scripts/test_hunter.py --segment tutrain --bypass-dedupe

Runs a full live hunt against real APIs and prints a summary + rich table of
the top 10 candidates.  Exits 0 on success (>=1 candidate), 1 on failure.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
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

from agents.company_hunter import CompanyHunter
from agents.icp_strategist import IcpStrategist
from config.settings import get_settings
from sinks.sqlite_store import LeadStore
from sources.companies_api_client import CompaniesAPIClient
from sources.newsdata_client import NewsDataClient
from sources.rss_feeds import RSSFundingMonitor
from sources.serpapi_client import SerpAPIClient

console = Console(legacy_windows=False)


def _build_hunter(settings) -> CompanyHunter:
    return CompanyHunter(
        settings=settings,
        icp_strategist=IcpStrategist(settings),
        rss_client=RSSFundingMonitor(settings),
        serpapi_client=SerpAPIClient(settings),
        newsdata_client=NewsDataClient(settings),
        companies_api_client=CompaniesAPIClient(settings),
        lead_store=LeadStore(settings),
    )


async def run(
    segment: str,
    target: int,
    enrichment_top: int,
    bypass_dedupe: bool,
) -> int:
    settings = get_settings()
    hunter = _build_hunter(settings)

    console.print(
        f"\n[bold cyan]CompanyHunter[/bold cyan] | segment=[bold]{segment}[/bold] "
        f"| target={target} | enrichment_top={enrichment_top} "
        f"| bypass_dedupe={bypass_dedupe}\n"
    )

    result = await hunter.hunt(
        segment=segment,
        target_count=target,
        enrichment_top_n=enrichment_top,
        bypass_dedupe=bypass_dedupe,
    )

    # --- Stage summary ---
    console.print("[bold yellow]Stage summary[/bold yellow]")
    rows = [
        ("Sources", f"RSS={result.source_counts['rss']}  SerpAPI={result.source_counts['serpapi']}  NewsData={result.source_counts['newsdata']}"),
        ("Merged", str(result.merged_count)),
        ("After ICP filter", str(result.after_filter)),
        ("After dedupe", str(result.after_dedupe)),
        ("Enriched", str(result.enriched_count)),
        ("Final candidates", str(len(result.candidates))),
        ("Duration", f"{result.duration_seconds:.1f}s"),
        ("Run ID", result.run_id),
    ]
    for label, value in rows:
        console.print(f"  [cyan]{label:20s}[/cyan] {value}")

    # --- API credits ---
    console.print("\n[bold yellow]API credits consumed[/bold yellow]")
    for source, credits in result.api_credits_used.items():
        console.print(f"  [cyan]{source:20s}[/cyan] {credits}")

    if result.errors:
        console.print("\n[bold red]Errors[/bold red]")
        for err in result.errors:
            console.print(f"  [red]•[/red] {err}")

    # --- Top 10 table ---
    top = result.candidates[:10]
    if top:
        console.print(f"\n[bold yellow]Top {len(top)} candidates[/bold yellow]")
        table = Table(show_header=True, header_style="bold magenta", expand=False)
        table.add_column("#", width=3)
        table.add_column("Domain", min_width=20)
        table.add_column("Name", min_width=18)
        table.add_column("Funding", min_width=10)
        table.add_column("Stage", min_width=10)
        table.add_column("Date", min_width=12)
        table.add_column("Conf", width=6)
        table.add_column("Sources", min_width=14)

        for i, c in enumerate(top, start=1):
            funding = (
                f"${c.funding_amount_usd/1e6:.1f}M"
                if c.funding_amount_usd
                else "-"
            )
            table.add_row(
                str(i),
                c.domain[:28],
                c.name[:22],
                funding,
                c.funding_stage or "-",
                str(c.funding_date) if c.funding_date else "-",
                f"{c.confidence:.2f}",
                c.raw_source[:14],
            )
        console.print(table)
    else:
        console.print("\n[yellow]No candidates found.[/yellow]")

    console.print()
    return 0 if len(result.candidates) >= 1 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Live smoke test for CompanyHunter.")
    parser.add_argument(
        "--segment",
        required=True,
        choices=["tutrain", "eqourse_content", "eqourse_ai_data"],
    )
    parser.add_argument("--target", type=int, default=30)
    parser.add_argument("--enrichment-top", type=int, default=5)
    parser.add_argument("--bypass-dedupe", action="store_true")
    args = parser.parse_args()

    return asyncio.run(
        run(
            segment=args.segment,
            target=args.target,
            enrichment_top=args.enrichment_top,
            bypass_dedupe=args.bypass_dedupe,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())

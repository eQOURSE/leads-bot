"""Inspect ICP strategies in a clean, section-formatted terminal view.

Usage:
    python scripts/show_icp.py --segment tutrain
    python scripts/show_icp.py --segment all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Ensure UTF-8 stdout so output survives redirection on Windows (cp1252) shells.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agents._models import IcpStrategy
from agents.icp_strategist import IcpStrategist
from config.settings import get_settings

# Force UTF-8 so redirected / non-UTF8 Windows stdout doesn't crash on symbols.
console = Console(legacy_windows=False)


def _kv_table(rows: list[tuple[str, str]]) -> Table:
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("k", style="cyan", no_wrap=True)
    table.add_column("v", style="white")
    for k, v in rows:
        table.add_row(k, v)
    return table


def _bullets(items: list[str], style: str = "white") -> str:
    return "\n".join(f"[{style}]•[/{style}] {i}" for i in items)


def render_strategy(strategy: IcpStrategy) -> None:
    ind = strategy.target_industries
    prof = strategy.target_company_profile
    weights = strategy.scoring_weights
    thresh = strategy.scoring_thresholds
    angle = strategy.outreach_angle

    console.print(
        Panel(
            f"[bold]{strategy.value_prop_one_liner}[/bold]\n\n{strategy.what_we_offer}",
            title=f"[bold green]{strategy.segment_name}[/bold green]",
            border_style="green",
        )
    )

    console.print("\n[bold yellow]Target Industries[/bold yellow]")
    console.print(
        _kv_table(
            [
                ("NAICS codes", ", ".join(ind.naics_codes)),
                ("LinkedIn", ", ".join(ind.linkedin_categories)),
                ("Keywords", ", ".join(ind.industry_keywords)),
            ]
        )
    )

    console.print("\n[bold yellow]Company Profile[/bold yellow]")
    console.print(
        _kv_table(
            [
                ("Size ranges", ", ".join(prof.size_ranges)),
                ("Revenue ranges", ", ".join(prof.revenue_ranges) or "—"),
                ("Funding stages", ", ".join(prof.funding_stages)),
                ("Funding recency", f"{prof.funding_recency_days} days"),
                ("Founded after", str(prof.founded_after_year)),
                (
                    "Geographies",
                    ", ".join(prof.geographies.countries) or "any",
                ),
            ]
        )
    )

    console.print("\n[bold yellow]Decision Makers[/bold yellow]")
    console.print(
        _kv_table(
            [
                ("Titles", ", ".join(strategy.target_titles)),
                ("Departments", ", ".join(strategy.target_departments)),
                ("Levels", ", ".join(strategy.target_levels)),
            ]
        )
    )

    console.print("\n[bold green]Positive Signals[/bold green]")
    console.print(_bullets(strategy.positive_signals, "green"))

    console.print("\n[bold red]Negative Signals[/bold red]")
    console.print(_bullets(strategy.negative_signals, "red"))

    console.print("\n[bold yellow]Scoring[/bold yellow]")
    console.print(
        _kv_table(
            [
                ("Funding recency", f"{weights.funding_recency}"),
                ("Segment fit", f"{weights.segment_fit}"),
                ("Buying signal", f"{weights.buying_signal}"),
                ("Reachability", f"{weights.reachability}"),
                (
                    "Thresholds",
                    f"drop <{thresh.auto_drop_below}  |  "
                    f"tier-2 >={thresh.tier_2_above}  |  tier-1 >={thresh.tier_1_above}",
                ),
            ]
        )
    )

    console.print("\n[bold yellow]Outreach Angle[/bold yellow]")
    console.print(
        _kv_table(
            [
                ("Pain", angle.pain_hypothesis),
                ("Value framing", angle.value_framing),
                ("Primary CTA", angle.primary_cta),
                ("Fallback CTA", angle.fallback_cta),
            ]
        )
    )
    console.print("\n" + "─" * 70 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Show ICP strategies.")
    strategist = IcpStrategist(get_settings())
    parser.add_argument(
        "--segment",
        required=True,
        choices=strategist.list_segments() + ["all"],
        help="Segment to display (or 'all').",
    )
    args = parser.parse_args()

    segments = strategist.list_segments() if args.segment == "all" else [args.segment]

    try:
        for seg in segments:
            render_strategy(strategist.load_strategy(seg))
    except ValueError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

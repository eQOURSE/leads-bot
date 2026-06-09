"""Phase 9 — CLI entry point for the lead generation pipeline.

Usage:
    python scripts/run_pipeline.py --segment tutrain --target 30
    python scripts/run_pipeline.py --all-segments --target 30
    python scripts/run_pipeline.py --resume <run_id>
    python scripts/run_pipeline.py --all-segments --dry-run
    python scripts/run_pipeline.py --segment eqourse_ai_data --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Make project root importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.settings import get_settings
from orchestrator.runner import PipelineRunner
from orchestrator.state import PipelineState


# ---------------------------------------------------------------------------
# Console factory — UTF-8 safe on Windows
# ---------------------------------------------------------------------------

def _make_console():
    """Return a rich Console configured to render safely on Windows.

    Forcing ``legacy_windows=False`` makes rich emit ANSI escapes through the
    (UTF-8 reconfigured) stdout instead of the cp1252 Win32 console API, which
    cannot encode glyphs like the arrow, check-mark, and em-dash used in output.
    """
    from rich.console import Console
    return Console(legacy_windows=False, safe_box=True)


# ---------------------------------------------------------------------------
# Rich table helpers (rich is in requirements.txt)
# ---------------------------------------------------------------------------

def _print_segment_summary(segment: str, state: PipelineState) -> None:
    try:
        from rich.table import Table
        from rich import box

        console = _make_console()

        hunt = state.get("hunt_result")
        qual = state.get("qualified_result")
        enh = state.get("enhanced_result")
        enr = state.get("enriched_result")
        msg = state.get("messaged_result")
        val = state.get("validated_result")

        table = Table(
            title=f"[bold cyan]{segment}[/bold cyan] Pipeline Summary",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Stage", style="cyan", no_wrap=True)
        table.add_column("Result", justify="right")
        table.add_column("Status", justify="center")

        def row(label: str, value, ok: bool = True):
            status = "[green]✓[/green]" if ok else "[yellow]⚠[/yellow]"
            table.add_row(label, str(value), status)

        row("Hunt — candidates", len(hunt.candidates) if hunt else 0, hunt is not None)
        row("Qualify — qualified", len(qual.qualified) if qual else 0, qual is not None)
        if enh:
            total_dms = sum(len(c.decision_makers) for c in enh.candidates_with_people)
            row("Find DMs — found", f"{total_dms} DMs", enh is not None)
        else:
            row("Find DMs", "skipped", False)
        row("Enrich — emails", (enr.stats.get("emails_found", 0) if enr else 0), enr is not None)
        row("Messages — generated", (msg.stats.get("messages_generated", 0) if msg else 0), msg is not None)

        if val:
            ready = val.stats.get("ready_to_send", 0)
            review = val.stats.get("needs_review", 0)
            rejected = val.stats.get("rejected", 0)
            row("Validate — ready_to_send", ready, ready > 0)
            row("Validate — needs_review", review, True)
            row("Validate — rejected", rejected, True)
        else:
            row("Validate", "skipped", False)

        # Node errors
        for node, err in (state.get("node_errors") or {}).items():
            table.add_row(
                f"[red]ERROR — {node}[/red]",
                err[:60] + ("…" if len(err) > 60 else ""),
                "[red]✗[/red]",
            )

        final_status = state.get("final_status", "unknown")
        duration = state.get("duration_seconds")
        duration_str = f"{duration:.1f}s" if duration else "N/A"

        status_colour = {
            "success": "green",
            "partial_success": "yellow",
            "failed": "red",
        }.get(final_status, "white")

        console.print(table)
        console.print(
            f"Status: [{status_colour}]{final_status}[/{status_colour}]  "
            f"Duration: {duration_str}"
        )
        console.print()

    except ImportError:
        # Fallback without rich
        print(f"\n=== {segment} ===")
        print(f"Status: {state.get('final_status', 'unknown')}")
        if state.get("node_errors"):
            print(f"Errors: {state.get('node_errors')}")


def _print_funnel_metrics(segment: str, state) -> None:
    """Phase 11 — print the funnel drop-off + source contributions for a segment."""
    try:
        from orchestrator.nodes import compute_funnel_metrics
        m = compute_funnel_metrics(state)
    except Exception:  # noqa: BLE001
        return

    print(f"\n  Funnel [{segment}]:")
    for stage, n in m["funnel_drop_off"].items():
        print(f"    {stage:24s}: {n}")
    sc = m.get("source_contributions", {})
    if sc:
        print("  Source contributions:")
        for src, n in sc.items():
            print(f"    {src:12s}: {n}")
    print(f"  article_link_resolution_rate: {m.get('article_link_resolution_rate', 0):.2f}")
    print(f"  gemini_retry_count: {m.get('gemini_retry_count', 0)}  "
          f"gemini_fallback_count: {m.get('gemini_fallback_count', 0)}")
    print(f"  apify_spend_estimate_usd: ${m.get('apify_spend_estimate_usd', 0):.2f}")


def _print_dropped_diagnostics(segment: str, state) -> None:
    """If 0 qualified, print the dropped list with reasons sorted by frequency."""
    qual = state.get("qualified_result")
    if qual is None:
        return
    dropped = getattr(qual, "dropped", []) or []
    if not dropped:
        return
    from collections import Counter
    reasons = Counter()
    for d in dropped:
        reason = (d.get("drop_reason") or "unknown") if isinstance(d, dict) else "unknown"
        # Normalize "pre_score N < threshold" → "pre_score below threshold"
        if reason.startswith("pre_score"):
            reason = "pre_score below threshold (40)"
        elif reason.startswith("score") and "auto_drop" in reason:
            reason = "gemini score below auto_drop (70)"
        reasons[reason] += 1
    print(f"\n  [{segment}] 0 qualified — dropped reasons (most frequent first):")
    for reason, count in reasons.most_common():
        print(f"    {count:3d}  {reason}")


def _print_consolidated_stats(
    results: dict[str, PipelineState], sheets_url: str = ""
) -> None:
    try:
        from rich.panel import Panel

        console = _make_console()

        total_candidates = 0
        total_qualified = 0
        total_ready = 0

        lines = []
        for seg, state in results.items():
            hunt = state.get("hunt_result")
            qual = state.get("qualified_result")
            val = state.get("validated_result")

            c = len(hunt.candidates) if hunt else 0
            q = len(qual.qualified) if qual else 0
            r = val.stats.get("ready_to_send", 0) if val else 0

            total_candidates += c
            total_qualified += q
            total_ready += r

            fs = state.get("final_status", "unknown")
            colour = {"success": "green", "partial_success": "yellow", "failed": "red"}.get(fs, "white")
            lines.append(
                f"  [{colour}]{seg}[/{colour}]: {c} hunted → {q} qualified → {r} ready_to_send"
            )

        summary = "\n".join(lines)
        summary += (
            f"\n\n  [bold]Total: {total_candidates} candidates → "
            f"{total_qualified} qualified → {total_ready} ready to send[/bold]"
        )
        if sheets_url:
            summary += f"\n\n  📋 [link={sheets_url}]Open in Google Sheets[/link]"

        console.print(Panel(summary, title="[bold]Consolidated Pipeline Stats[/bold]", expand=False))

    except ImportError:
        print("\n=== Consolidated Stats ===")
        for seg, state in results.items():
            print(f"  {seg}: {state.get('final_status')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lead Generation Pipeline — Phase 9 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single segment
  python scripts/run_pipeline.py --segment tutrain --target 30

  # Run all three segments concurrently
  python scripts/run_pipeline.py --all-segments --target 30

  # Resume from a checkpoint
  python scripts/run_pipeline.py --resume tutrain_20250604_013000

  # Dry run (skip dispatch)
  python scripts/run_pipeline.py --all-segments --dry-run --target 5
""",
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--segment", metavar="NAME",
                            help="Run a single segment (tutrain | eqourse_content | eqourse_ai_data)")
    mode_group.add_argument("--all-segments", action="store_true",
                            help="Run all three segments concurrently")
    mode_group.add_argument("--resume", metavar="RUN_ID",
                            help="Resume a segment from its checkpoint run_id")

    parser.add_argument("--target", type=int, default=30,
                        help="target_count per segment (default: 30)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run graph but skip dispatch node (no writes, no Telegram)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable DEBUG logging")

    return parser.parse_args(argv)


async def _async_main(args: argparse.Namespace) -> int:
    """Async entrypoint. Returns exit code."""
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    settings = get_settings()

    try:
        console = _make_console()
        console.print(f"[bold blue]Lead Generation Pipeline — Phase 9[/bold blue]")
        console.print(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    except ImportError:
        print("Lead Generation Pipeline — Phase 9")

    t_start = time.monotonic()

    async with PipelineRunner(settings, dry_run=args.dry_run) as runner:

        # ---- Single segment ----
        if args.segment:
            print(f"Running segment: {args.segment} (target={args.target})")
            state = await runner.run_segment(args.segment, target_count=args.target)
            _print_segment_summary(args.segment, state)
            _print_funnel_metrics(args.segment, state)
            _print_dropped_diagnostics(args.segment, state)
            _print_consolidated_stats({args.segment: state})
            return 0 if state.get("final_status") in ("success", "partial_success") else 1

        # ---- Resume ----
        if args.resume:
            print(f"Resuming run_id: {args.resume}")
            state = await runner.resume_segment(args.resume, target_count=args.target)
            segment = state.get("segment", args.resume)
            _print_segment_summary(segment, state)
            return 0 if state.get("final_status") in ("success", "partial_success") else 1

        # ---- All segments ----
        if args.all_segments:
            if args.dry_run:
                print("[DRY-RUN] Running all segments (dispatch skipped)…")
            else:
                print(f"Running all segments concurrently (target={args.target} each)…")

            results = await runner.run_all_segments(target_count=args.target)

            for seg, state in results.items():
                _print_segment_summary(seg, state)
                _print_funnel_metrics(seg, state)
                _print_dropped_diagnostics(seg, state)

            sheets_url = ""
            if settings.SHEET_ID:
                sheets_url = f"https://docs.google.com/spreadsheets/d/{settings.SHEET_ID}/edit"

            _print_consolidated_stats(results, sheets_url)

            if args.dry_run:
                print("[DRY-RUN] Telegram digest skipped")
            elif any(s.get("validated_result") for s in results.values()):
                print("Telegram digest sent (see logs for message_id)")

            if sheets_url:
                print(f"\nGoogle Sheet: {sheets_url}")

            elapsed = time.monotonic() - t_start
            print(f"\nTotal wall time: {elapsed:.1f}s")

            failed = [s for s, r in results.items() if r.get("final_status") == "failed"]
            return 1 if len(failed) == len(results) else 0

    return 0


def main() -> None:
    args = parse_args()
    exit_code = asyncio.run(_async_main(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

# Post-Launch Backlog

The system is in production (Phases 0–10 complete, daily cron live, dashboard
deployed). These are known issues and improvements to address during the
operating phase — none are launch blockers.

---

## 1. RSS / SerpAPI extraction resilience to Gemini 503 (HIGH)

**Observed:** June 9, 2026 cron run — 0 hunt across all three segments, despite
RSS feeds reliably returning 26 items the previous day.

**Root cause:** `_hunt_via_rss` and `_hunt_via_serpapi` (agents/company_hunter.py)
use Gemini Flash-Lite to extract company name / funding data from each headline.
When Gemini returns 503 (`UNAVAILABLE` — high demand on the free-tier model) for
every item in the batch, `batch_generate_json` returns all `None`, so the hunt
collapses to 0 candidates. The empty-run digest then correctly reports 0/0/0.

**Why it's not urgent:** The 04:59 UTC (10:29 AM IST) cron hits Gemini at a
quieter time than the midday manual trigger that surfaced this. Most days will
self-heal. The pipeline degrades gracefully (no crash, audit row written).

**Fix options (pick one):**
- **A — Retry with backoff:** wrap the per-item extraction in a tenacity retry
  (2–3 attempts, exponential backoff) before giving up. Cheapest, highest impact.
- **B — Model fallback:** on repeated 503 from Flash-Lite, retry the batch with
  `GEMINI_MODEL_FALLBACK` (gemini-2.5-flash) which has separate quota.
- **C — Local regex pre-extraction:** parse funding amount / company name from
  the headline with a regex first; only call Gemini to enrich/disambiguate.
  Makes the hunt independent of Gemini availability for the basic case.

Recommended: A + B together (retry, then fall back to a different model).

---

## 2. Master run_id across multi-segment runs (LOW)

**Current:** The consolidated digest and run records use one segment's `run_id`
as the identifier. There's no single master id tying the three segment runs of a
given cron invocation together.

**Fix:** Generate one master `run_id` in `run_all_segments`, pass it down so each
segment stores `master_run_id` alongside its own. Makes Run History filtering and
the dashboard's "show me everything from this morning's run" trivial.

---

## 3. `.unknown` domain resolution (carryover from Phases 3–5)

**Observed:** Most candidates have `.unknown` domains (e.g. `sphere.unknown`,
`gr3n.unknown`), which causes:
- CompaniesAPI 400 errors in `company_hunter._enrich_with_firmographics`
  (it calls `enrich_by_domain` on a fake domain).
- Low reachability sub-scores in the qualifier → leads dropped before tier-1.

**Root cause:** RSS/news extraction captures the *publisher's* URL
(techcrunch.com), not the funded company's real domain. The slug fallback then
generates `<companyname>.unknown`.

**Fix (highest-leverage, ~70% of cases):** In `_hunt_via_rss`, fetch the article
page for each funding item and extract the first non-news outbound link — that's
almost always the funded company's site. Costs one extra HTTP GET per article,
no API credits.

**Quick mitigation (do first, one line):** In
`company_hunter._enrich_with_firmographics`, skip the CompaniesAPI call when
`c.domain.endswith(".unknown")` to kill the 400 noise immediately:
```python
if c.domain.endswith(".unknown"):
    enriched.append(c)
    continue
```

**Secondary fix:** `qualifier._resolve_domain` calls `search_by_filters` with all
`None` filters and never passes `candidate.name` — so it can't actually resolve
the company. Wire the company name into the search query.

---

## 4. Python version parity (INFORMATIONAL)

Streamlit Cloud runs Python 3.14; local dev + CI run 3.12. All packages install
cleanly on 3.14 today. If a future dependency isn't 3.14-compatible, Cloud deploy
will fail. If something breaks on Cloud but works locally, check this first.
Streamlit Cloud does not currently allow pinning the Python version.

---

## 5. Operating-phase data flywheel (PROCESS, not code)

Every reply (and non-reply) is training data. Once outreach starts:
- Log responses via the dashboard's "Mark Replied" action.
- After ~30 days, correlate qualifier sub-scores against reply outcomes to
  recalibrate `scoring_thresholds` in `config/icp_configs.json`.
- Feed observed outcomes into the `IcpStrategist.suggest_refinements()` stub
  (built in Phase 2, currently a placeholder).

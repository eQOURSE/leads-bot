"""ICP Strategist agent.

Loads, validates and explains the Ideal Customer Profile (ICP) strategies that
every downstream agent pulls from. Backed by ``config/icp_configs.json`` and the
typed ``IcpStrategy`` model.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from agents._gemini_wrapper import GeminiAgent
from agents._models import IcpStrategy
from config.logging_config import setup_logging
from config.settings import Settings

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "icp_configs.json"
_CACHE_DIR = _PROJECT_ROOT / "data" / "cache" / "icp_explain"
_EXPLAIN_TTL_SECONDS = 24 * 60 * 60  # 24h

_VALID_SEGMENTS = ("tutrain", "eqourse_content", "eqourse_ai_data")


class IcpStrategist:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.log = setup_logging("agent.icp_strategist")
        self._raw_cache: Optional[Dict[str, dict]] = None

    # ----- loading -------------------------------------------------------------

    def _load_raw(self) -> Dict[str, dict]:
        if self._raw_cache is None:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
                self._raw_cache = json.load(fh)
        return self._raw_cache

    def list_segments(self) -> List[str]:
        """Return the three valid segment keys."""
        return list(_VALID_SEGMENTS)

    def load_strategy(self, segment: str) -> IcpStrategy:
        """Load and validate the strategy for ``segment``.

        Raises ``ValueError`` if the segment is unknown.
        """
        raw = self._load_raw()
        if segment not in raw:
            valid = ", ".join(sorted(raw.keys()))
            raise ValueError(
                f"Unknown ICP segment {segment!r}. Valid segments: {valid}"
            )
        return IcpStrategy.model_validate(raw[segment])

    # ----- explain (Gemini-formatted, disk-cached 24h) -------------------------

    def _cache_path(self, segment: str) -> Path:
        key = hashlib.sha256(segment.encode("utf-8")).hexdigest()[:16]
        return _CACHE_DIR / f"{segment}_{key}.txt"

    def _read_cache(self, segment: str) -> Optional[str]:
        path = self._cache_path(segment)
        if not path.is_file():
            return None
        if (time.time() - path.stat().st_mtime) > _EXPLAIN_TTL_SECONDS:
            return None
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return None

    def _write_cache(self, segment: str, content: str) -> None:
        path = self._cache_path(segment)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            self.log.warning("Failed to cache explanation for %s: %s", segment, exc)

    async def explain_strategy(self, segment: str) -> str:
        """Return a human-readable summary of the strategy (Gemini-formatted).

        Cached on disk for 24h. Falls back to a deterministic local summary if
        Gemini is unavailable, so callers always get usable output.
        """
        strategy = self.load_strategy(segment)  # validates / raises on unknown

        cached = self._read_cache(segment)
        if cached is not None:
            self.log.info("explain_strategy cache hit for %s", segment)
            return cached

        prompt = (
            "Summarize the following Ideal Customer Profile for a sales team in "
            "clear, concise prose (no JSON, no code fences). Cover who we target, "
            "why they need us, and how we reach out.\n\n"
            f"{json.dumps(strategy.model_dump(mode='json'), indent=2)}"
        )
        agent = GeminiAgent(self.settings.GEMINI_MODEL_QUALIFIER, self.settings)
        text = await agent.generate_text(prompt, temperature=0.3, max_tokens=800)

        if not text:
            text = self._local_summary(strategy)

        self._write_cache(segment, text)
        return text

    @staticmethod
    def _local_summary(strategy: IcpStrategy) -> str:
        """Deterministic fallback summary that needs no model call."""
        ind = strategy.target_industries
        prof = strategy.target_company_profile
        return (
            f"{strategy.segment_name}: {strategy.value_prop_one_liner}\n\n"
            f"We offer: {strategy.what_we_offer}\n\n"
            f"Targeting {', '.join(ind.industry_keywords)} companies "
            f"({', '.join(prof.size_ranges)} employees) at "
            f"{', '.join(prof.funding_stages)} stage, founded after "
            f"{prof.founded_after_year}, in "
            f"{', '.join(prof.geographies.countries) or 'any geography'}.\n\n"
            f"Decision makers: {', '.join(strategy.target_titles[:6])}...\n\n"
            f"Pain: {strategy.outreach_angle.pain_hypothesis}\n"
            f"Pitch: {strategy.outreach_angle.value_framing}\n"
            f"CTA: {strategy.outreach_angle.primary_cta}"
        )

    # ----- Phase 10 placeholder ------------------------------------------------

    async def suggest_refinements(
        self, segment: str, recent_lead_outcomes: List[dict]
    ) -> dict:
        """Phase 10 placeholder.

        Will use Gemini Pro to suggest tweaks to scoring weights / negative
        signals based on observed outcomes (replied / no-reply / bounced).
        For now, returns a stub but keeps the signature stable.
        """
        # Validate the segment exists so callers fail fast on typos.
        self.load_strategy(segment)
        return {
            "status": "not_yet_implemented",
            "data_collected": len(recent_lead_outcomes),
        }

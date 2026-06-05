"""Apify client with multi-token credit-based rotation.

Rotates through APIFY_TOKEN_1..4, always picking the token with the most
estimated credit remaining. Each token starts with an estimated balance
(default $5.00) and is decremented by an estimated per-call cost. When all
tokens are exhausted, calls return empty results.
"""

from __future__ import annotations

from typing import Optional

import httpx

from config.settings import Settings
from sources.base import BaseSourceClient
from sources.models import SearchResult

_DEFAULT_TIMEOUT = 120.0

# Rough per-call cost estimates (USD). Conservative so we stop before overspend.
_COST_GOOGLE_SEARCH = 0.30
_COST_LINKEDIN_COMPANY = 0.50

_ACTOR_GOOGLE_SEARCH = "apify~google-search-scraper"
_ACTOR_LINKEDIN_COMPANY = "dev_fusion~linkedin-company-scraper"


class ApifyMultiKeyClient(BaseSourceClient):
    source_name = "apify"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.tokens = settings.apify_tokens
        initial = settings.APIFY_INITIAL_CREDITS_USD
        # In-memory estimated credits per token.
        self.credits: dict[str, float] = {t: initial for t in self.tokens}

    def _pick_token(self, cost: float) -> Optional[str]:
        """Return the token with the most remaining credit that can afford cost."""
        if not self.credits:
            return None
        token, remaining = max(self.credits.items(), key=lambda kv: kv[1])
        if remaining < cost:
            return None
        return token

    def _charge(self, token: str, cost: float) -> None:
        self.credits[token] = max(0.0, self.credits.get(token, 0.0) - cost)

    async def _run_actor(self, actor_id: str, run_input: dict, cost: float) -> list[dict]:
        token = self._pick_token(cost)
        if token is None:
            self.log.error(
                "Apify: all tokens exhausted (need $%.2f) — returning empty.", cost
            )
            return []

        # Synchronous run that returns dataset items directly.
        url = f"{self.settings.APIFY_BASE_URL.rstrip('/')}/acts/{actor_id}/run-sync-get-dataset-items"
        params = {"token": token}
        try:
            async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
                resp = await self._request(
                    client, "POST", url, params=params, json=run_input
                )
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            self.log.error("apify actor %s failed: %s", actor_id, exc)
            return []

        self._charge(token, cost)
        remaining_total = int(sum(self.credits.values()) * 100)  # cents-ish proxy
        await self._track(int(cost * 100), remaining_total)

        if isinstance(data, list):
            return data
        return data.get("items", []) if isinstance(data, dict) else []

    async def google_search(
        self, query: str, num_results: int = 20
    ) -> list[SearchResult]:
        run_input = {
            "queries": query,
            "resultsPerPage": num_results,
            "maxPagesPerQuery": 1,
        }
        items = await self._run_actor(
            _ACTOR_GOOGLE_SEARCH, run_input, _COST_GOOGLE_SEARCH
        )

        results: list[SearchResult] = []
        position = 0
        for item in items:
            organic = item.get("organicResults") if isinstance(item, dict) else None
            if organic is None and isinstance(item, dict) and item.get("url"):
                organic = [item]
            for r in organic or []:
                position += 1
                try:
                    results.append(
                        SearchResult(
                            title=r.get("title") or "",
                            url=r.get("url") or "",
                            snippet=r.get("description") or r.get("snippet") or "",
                            position=r.get("position") or position,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    self.log.warning("apify google result skipped: %s", exc)
                if len(results) >= num_results:
                    return results
        return results

    async def linkedin_company(self, linkedin_url: str) -> dict:
        self.log.warning(
            "apify.linkedin_company called (expensive, use sparingly): %s",
            linkedin_url,
        )
        run_input = {"profileUrls": [linkedin_url]}
        items = await self._run_actor(
            _ACTOR_LINKEDIN_COMPANY, run_input, _COST_LINKEDIN_COMPANY
        )
        if items and isinstance(items[0], dict):
            return items[0]
        return {}

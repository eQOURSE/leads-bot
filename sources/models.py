"""Shared Pydantic models for the data source layer.

Every source client accepts and returns these typed models so the rest of the
system can treat heterogeneous sources uniformly.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator

_PROTOCOL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")


def _strip_domain(value: str) -> str:
    """Lightweight domain normalization used by validators.

    Heavier, tldextract-based normalization lives in ``sources._utils``; this
    keeps the models import-light and dependency-free.
    """
    v = value.strip().lower()
    v = _PROTOCOL_RE.sub("", v)
    if v.startswith("www."):
        v = v[4:]
    v = v.split("/")[0].split("?")[0]
    return v


class CompanyCandidate(BaseModel):
    """A company surfaced by a discovery / enrichment source."""

    domain: str
    name: str
    description: Optional[str] = None
    industry: Optional[str] = None
    naics_codes: list[str] = Field(default_factory=list)
    linkedin_category: Optional[str] = None
    size_range: Optional[str] = None
    revenue_range: Optional[str] = None
    hq_country: Optional[str] = None
    hq_region: Optional[str] = None
    website: Optional[str] = None
    funding_amount_usd: Optional[float] = None
    funding_stage: Optional[str] = None
    funding_date: Optional[date] = None
    funding_source: Optional[str] = None
    raw_source: str
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("domain")
    @classmethod
    def _normalize_domain(cls, v: str) -> str:
        return _strip_domain(v)

    @field_validator("naics_codes", mode="before")
    @classmethod
    def _coerce_naics(cls, v: object) -> list[str]:
        if v is None:
            return []
        if isinstance(v, (list, tuple)):
            return [str(x) for x in v]
        return [str(v)]


class ProspectCandidate(BaseModel):
    """A person (decision maker) associated with a company."""

    full_name: str
    title: str
    company_domain: str
    linkedin_url: Optional[str] = None
    email: Optional[str] = None
    email_confidence: Optional[float] = None
    phone: Optional[str] = None
    source: str

    @field_validator("company_domain")
    @classmethod
    def _normalize_company_domain(cls, v: str) -> str:
        return _strip_domain(v)

    @field_validator("email_confidence")
    @classmethod
    def _clamp_confidence(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        # Hunter returns 0-100; normalize anything > 1 onto a 0-1 scale.
        if v > 1.0:
            v = v / 100.0
        return max(0.0, min(1.0, v))


class NewsItem(BaseModel):
    """A news / announcement article."""

    title: str
    url: str
    published_at: datetime
    source_name: str
    snippet: str = ""
    company_name_guess: Optional[str] = None


class SearchResult(BaseModel):
    """A single organic search engine result."""

    title: str
    url: str
    snippet: str = ""
    position: int


class ScrapedContent(BaseModel):
    """Structured data extracted from a scraped page."""

    url: str
    extracted_data: dict = Field(default_factory=dict)
    credits_used: int = 0

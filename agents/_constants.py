"""Agent-level constants shared across Phase 5+ agents."""

from __future__ import annotations

import tldextract

# ---------------------------------------------------------------------------
# News / media domain blacklist — companies from these domains are flagged as
# "needs_manual_lookup" because we can't extract a real company website.
# ---------------------------------------------------------------------------

NEWS_SOURCE_DOMAINS: frozenset[str] = frozenset({
    "techcrunch.com", "strictlyvc.com", "news.crunchbase.com",
    "edsurge.com", "tech.eu", "sifted.eu", "thesaasnews.com",
    "venturebeat.com", "forbes.com", "bloomberg.com", "reuters.com",
    "wsj.com", "ft.com", "nytimes.com", "businesswire.com", "prnewswire.com",
    "axios.com", "theinformation.com", "fortune.com", "geekwire.com",
    "businessinsider.com", "cnbc.com", "medium.com", "substack.com",
    "wikipedia.org", "linkedin.com", "twitter.com", "x.com",
    "youtube.com", "github.com",
})

# ---------------------------------------------------------------------------
# Phase 11 — NAICS / ICP keyword → Crunchbase & Wellfound industry categories
# ---------------------------------------------------------------------------
# Crunchbase uses free-text "category groups". Wellfound uses "markets".
# These mappings translate our ICP NAICS codes / keywords into the category
# strings those discovery sources expect. Conservative coverage of the three
# segments; extend as needed.

NAICS_TO_CRUNCHBASE: dict[str, list[str]] = {
    # EdTech / education
    "611710": ["Education", "EdTech", "E-Learning"],
    "611699": ["Education", "EdTech", "E-Learning"],
    "611420": ["Education", "Training"],
    "611691": ["Education", "Tutoring", "Test Prep"],
    "511130": ["Publishing", "EdTech", "Content"],
    # AI / data / software
    "541512": ["Artificial Intelligence", "Machine Learning", "Software"],
    "541511": ["Software", "Artificial Intelligence"],
    "541715": ["Artificial Intelligence", "Machine Learning", "Research"],
}

KEYWORD_TO_CRUNCHBASE: dict[str, list[str]] = {
    "edtech": ["Education", "EdTech", "E-Learning"],
    "k-12": ["Education", "EdTech"],
    "test prep": ["Education", "Test Prep", "Tutoring"],
    "tutoring": ["Education", "Tutoring"],
    "online learning": ["Education", "E-Learning"],
    "language learning": ["Education", "Language Learning"],
    "ai": ["Artificial Intelligence", "Machine Learning"],
    "ml": ["Machine Learning", "Artificial Intelligence"],
    "llm": ["Artificial Intelligence", "Generative AI", "Natural Language Processing"],
    "machine learning": ["Machine Learning", "Artificial Intelligence"],
    "voice ai": ["Artificial Intelligence", "Speech Recognition"],
    "computer vision": ["Artificial Intelligence", "Computer Vision"],
    "nlp": ["Natural Language Processing", "Artificial Intelligence"],
    "rlhf": ["Artificial Intelligence", "Machine Learning"],
    "foundation model": ["Artificial Intelligence", "Generative AI"],
    "ai agent": ["Artificial Intelligence", "Generative AI"],
}

KEYWORD_TO_WELLFOUND_MARKET: dict[str, list[str]] = {
    "edtech": ["Education", "EdTech"],
    "k-12": ["Education", "K-12 Education"],
    "test prep": ["Education", "Test Preparation"],
    "tutoring": ["Education", "Tutoring"],
    "online learning": ["Education", "E-Learning"],
    "language learning": ["Language Learning"],
    "ai": ["Artificial Intelligence"],
    "ml": ["Machine Learning"],
    "llm": ["Artificial Intelligence", "Generative AI"],
    "machine learning": ["Machine Learning"],
    "voice ai": ["Artificial Intelligence", "Voice"],
    "computer vision": ["Computer Vision"],
    "nlp": ["Natural Language Processing"],
}


def crunchbase_categories_for_icp(naics_codes, keywords) -> list[str]:
    """Return de-duplicated Crunchbase category strings for an ICP."""
    cats: list[str] = []
    for code in (naics_codes or []):
        cats.extend(NAICS_TO_CRUNCHBASE.get(str(code), []))
    for kw in (keywords or []):
        cats.extend(KEYWORD_TO_CRUNCHBASE.get(str(kw).lower(), []))
    # de-dup, preserve order
    seen: set[str] = set()
    out: list[str] = []
    for c in cats:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def wellfound_markets_for_icp(keywords) -> list[str]:
    """Return de-duplicated Wellfound market strings for an ICP."""
    markets: list[str] = []
    for kw in (keywords or []):
        markets.extend(KEYWORD_TO_WELLFOUND_MARKET.get(str(kw).lower(), []))
    seen: set[str] = set()
    out: list[str] = []
    for m in markets:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


# ---------------------------------------------------------------------------
# Seniority ranking for decision-maker scoring
# ---------------------------------------------------------------------------

SENIORITY_RANK: dict[str, int] = {
    "founder": 100,
    "co-founder": 95,
    "ceo": 95,
    "cto": 90,
    "cpo": 90,
    "coo": 88,
    "cfo": 85,
    "cmo": 85,
    "cao": 85,
    "chief": 90,
    "president": 85,
    "vp": 75,
    "vice president": 75,
    "head of": 70,
    "director": 65,
    "lead": 55,
    "manager": 50,
    "senior": 45,
}


def domain_is_news_source(domain: str) -> bool:
    """Return True if the domain is a news/media blacklisted source.

    Handles subdomains via tldextract (e.g. 'news.techcrunch.com' → True).
    """
    if not domain:
        return False
    domain_lower = domain.strip().lower()

    # Direct match
    if domain_lower in NEWS_SOURCE_DOMAINS:
        return True

    # tldextract-based registered domain match (handles subdomains)
    try:
        extracted = tldextract.extract(domain_lower)
        registered = f"{extracted.domain}.{extracted.suffix}" if extracted.suffix else ""
        if registered and registered in NEWS_SOURCE_DOMAINS:
            return True
    except Exception:  # noqa: BLE001
        pass

    return False


def seniority_score(title: str) -> int:
    """Return the highest seniority rank that matches any keyword in the title.

    Case-insensitive substring match. Returns 0 if no keyword matches.
    """
    if not title:
        return 0
    title_lower = title.lower()
    best = 0
    for keyword, score in SENIORITY_RANK.items():
        if keyword in title_lower:
            if score > best:
                best = score
    return best


# ---------------------------------------------------------------------------
# Phase 8 — Profanity / spam-trigger word list for Validator
# ---------------------------------------------------------------------------

BANNED_WORDS: frozenset[str] = frozenset({
    # Standard profanity (broad category, not exhaustive)
    "shit", "fuck", "ass", "bitch", "bastard", "crap", "damn", "hell",
    "dick", "cock", "pussy", "whore", "slut",
    # Discriminatory slurs (omitted for brevity — covered by broad regex in validator)
    "faggot", "nigger",
    # Spam trigger words
    "viagra", "lottery", "winner", "you have won", "click here",
    "free money", "make money fast", "work from home",
})

"""Phase 11 — tests for article-based company domain extraction (pure logic)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sources._article_extract import (
    extract_candidate_domains,
    pick_company_domain,
    _is_excluded_domain,
    _publisher_domain,
)


_ARTICLE_URL = "https://techcrunch.com/2026/06/01/acme-raises-5m/"

_HTML = """
<html><head>
<link rel="canonical" href="https://techcrunch.com/2026/06/01/acme-raises-5m/"/>
</head><body>
<article>
  <p>Acme Labs, a startup, announced funding. Visit
     <a href="https://acmelabs.io/about">Acme Labs</a> for details.</p>
  <p>Follow on <a href="https://twitter.com/acmelabs">Twitter</a> and
     <a href="https://linkedin.com/company/acmelabs">LinkedIn</a>.</p>
  <p>Read more on <a href="https://techcrunch.com/tag/funding/">TechCrunch</a>.</p>
  <p>Privacy <a href="https://techcrunch.com/privacy">policy</a>.</p>
  <p>Investor: <a href="https://sequoiacap.com">Sequoia</a>.</p>
</article>
</body></html>
"""


def test_publisher_domain():
    assert _publisher_domain(_ARTICLE_URL) == "techcrunch.com"


def test_excludes_publisher_social_and_news():
    pub = "techcrunch.com"
    assert _is_excluded_domain("techcrunch.com", pub) is True
    assert _is_excluded_domain("twitter.com", pub) is True
    assert _is_excluded_domain("linkedin.com", pub) is True
    assert _is_excluded_domain("acmelabs.io", pub) is False


def test_extract_candidate_domains_finds_company_first():
    candidates = extract_candidate_domains(_HTML, _ARTICLE_URL)
    # acmelabs.io must be present and ranked before sequoiacap.com (appears earlier)
    assert "acmelabs.io" in candidates
    assert candidates.index("acmelabs.io") < candidates.index("sequoiacap.com")
    # publisher, social, and news are excluded
    assert "techcrunch.com" not in candidates
    assert "twitter.com" not in candidates
    assert "linkedin.com" not in candidates


def test_pick_company_domain_returns_first_candidate():
    assert pick_company_domain(_HTML, _ARTICLE_URL) == "acmelabs.io"


def test_pick_company_domain_none_when_only_excluded():
    html = """
    <body><a href="https://techcrunch.com/x">x</a>
    <a href="https://twitter.com/y">y</a></body>
    """
    assert pick_company_domain(html, _ARTICLE_URL) is None


def test_canonical_and_ogurl_ignored_when_publisher():
    html = """
    <head>
      <link rel="canonical" href="https://techcrunch.com/foo"/>
      <meta property="og:url" content="https://techcrunch.com/foo"/>
    </head><body><p>no company links</p></body>
    """
    assert pick_company_domain(html, _ARTICLE_URL) is None

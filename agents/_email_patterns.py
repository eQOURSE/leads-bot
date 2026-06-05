"""Email pattern utilities for the Contact Enricher.

Provides:
- _split_name          : (full_name) → (first, last) as clean ASCII lowercase strings
- _default_patterns    : standard email format guesses, priority-ordered
- _apply_pattern       : apply a Hunter-style pattern like "{first}.{last}" to a domain
- _name_matches_email  : loose check whether a full name is reflected in an email address
"""

from __future__ import annotations

import re
import unicodedata

# Honorifics / titles to strip from the start of a name
_HONORIFICS = {
    "dr", "dr.", "mr", "mr.", "mrs", "mrs.", "ms", "ms.", "prof", "prof.",
    "sir", "rev", "rev.", "phd", "md",
}

_NON_ALPHA_RE = re.compile(r"[^a-z0-9]")


def _ascii_fold(text: str) -> str:
    """Fold unicode characters to their ASCII equivalents (e.g. é → e)."""
    try:
        import unicodedata
        nfkd = unicodedata.normalize("NFKD", text)
        return "".join(c for c in nfkd if not unicodedata.combining(c))
    except Exception:  # noqa: BLE001
        return text


def _split_name(full_name: str) -> tuple[str, str]:
    """Return (first, last) as lowercased ASCII strings.

    Strips leading honorifics, handles 2-word and 3+-word names (uses first
    and last word). Returns ("", "") if the name is unusable.
    """
    parts = _ascii_fold(full_name.strip()).lower().split()
    # Drop leading honorifics
    while parts and parts[0].rstrip(".") in _HONORIFICS:
        parts.pop(0)

    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], parts[0]

    first = _NON_ALPHA_RE.sub("", parts[0])
    last = _NON_ALPHA_RE.sub("", parts[-1])
    return first, last


def _default_patterns(first: str, last: str, domain: str) -> list[str]:
    """Return candidate emails in priority order (most common first)."""
    if not first or not last:
        return []
    f = first[0]  # first initial
    l = last[0]   # last initial  # noqa: E741
    return [
        f"{first}.{last}@{domain}",
        f"{first}{last}@{domain}",
        f"{f}{last}@{domain}",
        f"{first}@{domain}",
        f"{first}.{l}@{domain}",
        f"{f}.{last}@{domain}",
    ]


def _apply_pattern(pattern: str, first: str, last: str, domain: str) -> str:
    """Apply a Hunter-style pattern string to produce an email address.

    Supported placeholders: {first}, {last}, {f} (first initial), {l} (last initial).
    Falls back to first.last if the pattern is unrecognised.
    """
    if not pattern or not first or not last:
        return f"{first}.{last}@{domain}"

    f = first[0] if first else ""
    l = last[0] if last else ""  # noqa: E741

    local = (
        pattern
        .replace("{first}", first)
        .replace("{last}", last)
        .replace("{f}", f)
        .replace("{l}", l)
    )
    # Strip any remaining braces from unknown placeholders
    local = re.sub(r"\{[^}]+\}", "", local)
    return f"{local}@{domain}"


def _name_matches_email(full_name: str, email: str) -> bool:
    """Return True if name parts appear in the email local-part (loose match).

    "Sara Chen" → matches sara.chen@, schen@, sara@, s.chen@
    Does NOT match marcus@ or other unrelated names.
    """
    first, last = _split_name(full_name)
    if not first:
        return False

    local = email.split("@")[0].lower()
    local_clean = _NON_ALPHA_RE.sub("", local)

    # Direct name appearances in raw local (with separators)
    if first in local and last in local:
        return True

    # Initial + last (e.g., "schen")
    if first[0] + last in local_clean:
        return True

    # First + initial (e.g., "sarac")
    if first + last[0] in local_clean:
        return True

    # Just first name (e.g., "sara@")
    if local_clean == first or local_clean.startswith(first):
        # Only accept if first is at least 4 chars (avoids false positives on "a@", "jo@")
        if len(first) >= 4:
            return True

    return False

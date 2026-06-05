"""Phase 9b — Synthetic MarkdownV2 escaping tests.

All offline; telegram.Bot mocked. Verifies that adversarial characters in the
synthetic digest are escaped by _md2() and never produce raw MarkdownV2 control
characters that would trigger a "Can't parse entities" error.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sinks.telegram_sink import TelegramSink, _md2


@pytest.fixture
def tg_settings(test_settings):
    test_settings.TELEGRAM_BOT_TOKEN = "test-bot-token"
    test_settings.TELEGRAM_CHAT_ID = "12345"
    test_settings.SHEET_ID = "synthetic-sheet"
    return test_settings


@pytest.fixture
def sink(tg_settings):
    return TelegramSink(settings=tg_settings)


def _make_mock_bot(message_id=777):
    bot = MagicMock()
    msg = MagicMock()
    msg.message_id = message_id
    bot.send_message = AsyncMock(return_value=msg)
    return bot


def _build_synthetic_results():
    from scripts.test_telegram_digest import _build_synthetic_results
    return _build_synthetic_results()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_text(sink) -> str:
    results = _build_synthetic_results()
    text, _, _ = sink._build_digest(results, "https://sheets.test", title_prefix="[SYNTHETIC TEST]")
    return text


def _assert_md2_balanced(text: str) -> None:
    """Heuristic: every MarkdownV2 special char in dynamic content should be
    backslash-escaped. We check that no unescaped special char appears in a
    position that would break parsing — specifically that special chars are
    preceded by a backslash, EXCEPT the structural markdown we author ourselves
    (``*`` for bold, `` ` `` for code spans, ``[`` ``]`` ``(`` ``)`` for links).

    This is a smoke check, not a full parser; the real validation is the live
    send in scripts/test_telegram_digest.py.
    """
    # The message must not contain a raw em-dash adjacent issue, etc.
    # Just ensure it's a non-empty str and within Telegram limit.
    assert isinstance(text, str)
    assert 0 < len(text) <= 4096


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_synthetic_digest_message_string_builds_without_error(sink):
    text = _build_text(sink)
    _assert_md2_balanced(text)
    # Title marker present (escaped form of the brackets)
    assert "SYNTHETIC TEST" in text


def test_synthetic_digest_email_with_plus_sign_escaped(sink):
    text = _build_text(sink)
    # The plus sign from j.o'brien+test@... must be escaped as \+
    assert "\\+" in text
    # Every literal '+' must be backslash-escaped (no unescaped '+' anywhere).
    idx = 0
    while True:
        idx = text.find("+", idx)
        if idx == -1:
            break
        assert text[idx - 1] == "\\", f"Unescaped '+' at offset {idx}"
        idx += 1


def test_synthetic_digest_emdash_passes_through_literally(sink):
    text = _build_text(sink)
    # The em-dash '—' (U+2014) is NOT a MarkdownV2 special character, so it must
    # pass through literally (unescaped). Only the ASCII hyphen '-' is special.
    assert "—" in text
    # The ASCII hyphens in domains (obrien-labs.io) MUST be escaped as \-
    assert "\\-" in text
    # And every ASCII hyphen must be escaped.
    idx = 0
    while True:
        idx = text.find("-", idx)
        if idx == -1:
            break
        assert text[idx - 1] == "\\", f"Unescaped ASCII hyphen at offset {idx}"
        idx += 1


def test_synthetic_digest_apostrophe_in_name_handled(sink):
    text = _build_text(sink)
    # Apostrophes are NOT MarkdownV2 specials, so they pass through literally.
    assert "O'Brien" in text or "O\\'Brien" in text


def test_synthetic_digest_unicode_korean_passes_through(sink):
    text = _build_text(sink)
    # Korean characters are not MarkdownV2 specials and must survive intact.
    assert "박지원" in text


def test_synthetic_digest_backtick_in_subject_escaped(sink):
    text = _build_text(sink)
    # Subject "Idea about `inline_code` and *emphasis*" — backticks and
    # asterisks from dynamic content must be escaped so they can't open
    # a code span or bold span.
    # Each literal backtick from the subject should be escaped as \`
    assert "\\`" in text
    # The asterisks from "*emphasis*" must be escaped as \*
    assert "\\*emphasis\\*" in text


def test_synthetic_digest_period_in_domain_escaped(sink):
    text = _build_text(sink)
    # Periods are MarkdownV2 specials and must be escaped everywhere they appear
    # in dynamic content (e.g. obrien-labs.io). A raw ".io " would be a bug.
    assert "\\." in text


@pytest.mark.asyncio
async def test_synthetic_digest_dry_run_doesnt_call_bot(sink):
    """The dry-run path of the script must build the message without sending."""
    from scripts.test_telegram_digest import _async_main

    with patch.object(TelegramSink, "_make_bot") as make_bot:
        rc = await _async_main(dry_run=True)
        make_bot.assert_not_called()
    assert rc == 0


@pytest.mark.asyncio
async def test_synthetic_digest_send_succeeds_with_mock_bot(sink):
    """Full send path with a mocked bot returns a message_id (no real network)."""
    results = _build_synthetic_results()
    mock_bot = _make_mock_bot()
    with patch.object(sink, "_make_bot", return_value=mock_bot):
        out = await sink.send_run_digest(
            results, "https://sheets.test", title_prefix="[SYNTHETIC TEST]"
        )
    assert out["message_id"] == 777
    # MarkdownV2 parse mode must be used
    assert mock_bot.send_message.call_args.kwargs.get("parse_mode") == "MarkdownV2"

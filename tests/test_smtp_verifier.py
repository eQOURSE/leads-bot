"""Tests for SMTPVerifier — Phase 6. All offline; smtplib mocked."""

from __future__ import annotations

import socket
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sources.smtp_verifier import SMTPResult, SMTPVerifier


def _make_smtp_mock(rcpt_code: int = 250, connect_raises=None, rcpt_raises=None):
    """Return a mock smtplib.SMTP class whose instance behaves as configured."""
    smtp_instance = MagicMock()
    smtp_instance.__enter__ = MagicMock(return_value=smtp_instance)
    smtp_instance.__exit__ = MagicMock(return_value=False)

    if connect_raises:
        smtp_instance.connect.side_effect = connect_raises
    else:
        smtp_instance.connect.return_value = (220, b"OK")

    smtp_instance.ehlo.return_value = (250, b"OK")
    smtp_instance.mail.return_value = (250, b"OK")

    if rcpt_raises:
        smtp_instance.rcpt.side_effect = rcpt_raises
    else:
        smtp_instance.rcpt.return_value = (rcpt_code, b"OK")

    smtp_instance.quit.return_value = (221, b"Bye")
    return smtp_instance


def _patch_dns_and_smtp(mx_hosts, smtp_mock):
    """Context managers to patch DNS resolver and smtplib.SMTP together."""
    dns_patch = patch(
        "sources.smtp_verifier.dns.resolver.resolve",
        side_effect=lambda *a, **kw: _fake_mx_answers(mx_hosts),
    )
    smtp_patch = patch("sources.smtp_verifier.smtplib.SMTP", return_value=smtp_mock)
    return dns_patch, smtp_patch


def _fake_mx_answers(mx_hosts: list[tuple[str, int]]):
    answers = []
    for host, priority in mx_hosts:
        r = MagicMock()
        r.exchange = host + "."
        r.preference = priority
        answers.append(r)
    return answers


# ---------------------------------------------------------------------------
# test_verify_email_success_on_250
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_verify_email_success_on_250(test_settings):
    smtp_mock = _make_smtp_mock(rcpt_code=250)
    dns_p, smtp_p = _patch_dns_and_smtp([("mail.acme.com", 10)], smtp_mock)
    with dns_p, smtp_p:
        verifier = SMTPVerifier(test_settings)
        result = await verifier.verify_email("alice@acme.com")

    assert result.exists is True
    assert result.smtp_response == "accepted"
    assert result.mx_records_found is True


# ---------------------------------------------------------------------------
# test_verify_email_failure_on_550
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_verify_email_failure_on_550(test_settings):
    smtp_mock = _make_smtp_mock(rcpt_code=550)
    dns_p, smtp_p = _patch_dns_and_smtp([("mail.acme.com", 10)], smtp_mock)
    with dns_p, smtp_p:
        verifier = SMTPVerifier(test_settings)
        result = await verifier.verify_email("nobody@acme.com")

    assert result.exists is False
    assert result.smtp_response == "rejected"


# ---------------------------------------------------------------------------
# test_verify_email_inconclusive_on_greylisting
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_verify_email_inconclusive_on_greylisting(test_settings):
    smtp_mock = _make_smtp_mock(rcpt_code=451)  # 4xx = greylisting
    dns_p, smtp_p = _patch_dns_and_smtp([("mail.startup.io", 10)], smtp_mock)
    with dns_p, smtp_p:
        verifier = SMTPVerifier(test_settings)
        result = await verifier.verify_email("test@startup.io")

    assert result.exists is None
    assert result.smtp_response == "greylisted_or_inconclusive"


# ---------------------------------------------------------------------------
# test_verify_email_no_mx_handled
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_verify_email_no_mx_handled(test_settings):
    import dns.resolver

    with patch(
        "sources.smtp_verifier.dns.resolver.resolve",
        side_effect=dns.resolver.NXDOMAIN,
    ):
        verifier = SMTPVerifier(test_settings)
        result = await verifier.verify_email("test@thisisnotarealatall.com")

    assert result.exists is False
    assert result.smtp_response == "no_mx"
    assert result.mx_records_found is False


# ---------------------------------------------------------------------------
# test_verify_email_timeout_returns_inconclusive
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_verify_email_timeout_returns_inconclusive(test_settings):
    smtp_mock = _make_smtp_mock(connect_raises=socket.timeout("timed out"))
    dns_p, smtp_p = _patch_dns_and_smtp([("mail.slow.com", 10)], smtp_mock)
    with dns_p, smtp_p:
        verifier = SMTPVerifier(test_settings)
        result = await verifier.verify_email("user@slow.com")

    assert result.exists is None
    assert result.smtp_response == "timeout"


# ---------------------------------------------------------------------------
# test_cache_hit_skips_smtp_handshake
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cache_hit_skips_smtp_handshake(test_settings):
    smtp_mock = _make_smtp_mock(rcpt_code=250)
    dns_p, smtp_p = _patch_dns_and_smtp([("mail.cached.com", 10)], smtp_mock)
    with dns_p, smtp_p:
        verifier = SMTPVerifier(test_settings)
        first = await verifier.verify_email("user@cached.com")
        second = await verifier.verify_email("user@cached.com")

    assert first.exists == second.exists
    # SMTP was only called once (second call from cache)
    assert smtp_mock.connect.call_count == 1


# ---------------------------------------------------------------------------
# test_batch_verify_respects_concurrency
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_batch_verify_respects_concurrency(test_settings):
    """batch_verify should return one result per email."""
    smtp_mock = _make_smtp_mock(rcpt_code=550)
    dns_p, smtp_p = _patch_dns_and_smtp([("mail.x.com", 10)], smtp_mock)
    emails = [f"user{i}@x.com" for i in range(4)]
    with dns_p, smtp_p:
        verifier = SMTPVerifier(test_settings)
        results = await verifier.batch_verify(emails, concurrency=2)

    assert set(results.keys()) == set(emails)
    for r in results.values():
        assert isinstance(r, SMTPResult)

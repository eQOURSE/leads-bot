"""SMTP email verifier — no API key required.

Performs a SMTP handshake (EHLO → MAIL FROM → RCPT TO → QUIT) to check
whether a mailbox exists without ever sending a message. DNS MX lookup is done
via dnspython. Both the DNS call and the SMTP exchange run in asyncio.to_thread
so they do not block the event loop.

Results are cached in SQLite for 7 days so repeated runs do not re-probe the
same addresses.

NEVER sends DATA. The connection is closed immediately after reading the RCPT
TO response code.
"""

from __future__ import annotations

import asyncio
import smtplib
import socket
from typing import Optional

import dns.resolver
from pydantic import BaseModel

from config.logging_config import setup_logging
from config.settings import Settings
from sources._cache import cache_get, cache_set

_SMTP_TIMEOUT = 8.0      # seconds per SMTP connection attempt
_CACHE_TTL_DAYS = 7
_HELO_DOMAIN_DEFAULT = "verify.eqourse.com"
_MAIL_FROM_PREFIX = "noreply"


class SMTPResult(BaseModel):
    email: str
    exists: Optional[bool]      # True = accepted, False = rejected, None = inconclusive
    smtp_response: str          # "accepted", "rejected", "greylisted_or_inconclusive",
                                #   "timeout", "no_mx", "connection_error"
    mx_records_found: bool
    duration_ms: int


class SMTPVerifier:
    """Verify email addresses via SMTP without sending mail."""

    source_name = "smtp_verifier"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.helo_domain = getattr(settings, "SMTP_HELO_DOMAIN", _HELO_DOMAIN_DEFAULT)
        self.log = setup_logging(f"source.{self.source_name}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def verify_email(
        self, email: str, timeout: float = _SMTP_TIMEOUT
    ) -> SMTPResult:
        """Verify a single email via SMTP handshake.

        Returns SMTPResult; never raises.
        """
        # Cache check
        cached = await cache_get(self.source_name, email, self.settings)
        if cached is not None:
            self.log.info("smtp_verifier: cache hit for %s", email)
            return SMTPResult(**cached)

        result = await asyncio.to_thread(self._verify_sync, email, timeout)

        # Cache successful probes (both accepts and rejects)
        await cache_set(
            self.source_name,
            email,
            result.model_dump(),
            _CACHE_TTL_DAYS,
            self.settings,
        )
        return result

    async def batch_verify(
        self,
        emails: list[str],
        concurrency: int = 5,
    ) -> dict[str, SMTPResult]:
        """Verify multiple emails with a concurrency cap."""
        semaphore = asyncio.Semaphore(concurrency)

        async def _verify_one(email: str) -> tuple[str, SMTPResult]:
            async with semaphore:
                return email, await self.verify_email(email)

        pairs = await asyncio.gather(*[_verify_one(e) for e in emails])
        return dict(pairs)

    # ------------------------------------------------------------------
    # Synchronous implementation (runs in thread)
    # ------------------------------------------------------------------

    def _verify_sync(self, email: str, timeout: float) -> SMTPResult:
        import time
        start = time.perf_counter()

        if "@" not in email:
            return SMTPResult(
                email=email,
                exists=False,
                smtp_response="invalid_format",
                mx_records_found=False,
                duration_ms=0,
            )

        _, domain = email.rsplit("@", 1)
        mx_hosts = self._resolve_mx(domain)

        if not mx_hosts:
            ms = int((time.perf_counter() - start) * 1000)
            return SMTPResult(
                email=email,
                exists=False,
                smtp_response="no_mx",
                mx_records_found=False,
                duration_ms=ms,
            )

        mail_from = f"{_MAIL_FROM_PREFIX}@{self.helo_domain}"

        for mx_host, _priority in mx_hosts:
            try:
                smtp = smtplib.SMTP(timeout=timeout)
                smtp.connect(mx_host, 25)
                smtp.ehlo(self.helo_domain)
                smtp.mail(mail_from)
                code, _msg = smtp.rcpt(email)
                smtp.quit()

                ms = int((time.perf_counter() - start) * 1000)
                self.log.debug("smtp_verifier: %s → code %s (%dms)", email, code, ms)

                if code == 250:
                    return SMTPResult(
                        email=email,
                        exists=True,
                        smtp_response="accepted",
                        mx_records_found=True,
                        duration_ms=ms,
                    )
                elif code == 550:
                    return SMTPResult(
                        email=email,
                        exists=False,
                        smtp_response="rejected",
                        mx_records_found=True,
                        duration_ms=ms,
                    )
                else:
                    # 4xx greylisting or other transient
                    return SMTPResult(
                        email=email,
                        exists=None,
                        smtp_response="greylisted_or_inconclusive",
                        mx_records_found=True,
                        duration_ms=ms,
                    )

            except socket.timeout:
                ms = int((time.perf_counter() - start) * 1000)
                self.log.debug("smtp_verifier: timeout for %s on %s", email, mx_host)
                return SMTPResult(
                    email=email,
                    exists=None,
                    smtp_response="timeout",
                    mx_records_found=True,
                    duration_ms=ms,
                )
            except Exception as exc:  # noqa: BLE001
                self.log.debug(
                    "smtp_verifier: connection error for %s on %s: %s",
                    email, mx_host, exc,
                )
                # Try next MX host

        ms = int((time.perf_counter() - start) * 1000)
        return SMTPResult(
            email=email,
            exists=None,
            smtp_response="connection_error",
            mx_records_found=True,
            duration_ms=ms,
        )

    def _resolve_mx(self, domain: str) -> list[tuple[str, int]]:
        """Return list of (hostname, priority) sorted by priority asc."""
        try:
            answers = dns.resolver.resolve(domain, "MX", lifetime=5.0)
            records = sorted(
                [(str(r.exchange).rstrip("."), r.preference) for r in answers],
                key=lambda x: x[1],
            )
            return records
        except Exception as exc:  # noqa: BLE001
            self.log.debug("smtp_verifier: MX lookup failed for %s: %s", domain, exc)
            return []

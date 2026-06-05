"""Centralized logging configuration for the lead generation system."""

from __future__ import annotations

import logging
import os
import sys
from datetime import date
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Libraries that are noisy at INFO/DEBUG; pin them to WARNING.
_NOISY_LIBRARIES = ("urllib3", "httpx", "httpcore")


def _ensure_utf8_streams() -> None:
    """Force stdout/stderr to UTF-8 so Unicode glyphs (→, ✓, —) never crash.

    On Windows the default console encoding is cp1252, which cannot encode the
    arrows and check-marks used throughout our log messages and CLI output.
    Reconfiguring the streams in place (Python 3.7+) keeps existing references
    valid — including the one ``rich`` holds — and uses ``backslashreplace`` so
    a stray un-encodable character degrades gracefully instead of raising.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except (ValueError, OSError):
            # Stream may be detached or already wrapped; safe to ignore.
            pass


# Apply once at import time — this module is imported very early by every
# component, so the streams are UTF-8 before any log line is emitted.
_ensure_utf8_streams()


def setup_logging(name: str) -> logging.Logger:
    """Configure and return a logger that writes to stdout and a dated file.

    Args:
        name: Logical name for the logger. Also used in the log file name
            ``logs/{name}_{YYYY-MM-DD}.log``.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"{name}_{date.today().isoformat()}.log"

    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Avoid duplicate handlers if setup_logging is called more than once.
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Suppress noisy third-party libraries.
    for noisy in _NOISY_LIBRARIES:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logger

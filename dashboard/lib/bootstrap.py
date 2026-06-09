"""Shared bootstrap for dashboard pages.

Streamlit executes each file in pages/ as a standalone script, so the project
root must be on sys.path before importing config/dashboard modules. Each page
calls ``bootstrap()`` at the top.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Also add pages/ directory so relative imports between pages work if needed.
_PAGES_DIR = Path(__file__).resolve().parent.parent / "pages"
if str(_PAGES_DIR) not in sys.path:
    sys.path.insert(0, str(_PAGES_DIR))


def bootstrap() -> None:
    """Ensure project root is importable and enforce auth gate."""
    import streamlit as st
    from dashboard.lib.auth import check_password

    if not check_password():
        st.stop()

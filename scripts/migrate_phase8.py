"""Phase 8 schema migration — adds idempotency columns to the leads table.

Safe to run multiple times (checks PRAGMA table_info before each ALTER).

    python scripts/migrate_phase8.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sqlalchemy import create_engine, text

from config.settings import get_settings

_NEW_COLUMNS = [
    ("sent_to_sheets_at",  "TIMESTAMP"),
    ("sent_to_telegram_at", "TIMESTAMP"),
    ("sheets_row_index",   "INTEGER"),
    ("validation_reasons", "TEXT"),
]


def migrate(db_path: str | None = None) -> None:
    settings = get_settings()
    path = db_path or settings.SQLITE_PATH

    engine = create_engine(f"sqlite:///{path}", future=True)
    with engine.connect() as conn:
        # Get existing column names
        rows = conn.execute(text("PRAGMA table_info(leads)")).fetchall()
        existing = {row[1] for row in rows}   # column name is index 1

        for col_name, col_type in _NEW_COLUMNS:
            if col_name in existing:
                print(f"  SKIP  {col_name} (already exists)")
            else:
                conn.execute(text(f"ALTER TABLE leads ADD COLUMN {col_name} {col_type} NULL"))
                conn.commit()
                print(f"  ADDED {col_name} {col_type}")

    engine.dispose()
    print("Migration complete.")


if __name__ == "__main__":
    migrate()

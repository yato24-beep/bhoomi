"""SQLite DB Column Migration Script.

Ensures extracted_fields has the required columns:
- english_value
- translation_status
- translation_engine
"""

import glob
import sqlite3
from pathlib import Path


def migrate():
    repo_root = Path(__file__).resolve().parent.parent
    db_paths = list(repo_root.glob("**/*.db"))
    print(f"Found {len(db_paths)} .db files to check.")

    for p in db_paths:
        try:
            conn = sqlite3.connect(p)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='extracted_fields';")
            if not cur.fetchone():
                conn.close()
                continue

            cur.execute("PRAGMA table_info(extracted_fields);")
            cols = [row[1] for row in cur.fetchall()]
            print(f"Checking {p.relative_to(repo_root)} (columns: {len(cols)})")

            for new_col in ("english_value", "translation_status", "translation_engine", "source_type"):
                if new_col not in cols:
                    print(f"  Adding missing column '{new_col}' to {p.name}")
                    cur.execute(f"ALTER TABLE extracted_fields ADD COLUMN {new_col} TEXT;")

            conn.commit()
            conn.close()
            print(f"Migration completed for {p.name}")
        except Exception as e:
            print(f"Error checking {p}: {e}")


if __name__ == "__main__":
    migrate()

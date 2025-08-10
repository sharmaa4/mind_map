import sqlite3
from pathlib import Path
import os
from datetime import datetime

DB_FILENAME = "notes.db"

def _get_db_path():
    return Path(DB_FILENAME)

def migrate_database():
    """Check and update DB schema to match the latest app requirements."""
    db_path = _get_db_path()
    if not db_path.exists():
        return  # No DB yet — nothing to migrate

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Get existing columns
    cursor.execute("PRAGMA table_info(notes)")
    existing_cols = [col[1] for col in cursor.fetchall()]

    # Add 'created_at' column if missing
    if "created_at" not in existing_cols:
        cursor.execute(
            "ALTER TABLE notes ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        )
        # Backfill existing rows with current time
        cursor.execute(
            "UPDATE notes SET created_at = ? WHERE created_at IS NULL",
            (datetime.now(),)
        )
        conn.commit()

    conn.close()

# ---- Your other DB functions remain unchanged below ----

def get_all_notes_with_details(limit: int = 200):
    db_path = _get_db_path()
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, note_type, title, content_path, links, created_at
        FROM notes
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows


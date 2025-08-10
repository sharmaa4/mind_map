# database.py

import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path
import os
import json
from typing import Optional

# Use a deterministic base notes folder relative to cwd (avoid surprises with relative paths)
BASE_NOTES_DIR = Path.cwd() / "notes"

# --- Configuration for Note Categories ---
NOTE_CATEGORIES = {
    "personal": {"emoji": "📝", "needs_embedding": True, "audio": False, "priority": 1},
    "achievement": {"emoji": "🏆", "needs_embedding": True, "audio": False, "priority": 2},
    "mistake": {"emoji": "⚠️", "needs_embedding": True, "audio": False, "priority": 3},
    "youtube_summary": {"emoji": "📺", "needs_embedding": True, "audio": True, "priority": 4},
    "youtube_mindmap": {"emoji": "🧠", "needs_embedding": True, "audio": False, "priority": 5},
}

# --- Helper utilities ---


def _ensure_dirs_exist():
    """
    Creates the notes directory structure if missing.
    """
    BASE_NOTES_DIR.mkdir(parents=True, exist_ok=True)
    for category in NOTE_CATEGORIES.keys():
        (BASE_NOTES_DIR / category).mkdir(exist_ok=True)
    (BASE_NOTES_DIR / "audio_summaries").mkdir(exist_ok=True)
    (BASE_NOTES_DIR / "metadata").mkdir(exist_ok=True)
    (BASE_NOTES_DIR / "embeddings").mkdir(exist_ok=True)


def _get_db_path() -> Path:
    return BASE_NOTES_DIR / "metadata" / "notes_database.db"


# --- Database initialization & migration ---


def init_advanced_notes_database() -> str:
    """
    Initialize the advanced notes database with migration support.
    Ensures the folder structure exists and database schema is present.
    Returns the DB path as a string.
    """
    _ensure_dirs_exist()
    db_path = _get_db_path()
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            note_type TEXT,
            title TEXT,
            content_path TEXT UNIQUE,
            links TEXT,
            tags TEXT,
            file_hash TEXT UNIQUE,
            file_size INTEGER,
            search_priority INTEGER DEFAULT 1,
            has_embedding BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embedding_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            note_id INTEGER,
            status TEXT DEFAULT 'pending',
            attempts INTEGER DEFAULT 0,
            last_error TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(note_id) REFERENCES notes(id)
        )
        """
    )

    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS notes_updated_at
        AFTER UPDATE ON notes
        BEGIN
            UPDATE notes SET modified_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
        END;
        """
    )

    conn.commit()
    conn.close()
    return str(db_path)


# --- Scanning Google Drive / local notes and queueing embeddings ---


def scan_and_queue_new_notes():
    """
    Scans notes directories for files not in the database, registers them,
    and queues them for embedding without creating duplicates.

    This function is defensive:
      - It will call init_advanced_notes_database() if DB or required tables are missing.
      - It uses absolute paths tied to BASE_NOTES_DIR to avoid working-dir surprises.
    """
    print("[database] Starting scan_and_queue_new_notes()")
    _ensure_dirs_exist()
    db_path = _get_db_path()

    # Initialize DB if missing
    if not db_path.exists():
        print("[database] DB file missing; initializing database first.")
        init_advanced_notes_database()

    # Quick table existence check; if corrupted/missing, re-init
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('notes','embedding_jobs')"
        )
        found = {r[0] for r in cursor.fetchall()}
        missing = set(["notes", "embedding_jobs"]) - found
        if missing:
            print(f"[database] Missing tables: {missing}. Re-initializing DB schema.")
            conn.close()
            init_advanced_notes_database()
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
    except Exception as e:
        print(f"[database] Error accessing DB: {e}. Re-initializing DB.")
        try:
            conn.close()
        except:
            pass
        init_advanced_notes_database()
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

    all_note_files = []
    for category in NOTE_CATEGORIES.keys():
        category_path = BASE_NOTES_DIR / category
        if category_path.exists():
            all_note_files.extend(sorted(category_path.glob("*.txt")))

    print(f"[database] Found {len(all_note_files)} local note files across categories.")

    for note_file in all_note_files:
        try:
            # Read content
            with open(note_file, "r", encoding="utf-8") as f:
                content = f.read()

            if not content:
                print(f"[database] Skipping empty file: {note_file}")
                continue

            file_hash = hashlib.md5(content.encode("utf-8")).hexdigest()

            # Skip if file hash already in DB
            cursor.execute("SELECT id FROM notes WHERE file_hash = ?", (file_hash,))
            if cursor.fetchone():
                # Already registered
                continue

            print(f"[database] New note found: {note_file.name}. Registering in DB.")

            lines = content.splitlines()
            title = "Untitled Note"
            note_type = "personal"
            tags = ""
            links = ""

            for line in lines:
                low = line.strip().lower()
                if low.startswith("title:"):
                    title = line.split(":", 1)[1].strip()
                elif low.startswith("category:"):
                    nt = line.split(":", 1)[1].strip().lower()
                    if nt in NOTE_CATEGORIES:
                        note_type = nt
                elif low.startswith("tags:"):
                    tags = line.split(":", 1)[1].strip()
                elif low.startswith("links:"):
                    links = line.split(":", 1)[1].strip()

            file_size = note_file.stat().st_size
            search_priority = NOTE_CATEGORIES.get(note_type, {}).get("priority", 1)
            content_path = str(note_file.resolve())

            cursor.execute(
                """
                INSERT INTO notes (note_type, title, content_path, links, tags, file_hash, file_size, search_priority, has_embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (note_type, title, content_path, links, tags, file_hash, file_size, search_priority, False),
            )

            note_id = cursor.lastrowid

            if NOTE_CATEGORIES.get(note_type, {}).get("needs_embedding", False):
                cursor.execute(
                    "INSERT INTO embedding_jobs (note_id, status) VALUES (?, ?)",
                    (note_id, "pending"),
                )

        except Exception as e:
            print(f"[database] Error processing file {note_file.name}: {e}")

    conn.commit()
    conn.close()
    print("[database] scan_and_queue_new_notes() finished.")


# --- Functions used by app when creating a new note in UI ---


def save_advanced_note(note_type, title, content, links="", tags="", audio_file=None) -> int:
    """
    Save a manual note created in the sidebar UI.
    Returns the created note_id.
    """
    if note_type not in NOTE_CATEGORIES:
        raise ValueError(f"Invalid note type: {note_type}")

    _ensure_dirs_exist()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = "".join(c for c in title if c.isalnum() or c in (" ", "_", "-")).strip()[:200]
    filename = f"{timestamp}_{safe_title or 'note'}.txt"
    content_path = BASE_NOTES_DIR / note_type / filename

    formatted = f"Title: {title}\nCategory: {note_type}\nCreated: {datetime.now().isoformat()}\nLinks: {links}\n{'-'*50}\n\n{content}"
    with open(content_path, "w", encoding="utf-8") as f:
        f.write(formatted)

    file_hash = hashlib.md5(formatted.encode("utf-8")).hexdigest()
    file_size = content_path.stat().st_size

    db_path = _get_db_path()
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO notes (note_type, title, content_path, links, tags, file_hash, file_size, search_priority, has_embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (note_type, title, str(content_path.resolve()), links, tags, file_hash, file_size, NOTE_CATEGORIES[note_type]["priority"], False),
    )
    note_id = cursor.lastrowid

    if NOTE_CATEGORIES[note_type]["needs_embedding"]:
        cursor.execute("INSERT INTO embedding_jobs (note_id, status) VALUES (?, ?)", (note_id, "pending"))

    conn.commit()
    conn.close()
    return note_id


# --- Utility getters ---


def get_all_notes_with_details(limit: int = 100):
    db_path = _get_db_path()
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT id, note_type, title, content_path, links, tags, has_embedding FROM notes ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_pending_embedding_jobs(limit: int = 50):
    db_path = _get_db_path()
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT id, note_id, status FROM embedding_jobs WHERE status = 'pending' ORDER BY created_at ASC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows


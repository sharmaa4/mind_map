import sqlite3
from pathlib import Path
from datetime import datetime

DB_FILENAME = "notes.db"
DB_PATH = Path(DB_FILENAME)

# Define the columns your notes table must always have
REQUIRED_COLUMNS = {
    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
    "note_type": "TEXT",
    "title": "TEXT",
    "content_path": "TEXT",
    "links": "TEXT",
    "created_at": "TEXT"
}

def _get_db_path() -> Path:
    return DB_PATH

def _initialize_database():
    """Create database with all required columns if it doesn't exist."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    columns_sql = ", ".join([f"{col} {ctype}" for col, ctype in REQUIRED_COLUMNS.items()])
    cursor.execute(f"CREATE TABLE IF NOT EXISTS notes ({columns_sql})")

    conn.commit()
    conn.close()

def migrate_database():
    """
    Ensures all required columns exist in the database.
    Adds missing columns with default values and backfills where needed.
    """
    _initialize_database()

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # Get existing columns
    cursor.execute("PRAGMA table_info(notes)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    # Add missing columns
    for col, col_type in REQUIRED_COLUMNS.items():
        if col not in existing_columns:
            cursor.execute(f"ALTER TABLE notes ADD COLUMN {col} {col_type}")
            # If created_at, backfill with current time
            if col == "created_at":
                cursor.execute(
                    "UPDATE notes SET created_at = ? WHERE created_at IS NULL OR created_at = ''",
                    (datetime.now().isoformat(),)
                )

    conn.commit()
    conn.close()

def insert_note(note_type: str, title: str, content: str, created_at: datetime = None) -> int:
    """Insert a new note into the DB."""
    if created_at is None:
        created_at = datetime.now()

    db_path = _get_db_path()
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Save content to file
    content_file = Path(f"note_{datetime.now().timestamp()}.txt")
    with open(content_file, "w") as f:
        f.write(content)

    cursor.execute(
        "INSERT INTO notes (note_type, title, content_path, created_at) VALUES (?, ?, ?, ?)",
        (note_type, title, str(content_file), created_at.isoformat())
    )

    conn.commit()
    note_id = cursor.lastrowid
    conn.close()

    return note_id

def get_all_notes_with_details(limit: int = 200):
    """Fetch all notes with details, ordered by created_at desc."""
    db_path = _get_db_path()
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id, note_type, title, content_path, links, created_at
        FROM notes
        ORDER BY datetime(created_at) DESC
        LIMIT ?
        """,
        (limit,)
    )

    rows = cursor.fetchall()
    conn.close()
    return rows


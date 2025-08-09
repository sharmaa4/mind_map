# database.py

import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path
import os
import json

# --- Configuration for Note Categories ---
NOTE_CATEGORIES = {
    "personal": {"emoji": "📝", "needs_embedding": True, "audio": False, "priority": 1},
    "achievement": {"emoji": "🏆", "needs_embedding": True, "audio": False, "priority": 2},
    "mistake": {"emoji": "⚠️", "needs_embedding": True, "audio": False, "priority": 3},
    "youtube_summary": {"emoji": "📺", "needs_embedding": True, "audio": True, "priority": 4},
    "youtube_mindmap": {"emoji": "🧠", "needs_embedding": True, "audio": False, "priority": 5},
    "lecture_notes": {"emoji": "📚", "needs_embedding": True, "audio": True, "priority": 6},
    "lecture_mindmap": {"emoji": "🗺️", "needs_embedding": True, "audio": False, "priority": 7}
}

def migrate_database_to_phase3(db_path):
    """Migrate existing database to Phase 3+ schema."""
    if not Path(db_path).exists():
        return
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT search_priority FROM notes LIMIT 1")
        conn.close()
        return
    except sqlite3.OperationalError:
        print("INFO: Migrating database to Phase 3+ schema...")
    
    new_columns = [
        ("tags", "TEXT DEFAULT ''"),
        ("last_modified", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
        ("embedding_dimension", "INTEGER DEFAULT 384"),
        ("search_priority", "INTEGER DEFAULT 1")
    ]
    for col_name, col_def in new_columns:
        try:
            cursor.execute(f"ALTER TABLE notes ADD COLUMN {col_name} {col_def}")
        except sqlite3.OperationalError: pass
    
    for note_type, config in NOTE_CATEGORIES.items():
        cursor.execute("UPDATE notes SET search_priority = ? WHERE note_type = ?", (config["priority"], note_type))

    embedding_columns = [
        ("processing_status", "TEXT DEFAULT 'pending'"),
        ("error_message", "TEXT"),
        ("embedding_id", "TEXT")
    ]
    for col_name, col_def in embedding_columns:
        try:
            cursor.execute(f"ALTER TABLE embedding_status ADD COLUMN {col_name} {col_def}")
        except sqlite3.OperationalError: pass

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embedding_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, note_id INTEGER, status TEXT DEFAULT 'pending',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP, started_at DATETIME, completed_at DATETIME,
            error_message TEXT, FOREIGN KEY (note_id) REFERENCES notes (id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS note_relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT, source_note_id INTEGER, target_note_id INTEGER,
            relationship_type TEXT, confidence_score REAL, created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (source_note_id) REFERENCES notes (id), FOREIGN KEY (target_note_id) REFERENCES notes (id)
        )
    """)
    conn.commit()
    conn.close()
    print("INFO: Database migration to Phase 3+ complete!")

def init_advanced_notes_database():
    """Initialize the advanced notes database with migration support."""
    base_dir = Path("notes")
    base_dir.mkdir(exist_ok=True)
    for category in NOTE_CATEGORIES.keys():
        (base_dir / category).mkdir(exist_ok=True)
    (base_dir / "audio_summaries").mkdir(exist_ok=True)
    (base_dir / "metadata").mkdir(exist_ok=True)
    (base_dir / "embeddings").mkdir(exist_ok=True)
    db_path = base_dir / "metadata" / "notes_database.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT, note_type TEXT NOT NULL, title TEXT NOT NULL,
            content_path TEXT, audio_path TEXT, links TEXT, tags TEXT DEFAULT '',
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, last_modified DATETIME DEFAULT CURRENT_TIMESTAMP,
            has_embedding BOOLEAN DEFAULT FALSE, embedding_model TEXT, file_hash TEXT,
            file_size INTEGER, embedding_dimension INTEGER DEFAULT 384, search_priority INTEGER DEFAULT 1
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embedding_status (
            id INTEGER PRIMARY KEY AUTOINCREMENT, file_path TEXT UNIQUE, needs_embedding BOOLEAN DEFAULT TRUE,
            last_checked DATETIME DEFAULT CURRENT_TIMESTAMP, embedding_model TEXT, file_hash TEXT,
            processing_status TEXT DEFAULT 'pending', error_message TEXT, embedding_id TEXT
        )
    """)
    migrate_database_to_phase3(db_path)
    conn.commit()
    conn.close()
    return str(db_path)

def scan_and_queue_new_notes():
    """
    Scans notes directories for files not in the database, registers them,
    and queues them for embedding without creating duplicates.
    """
    print("Scanning for new notes from Google Drive sync...")
    db_path = Path("notes") / "metadata" / "notes_database.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    all_note_files = []
    for category in NOTE_CATEGORIES.keys():
        category_path = Path("notes") / category
        if category_path.exists():
            all_note_files.extend(list(category_path.glob("*.txt")))

    for note_file in all_note_files:
        try:
            with open(note_file, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content: continue

            file_hash = hashlib.md5(content.encode()).hexdigest()
            cursor.execute("SELECT id FROM notes WHERE file_hash = ?", (file_hash,))
            if cursor.fetchone():
                continue

            print(f"New note found: {note_file.name}. Registering in database.")
            
            lines = content.splitlines()
            title = "Untitled Note"
            note_type = "personal"
            tags = ""
            links = ""
            for line in lines:
                if line.lower().startswith("title:"):
                    title = line.split(":", 1)[1].strip()
                elif line.lower().startswith("category:"):
                    note_type_from_file = line.split(":", 1)[1].strip().lower()
                    if note_type_from_file in NOTE_CATEGORIES:
                        note_type = note_type_from_file
                elif line.lower().startswith("tags:"):
                    tags = line.split(":", 1)[1].strip()
                elif line.lower().startswith("links:"):
                    links = line.split(":", 1)[1].strip()

            file_size = note_file.stat().st_size
            search_priority = NOTE_CATEGORIES.get(note_type, {}).get("priority", 1)
            content_path = str(note_file)

            cursor.execute("""
                INSERT INTO notes (note_type, title, content_path, links, tags, file_hash, file_size, search_priority, has_embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (note_type, title, content_path, links, tags, file_hash, file_size, search_priority, False))
            
            note_id = cursor.lastrowid
            
            if NOTE_CATEGORIES[note_type]["needs_embedding"]:
                cursor.execute("INSERT INTO embedding_jobs (note_id, status) VALUES (?, ?)", (note_id, 'pending'))

        except Exception as e:
            print(f"Error processing file {note_file.name}: {e}")

    conn.commit()
    conn.close()

def save_advanced_note(note_type, title, content, links="", tags="", audio_file=None):
    if note_type not in NOTE_CATEGORIES:
        raise ValueError(f"Invalid note type: {note_type}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')[:50]
    content_filename = f"{timestamp}_{safe_title}.txt"
    content_path = Path("notes") / note_type / content_filename
    
    formatted_content = f"Title: {title}\nCategory: {note_type}\nTags: {tags}\nCreated: {datetime.now().isoformat()}\nLinks: {links}\n{'-' * 50}\n\n{content}"
    with open(content_path, 'w', encoding='utf-8') as f:
        f.write(formatted_content)
    
    content_hash = hashlib.md5(formatted_content.encode()).hexdigest()
    file_size = content_path.stat().st_size
    search_priority = NOTE_CATEGORIES[note_type]["priority"]
    audio_path = None
    if audio_file and NOTE_CATEGORIES[note_type]["audio"]:
        audio_filename = f"{timestamp}_{safe_title}.mp3"
        audio_path = str(Path("notes") / "audio_summaries" / audio_filename)
    
    db_path = Path("notes") / "metadata" / "notes_database.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO notes (note_type, title, content_path, audio_path, links, tags, has_embedding, embedding_model, file_hash, file_size, search_priority)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (note_type, title, str(content_path), audio_path, links, tags, False, None, content_hash, file_size, search_priority))
    note_id = cursor.lastrowid
    
    if NOTE_CATEGORIES[note_type]["needs_embedding"]:
        cursor.execute("INSERT INTO embedding_jobs (note_id, status) VALUES (?, ?)", (note_id, 'pending'))
    
    conn.commit()
    conn.close()
    return note_id, str(content_path)

def get_advanced_notes_stats():
    db_path = Path("notes") / "metadata" / "notes_database.db"
    if not db_path.exists():
        return {"total_notes": 0, "needs_embedding": 0, "has_embedding": 0, "pending_jobs": 0}
    conn = sqlite3.connect(str(db_path))
    try:
        total_notes = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        needs_embedding = conn.execute("SELECT COUNT(*) FROM embedding_jobs WHERE status = 'pending'").fetchone()[0]
        has_embedding = conn.execute("SELECT COUNT(*) FROM notes WHERE has_embedding = TRUE").fetchone()[0]
        pending_jobs = needs_embedding
        return {"total_notes": total_notes, "needs_embedding": needs_embedding, "has_embedding": has_embedding, "pending_jobs": pending_jobs}
    finally:
        conn.close()

def get_all_notes_with_details():
    db_path = Path("notes") / "metadata" / "notes_database.db"
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    try:
        notes = conn.execute("""
            SELECT id, note_type, title, content_path, COALESCE(links, '') as links, COALESCE(tags, '') as tags,
                   timestamp, COALESCE(last_modified, timestamp) as last_modified, has_embedding
            FROM notes ORDER BY timestamp DESC
        """).fetchall()
        return [{"id": n[0], "type": n[1], "title": n[2], "content_path": n[3], "links": n[4], "tags": n[5],
                 "timestamp": n[6], "last_modified": n[7], "has_embedding": n[8]} for n in notes]
    finally:
        conn.close()

def delete_note(note_id):
    db_path = Path("notes") / "metadata" / "notes_database.db"
    conn = sqlite3.connect(str(db_path))
    try:
        note_info = conn.execute("SELECT content_path, COALESCE(audio_path, '') FROM notes WHERE id = ?", (note_id,)).fetchone()
        if note_info:
            content_path, audio_path = note_info
            if content_path and Path(content_path).exists(): Path(content_path).unlink()
            if audio_path and Path(audio_path).exists(): Path(audio_path).unlink()
            conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
            conn.execute("DELETE FROM embedding_status WHERE file_path = ?", (content_path,))
            conn.execute("DELETE FROM embedding_jobs WHERE note_id = ?", (note_id,))
            conn.commit()
    finally:
        conn.close()

# The following functions are not used in the main app flow but are included for completeness
def get_recent_notes(limit=5):
    db_path = Path("notes") / "metadata" / "notes_database.db"
    if not db_path.exists(): return []
    conn = sqlite3.connect(str(db_path))
    try:
        notes = conn.execute("SELECT id, note_type, title, timestamp, has_embedding FROM notes ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
        return [{"id": n[0], "type": n[1], "title": n[2], "timestamp": n[3], "has_embedding": n[4]} for n in notes]
    finally:
        conn.close()

def get_note_content(note_id):
    db_path = Path("notes") / "metadata" / "notes_database.db"
    conn = sqlite3.connect(str(db_path))
    try:
        note_info = conn.execute("SELECT note_type, title, content_path, COALESCE(links, '') as links, COALESCE(tags, '') as tags, timestamp, has_embedding FROM notes WHERE id = ?", (note_id,)).fetchone()
        if not note_info: return None
        content_path = note_info[2]
        content = "Content file not found"
        if content_path and Path(content_path).exists():
            with open(content_path, 'r', encoding='utf-8') as f:
                full_content = f.read()
            content = full_content.split("-" * 50, 1)[1].strip() if "----" in full_content else full_content
        return {"id": note_id, "type": note_info[0], "title": note_info[1], "content": content, "links": note_info[3], "tags": note_info[4], "timestamp": note_info[5], "has_embedding": note_info[6]}
    finally:
        conn.close()

def update_note(note_id, title, content, links="", tags=""):
    db_path = Path("notes") / "metadata" / "notes_database.db"
    conn = sqlite3.connect(str(db_path))
    try:
        note_info = conn.execute("SELECT content_path, note_type FROM notes WHERE id = ?", (note_id,)).fetchone()
        if note_info:
            content_path, note_type = note_info
            formatted_content = f"Title: {title}\nCategory: {note_type}\nTags: {tags}\nCreated: {datetime.now().isoformat()}\nLinks: {links}\n{'-' * 50}\n\n{content}"
            with open(content_path, 'w', encoding='utf-8') as f:
                f.write(formatted_content)
            new_hash = hashlib.md5(formatted_content.encode()).hexdigest()
            file_size = Path(content_path).stat().st_size
            conn.execute("""
                UPDATE notes SET title = ?, links = ?, tags = ?, file_hash = ?, file_size = ?, last_modified = CURRENT_TIMESTAMP, has_embedding = FALSE WHERE id = ?
            """, (title, links, tags, new_hash, file_size, note_id))
            conn.execute("UPDATE embedding_status SET needs_embedding = TRUE, file_hash = ?, last_checked = CURRENT_TIMESTAMP, processing_status = 'pending' WHERE file_path = ?", (new_hash, content_path))
            conn.execute("INSERT INTO embedding_jobs (note_id, status) VALUES (?, ?)", (note_id, 'pending'))
            conn.commit()
    finally:
        conn.close()

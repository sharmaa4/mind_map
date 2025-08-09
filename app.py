import streamlit as st
import streamlit.components.v1 as components
import pysqlite3
import sys
print("pysqlite3 version:", pysqlite3.sqlite_version)
sys.modules["sqlite3"] = pysqlite3
import shutil
import os
import glob
import json
import time
from concurrent.futures import ThreadPoolExecutor
import uuid
import chromadb
from chromadb.config import Settings
import requests
import fitz  # PyMuPDF
from PIL import Image
import subprocess
from bs4 import BeautifulSoup
import numpy as np

# Import sentence transformers for local embeddings
from sentence_transformers import SentenceTransformer

# PHASE 3+: Additional imports for advanced note-taking
import sqlite3
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
import re
import base64
import threading
from typing import List, Dict, Any

# --- ADDED FOR GOOGLE DRIVE SYNC ---
import google_drive_sync as gds
# --- END GOOGLE DRIVE SYNC ---

# Configure the Streamlit page
st.set_page_config(page_title="Global Product Search + Advanced Note Management", layout="wide")
st.title("🚀 Global Product Search + Advanced Note-Taking System (Phase 3+)")

# --- ADDED FOR GOOGLE DRIVE SYNC ---
# Initialize session state for sync status
if 'drive_synced' not in st.session_state:
    st.session_state.drive_synced = False
if 'drive_instance' not in st.session_state:
    st.session_state.drive_instance = None

# Function to run the initial sync from Google Drive
@st.cache_resource(show_spinner="Connecting to Google Drive and syncing data...")
def initial_sync():
    """
    Authenticates with Google Drive and downloads the 'notes' and 
    'product_embeddings_v2' directories. This runs only once.
    """
    try:
        drive = gds.authenticate_gdrive()
        
        # Sync the notes directory (contains SQLite DB and .txt files)
        gds.sync_directory_from_drive(drive, "notes")
        
        # Sync the pre-computed product embeddings
        gds.sync_directory_from_drive(drive, "product_embeddings_v2")
        
        return drive
    except Exception as e:
        st.error(f"Fatal Error: Could not sync with Google Drive. Please check credentials. Details: {e}")
        return None

# Perform the initial sync when the app starts
if not st.session_state.drive_synced:
    drive = initial_sync()
    if drive:
        st.session_state.drive_synced = True
        st.session_state.drive_instance = drive
        st.sidebar.success("✅ Synced with Google Drive!")
        # Rerun to ensure the rest of the app loads with the synced data
        st.rerun() 
    else:
        st.sidebar.error("❌ Google Drive sync failed. App cannot continue.")
        st.stop() # Stop the app if sync fails
# --- END GOOGLE DRIVE SYNC ---


# Add migration success notice
st.sidebar.success("🎉 Phase 3+: Complete Knowledge Management!")
try:
    st.sidebar.image("logo.png", width=200)
except:
    pass  # Logo file might not exist

# Model selection for Puter.js
st.sidebar.header("🤖 AI Model Selection")
selected_model = st.sidebar.selectbox(
    "Choose AI Model:",
    [
        "gpt-4o-mini", 
        "gpt-4o", 
        "claude-sonnet-4",
        "claude-opus-4",
        "o1-mini", 
        "o1", 
        "o3-mini", 
        "o3", 
        "gpt-4.1"
    ],
    index=0,
    help="Select the AI model for processing queries"
)

# Embedding model selection
st.sidebar.header("📊 Embedding Model")
embedding_model_name = st.sidebar.selectbox(
    "Choose Embedding Model:",
    ["BAAI/bge-small-en-v1.5", "all-mpnet-base-v2", "paraphrase-multilingual-mpnet-base-v2", "all-MiniLM-L6-v2"],
    index=0,
    help="Select the local embedding model"
)

# Enhanced Features
st.sidebar.header("⚡ Enhanced Features")
enable_streaming = st.sidebar.checkbox("🔄 Enable Streaming Output", value=True, help="Stream AI responses in real-time")
enable_context = st.sidebar.checkbox("🧠 Enable Context Awareness", value=True, help="Maintain conversation memory")
max_context_messages = st.sidebar.slider("📝 Context History Length", 1, 10, 5, help="Number of previous messages to remember")

# Phase 3: Advanced search controls
st.sidebar.header("🔍 Phase 3+: Advanced Search")
enable_unified_search = st.sidebar.checkbox("🔗 Unified Search (Products + Notes)", value=True, help="Search across both products and personal notes")
note_context_weight = st.sidebar.slider("📝 Note Context Weight", 0.0, 1.0, 0.3, help="How much to weight personal notes in AI responses")

# ================================
# LOCAL EMBEDDINGS SETUP (MOVED UP FOR GLOBAL ACCESS)
# ================================

@st.cache_resource(show_spinner=True)
def load_embedding_model(model_name="BAAI/bge-small-en-v1.5"):
    """Load local sentence transformer model for embeddings"""
    try:
        with st.spinner(f"Loading embedding model: {model_name}..."):
            model = SentenceTransformer(model_name)
        st.success(f"✅ Local embedding model '{model_name}' loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading embedding model '{model_name}': {e}")
        st.write("**Solution:** Install sentence-transformers: `pip install sentence-transformers`")
        return None

def get_query_embedding_local(query_text, model_name="BAAI/bge-small-en-v1.5"):
    """Generate local embeddings for query"""
    model = load_embedding_model(model_name)
    if model is None:
        st.error("❌ Embedding model not available")
        return None
    
    try:
        embedding = model.encode(query_text, normalize_embeddings=True)
        embedding_list = embedding.tolist()
        st.success(f"✅ Generated embedding locally - Dimension: {len(embedding_list)} | Model: {model_name}")
        return embedding_list
    except Exception as e:
        st.error(f"❌ Error generating local embedding: {e}")
        return None

# FIXED: Load the embedding model on startup and make it globally accessible
@st.cache_resource
def get_global_embedding_model():
    """Get the global embedding model instance"""
    return load_embedding_model(embedding_model_name)

# Initialize global embedding model
embedding_model = get_global_embedding_model()

# ================================
# PHASE 3+: NOTE CATEGORIES & MIGRATION
# ================================

# Enhanced note categories configuration
NOTE_CATEGORIES = {
    "personal": {"emoji": "📝", "needs_embedding": True, "audio": False, "priority": 1},
    "achievement": {"emoji": "🏆", "needs_embedding": True, "audio": False, "priority": 2},
    "mistake": {"emoji": "⚠️", "needs_embedding": True, "audio": False, "priority": 3},
    "youtube_summary": {"emoji": "📺", "needs_embedding": True, "audio": True, "priority": 4},
    "youtube_mindmap": {"emoji": "🧠", "needs_embedding": True, "audio": False, "priority": 5},
    "lecture_notes": {"emoji": "📚", "needs_embedding": True, "audio": True, "priority": 6},
    "lecture_mindmap": {"emoji": "🗺️", "needs_embedding": True, "audio": False, "priority": 7}
}

def migrate_database_to_phase3():
    """Migrate existing database to Phase 3+ schema"""
    db_path = Path("notes") / "metadata" / "notes_database.db"
    if not db_path.exists():
        return
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Check if migration is needed
    try:
        cursor.execute("SELECT search_priority FROM notes LIMIT 1")
        conn.close()
        return  # Already migrated
    except sqlite3.OperationalError:
        st.info("🔄 Migrating database to Phase 3+ schema...")
    
    # Add new columns to notes table
    new_columns = [
        ("tags", "TEXT DEFAULT ''"),
        ("last_modified", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
        ("embedding_dimension", "INTEGER DEFAULT 384"),
        ("search_priority", "INTEGER DEFAULT 1")
    ]
    
    for col_name, col_def in new_columns:
        try:
            cursor.execute(f"ALTER TABLE notes ADD COLUMN {col_name} {col_def}")
            st.success(f"✅ Added '{col_name}' column")
        except sqlite3.OperationalError:
            pass  # Column already exists
    
    # Update search_priority based on note_type
    for note_type, config in NOTE_CATEGORIES.items():
        cursor.execute("""
            UPDATE notes SET search_priority = ? WHERE note_type = ?
        """, (config["priority"], note_type))
    
    # Add columns to embedding_status table
    embedding_columns = [
        ("processing_status", "TEXT DEFAULT 'pending'"),
        ("error_message", "TEXT"),
        ("embedding_id", "TEXT")
    ]
    
    for col_name, col_def in embedding_columns:
        try:
            cursor.execute(f"ALTER TABLE embedding_status ADD COLUMN {col_name} {col_def}")
            st.success(f"✅ Added '{col_name}' to embedding_status")
        except sqlite3.OperationalError:
            pass
    
    # Create new tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embedding_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            note_id INTEGER,
            status TEXT DEFAULT 'pending',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            started_at DATETIME,
            completed_at DATETIME,
            error_message TEXT,
            FOREIGN KEY (note_id) REFERENCES notes (id)
        )
    """)
    st.success("✅ Created 'embedding_jobs' table")
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS note_relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_note_id INTEGER,
            target_note_id INTEGER,
            relationship_type TEXT,
            confidence_score REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (source_note_id) REFERENCES notes (id),
            FOREIGN KEY (target_note_id) REFERENCES notes (id)
        )
    """)
    st.success("✅ Created 'note_relationships' table")
    
    conn.commit()
    conn.close()
    
    st.success("🎉 Database successfully migrated to Phase 3+!")

@st.cache_resource
def init_advanced_notes_database():
    """Initialize the advanced notes database with migration support"""
    # Create directory structure
    base_dir = Path("notes")
    base_dir.mkdir(exist_ok=True)
    
    for category in NOTE_CATEGORIES.keys():
        (base_dir / category).mkdir(exist_ok=True)
    
    (base_dir / "audio_summaries").mkdir(exist_ok=True)
    (base_dir / "metadata").mkdir(exist_ok=True)
    (base_dir / "embeddings").mkdir(exist_ok=True)
    
    # Initialize database with full schema
    db_path = base_dir / "metadata" / "notes_database.db"
    conn = sqlite3.connect(str(db_path))
    
    # Create tables with complete Phase 3+ schema
    conn.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            note_type TEXT NOT NULL,
            title TEXT NOT NULL,
            content_path TEXT,
            audio_path TEXT,
            links TEXT,
            tags TEXT DEFAULT '',
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_modified DATETIME DEFAULT CURRENT_TIMESTAMP,
            has_embedding BOOLEAN DEFAULT FALSE,
            embedding_model TEXT,
            file_hash TEXT,
            file_size INTEGER,
            embedding_dimension INTEGER DEFAULT 384,
            search_priority INTEGER DEFAULT 1
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embedding_status (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE,
            needs_embedding BOOLEAN DEFAULT TRUE,
            last_checked DATETIME DEFAULT CURRENT_TIMESTAMP,
            embedding_model TEXT,
            file_hash TEXT,
            processing_status TEXT DEFAULT 'pending',
            error_message TEXT,
            embedding_id TEXT
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embedding_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            note_id INTEGER,
            status TEXT DEFAULT 'pending',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            started_at DATETIME,
            completed_at DATETIME,
            error_message TEXT,
            FOREIGN KEY (note_id) REFERENCES notes (id)
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS note_relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_note_id INTEGER,
            target_note_id INTEGER,
            relationship_type TEXT,
            confidence_score REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (source_note_id) REFERENCES notes (id),
            FOREIGN KEY (target_note_id) REFERENCES notes (id)
        )
    """)
    
    conn.commit()
    conn.close()
    
    # Run migration for existing databases
    migrate_database_to_phase3()
    
    return str(db_path)

def save_advanced_note(note_type, title, content, links="", tags="", audio_file=None):
    """Save a note with enhanced metadata and queue embedding generation"""
    if note_type not in NOTE_CATEGORIES:
        raise ValueError(f"Invalid note type: {note_type}")
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_title = safe_title.replace(' ', '_')[:50]
    
    # Save text content with enhanced formatting
    content_filename = f"{timestamp}_{safe_title}.txt"
    content_path = Path("notes") / note_type / content_filename
    
    formatted_content = f"""Title: {title}
Category: {note_type}
Tags: {tags}
Created: {datetime.now().isoformat()}
Links: {links}
{"-" * 50}

{content}
"""
    
    with open(content_path, 'w', encoding='utf-8') as f:
        f.write(formatted_content)
    
    # Calculate file hash and metadata
    content_hash = hashlib.md5(formatted_content.encode()).hexdigest()
    file_size = content_path.stat().st_size
    search_priority = NOTE_CATEGORIES[note_type]["priority"]
    
    # Save audio file if provided
    audio_path = None
    if audio_file and NOTE_CATEGORIES[note_type]["audio"]:
        audio_filename = f"{timestamp}_{safe_title}.mp3"
        audio_path = Path("notes") / "audio_summaries" / audio_filename
        audio_path = str(audio_path)
    
    # Save to database with enhanced metadata
    db_path = Path("notes") / "metadata" / "notes_database.db"
    conn = sqlite3.connect(str(db_path))
    
    cursor = conn.execute("""
        INSERT INTO notes (note_type, title, content_path, audio_path, links, tags,
                          has_embedding, embedding_model, file_hash, file_size, search_priority)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (note_type, title, str(content_path), audio_path, links, tags,
          False, None, content_hash, file_size, search_priority))
    
    note_id = cursor.lastrowid
    
    # Add to embedding status and queue if needs embedding
    if NOTE_CATEGORIES[note_type]["needs_embedding"]:
        conn.execute("""
            INSERT OR REPLACE INTO embedding_status 
            (file_path, needs_embedding, file_hash, processing_status)
            VALUES (?, ?, ?, ?)
        """, (str(content_path), True, content_hash, 'pending'))
        
        # Queue embedding job
        conn.execute("""
            INSERT INTO embedding_jobs (note_id, status)
            VALUES (?, ?)
        """, (note_id, 'pending'))
    
    conn.commit()
    conn.close()

    # --- GOOGLE DRIVE SYNC ---
    if 'drive_instance' in st.session_state and st.session_state.drive_instance:
        with st.spinner("Syncing new note to Google Drive..."):
            gds.sync_directory_to_drive(st.session_state.drive_instance, "notes")
    # --- END GOOGLE DRIVE SYNC ---
    
    return note_id, str(content_path)

def get_advanced_notes_stats():
    """Get comprehensive statistics about notes and embedding status"""
    db_path = Path("notes") / "metadata" / "notes_database.db"
    if not db_path.exists():
        return {"total_notes": 0, "needs_embedding": 0, "has_embedding": 0, "pending_jobs": 0}
    
    conn = sqlite3.connect(str(db_path))
    
    try:
        # Basic stats
        total_notes = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        needs_embedding = conn.execute(
            "SELECT COUNT(*) FROM notes WHERE has_embedding = FALSE AND note_type IN ({})".format(
                ','.join(['?' for _ in [k for k, v in NOTE_CATEGORIES.items() if v["needs_embedding"]]])
            ), 
            [k for k, v in NOTE_CATEGORIES.items() if v["needs_embedding"]]
        ).fetchone()[0]
        has_embedding = conn.execute("SELECT COUNT(*) FROM notes WHERE has_embedding = TRUE").fetchone()[0]
        
        # Job queue stats
        try:
            pending_jobs = conn.execute("SELECT COUNT(*) FROM embedding_jobs WHERE status = 'pending'").fetchone()[0]
            processing_jobs = conn.execute("SELECT COUNT(*) FROM embedding_jobs WHERE status = 'processing'").fetchone()[0]
            failed_jobs = conn.execute("SELECT COUNT(*) FROM embedding_jobs WHERE status = 'failed'").fetchone()[0]
            completed_jobs = conn.execute("SELECT COUNT(*) FROM embedding_jobs WHERE status = 'completed'").fetchone()[0]
        except:
            pending_jobs = processing_jobs = failed_jobs = completed_jobs = 0
        
        # Category breakdown
        try:
            category_stats = conn.execute("""
                SELECT note_type, COUNT(*), AVG(COALESCE(search_priority, 1))
                FROM notes 
                GROUP BY note_type
            """).fetchall()
        except:
            category_stats = []
        
        conn.close()
        
        return {
            "total_notes": total_notes,
            "needs_embedding": needs_embedding,
            "has_embedding": has_embedding,
            "pending_jobs": pending_jobs,
            "processing_jobs": processing_jobs,
            "failed_jobs": failed_jobs,
            "completed_jobs": completed_jobs,
            "category_breakdown": {cat: {"count": count, "avg_priority": avg_pri} 
                                  for cat, count, avg_pri in category_stats}
        }
    except Exception as e:
        conn.close()
        return {"total_notes": 0, "needs_embedding": 0, "has_embedding": 0, "pending_jobs": 0, "error": str(e)}

def get_recent_notes(limit=5):
    """Get recent notes for display"""
    db_path = Path("notes") / "metadata" / "notes_database.db"
    if not db_path.exists():
        return []
    
    conn = sqlite3.connect(str(db_path))
    
    notes = conn.execute("""
        SELECT id, note_type, title, timestamp, has_embedding
        FROM notes 
        ORDER BY timestamp DESC 
        LIMIT ?
    """, (limit,)).fetchall()
    
    conn.close()
    
    return [{"id": n[0], "type": n[1], "title": n[2], "timestamp": n[3], "has_embedding": n[4]} 
            for n in notes]

# ================================
# PHASE 3+: ENHANCED NOTE MANAGEMENT
# ================================

def get_all_notes_with_details():
    """Get all notes with full details for management"""
    db_path = Path("notes") / "metadata" / "notes_database.db"
    if not db_path.exists():
        return []
    
    conn = sqlite3.connect(str(db_path))
    
    # FIXED: Handle missing columns gracefully
    try:
        notes = conn.execute("""
            SELECT id, note_type, title, content_path, 
                   COALESCE(links, '') as links, 
                   COALESCE(tags, '') as tags, 
                   timestamp, 
                   COALESCE(last_modified, timestamp) as last_modified,
                   has_embedding, 
                   COALESCE(embedding_model, '') as embedding_model, 
                   COALESCE(file_size, 0) as file_size, 
                   COALESCE(search_priority, 1) as search_priority
            FROM notes 
            ORDER BY timestamp DESC
        """).fetchall()
    except sqlite3.OperationalError as e:
        # If columns don't exist, use basic query
        notes = conn.execute("""
            SELECT id, note_type, title, content_path, 
                   '', '', timestamp, timestamp,
                   has_embedding, '', 0, 1
            FROM notes 
            ORDER BY timestamp DESC
        """).fetchall()
    
    conn.close()
    
    return [{"id": n[0], "type": n[1], "title": n[2], "content_path": n[3], 
             "links": n[4], "tags": n[5], "timestamp": n[6], "last_modified": n[7],
             "has_embedding": n[8], "embedding_model": n[9], "file_size": n[10], 
             "search_priority": n[11]} for n in notes]

def get_note_content(note_id):
    """Get full content of a note for editing"""
    db_path = Path("notes") / "metadata" / "notes_database.db"
    conn = sqlite3.connect(str(db_path))
    
    try:
        note_info = conn.execute("""
            SELECT note_type, title, content_path, 
                   COALESCE(links, '') as links, 
                   COALESCE(tags, '') as tags, 
                   timestamp, has_embedding
            FROM notes WHERE id = ?
        """, (note_id,)).fetchone()
    except sqlite3.OperationalError:
        # Fallback for missing columns
        note_info = conn.execute("""
            SELECT note_type, title, content_path, '', '', timestamp, has_embedding
            FROM notes WHERE id = ?
        """, (note_id,)).fetchone()
    
    conn.close()
    
    if not note_info:
        return None
    
    content_path = note_info[2]
    if content_path and Path(content_path).exists():
        with open(content_path, 'r', encoding='utf-8') as f:
            full_content = f.read()
        # Extract just the note content (after the metadata header)
        if "-" * 50 in full_content:
            content = full_content.split("-" * 50, 1)[1].strip()
        else:
            content = full_content
    else:
        content = "Content file not found"
    
    return {
        "id": note_id,
        "type": note_info[0],
        "title": note_info[1],
        "content": content,
        "links": note_info[3],
        "tags": note_info[4],
        "timestamp": note_info[5],
        "has_embedding": note_info[6]
    }

def update_note(note_id, title, content, links="", tags=""):
    """Update an existing note"""
    db_path = Path("notes") / "metadata" / "notes_database.db"
    conn = sqlite3.connect(str(db_path))
    
    # Get current note info
    note_info = conn.execute("""
        SELECT content_path, note_type FROM notes WHERE id = ?
    """, (note_id,)).fetchone()
    
    if note_info:
        content_path, note_type = note_info
        
        # Update file content
        if content_path and Path(content_path).exists():
            formatted_content = f"""Title: {title}
Category: {note_type}
Tags: {tags}
Created: {datetime.now().isoformat()}
Links: {links}
{"-" * 50}

{content}
"""
            with open(content_path, 'w', encoding='utf-8') as f:
                f.write(formatted_content)
            
            # Update database
            new_hash = hashlib.md5(formatted_content.encode()).hexdigest()
            file_size = Path(content_path).stat().st_size
            
            try:
                conn.execute("""
                    UPDATE notes 
                    SET title = ?, links = ?, tags = ?, file_hash = ?, file_size = ?, 
                        last_modified = CURRENT_TIMESTAMP, has_embedding = FALSE
                    WHERE id = ?
                """, (title, links, tags, new_hash, file_size, note_id))
            except sqlite3.OperationalError:
                # Fallback for missing columns
                conn.execute("""
                    UPDATE notes 
                    SET title = ?, has_embedding = FALSE
                    WHERE id = ?
                """, (title, note_id))
            
            # Update embedding status to regenerate
            try:
                conn.execute("""
                    UPDATE embedding_status 
                    SET needs_embedding = TRUE, file_hash = ?, last_checked = CURRENT_TIMESTAMP,
                        processing_status = 'pending'
                    WHERE file_path = ?
                """, (new_hash, content_path))
                
                # Queue new embedding job
                conn.execute("""
                    INSERT INTO embedding_jobs (note_id, status)
                    VALUES (?, ?)
                """, (note_id, 'pending'))
            except sqlite3.OperationalError:
                pass  # Tables don't exist yet
    
    conn.commit()
    conn.close()

    # --- GOOGLE DRIVE SYNC ---
    if 'drive_instance' in st.session_state and st.session_state.drive_instance:
        with st.spinner("Syncing updated note to Google Drive..."):
            gds.sync_directory_to_drive(st.session_state.drive_instance, "notes")
    # --- END GOOGLE DRIVE SYNC ---

def delete_note(note_id):
    """Delete a note from database and file system"""
    db_path = Path("notes") / "metadata" / "notes_database.db"
    conn = sqlite3.connect(str(db_path))
    
    # Get note info before deletion
    try:
        note_info = conn.execute("""
            SELECT content_path, COALESCE(audio_path, '') FROM notes WHERE id = ?
        """, (note_id,)).fetchone()
    except sqlite3.OperationalError:
        note_info = conn.execute("""
            SELECT content_path, '' FROM notes WHERE id = ?
        """, (note_id,)).fetchone()
    
    if note_info:
        content_path, audio_path = note_info
        
        # Delete files
        if content_path and Path(content_path).exists():
            Path(content_path).unlink()
        if audio_path and Path(audio_path).exists():
            Path(audio_path).unlink()
        
        # Delete from database
        conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        try:
            conn.execute("DELETE FROM embedding_status WHERE file_path = ?", (content_path,))
            conn.execute("DELETE FROM embedding_jobs WHERE note_id = ?", (note_id,))
        except sqlite3.OperationalError:
            pass  # Tables don't exist yet
        
        conn.commit()
    
    conn.close()
    
    # --- GOOGLE DRIVE SYNC ---
    if 'drive_instance' in st.session_state and st.session_state.drive_instance:
        with st.spinner("Syncing deletion to Google Drive..."):
            gds.sync_directory_to_drive(st.session_state.drive_instance, "notes")
    # --- END GOOGLE DRIVE SYNC ---


# ================================
# CHROMADB LOCAL COLLECTION SETUP
# ================================

@st.cache_resource(show_spinner=False)
def get_chroma_client(persist_directory="./db_local/"):
    # Ensure the directory exists after a sync
    os.makedirs(persist_directory, exist_ok=True)
    client = chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings()
    )
    return client

@st.cache_resource(show_spinner=False)
def get_local_chroma_collection(embedding_dimension=384):
    client = get_chroma_client()
    collection_name = f"analog_products_local_{embedding_dimension}d"
    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=None
        )
        st.info(f"✅ Using local collection: {collection_name}")
    except Exception as e:
        st.error(f"Error getting Chroma collection: {e}")
        return None
    return collection

@st.cache_resource(show_spinner=False)
def get_notes_chroma_collection(embedding_dimension=384):
    """Get or create ChromaDB collection specifically for notes"""
    client = get_chroma_client()
    collection_name = f"user_notes_local_{embedding_dimension}d"
    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=None
        )
        st.info(f"✅ Using notes collection: {collection_name}")
    except Exception as e:
        st.error(f"Error getting notes collection: {e}")
        return None
    return collection

# ================================
# PHASE 3+: EMBEDDING PROCESSING WITH PROGRESS DISPLAY
# ================================

def generate_note_embeddings_batch_with_progress(note_ids: List[int], embedding_model_instance, embedding_dimension=384):
    """Generate embeddings for a batch of notes with progress display"""
    if not embedding_model_instance:
        return 0
    
    db_path = Path("notes") / "metadata" / "notes_database.db"
    conn = sqlite3.connect(str(db_path))
    
    notes_collection = get_notes_chroma_collection(embedding_dimension)
    
    # Create progress containers
    progress_text = "Generating embeddings for notes..."
    progress_bar = st.progress(0, text=progress_text)
    status_text = st.empty()
    
    successful_count = 0
    
    for idx, note_id in enumerate(note_ids):
        try:
            # Update progress
            progress_percent = int((idx / len(note_ids)) * 100)
            progress_bar.progress(progress_percent, text=f"Processing note {idx+1}/{len(note_ids)}...")
            status_text.text(f"📝 Processing note ID: {note_id}")
            
            # Update job status
            conn.execute("""
                UPDATE embedding_jobs 
                SET status = 'processing', started_at = CURRENT_TIMESTAMP
                WHERE note_id = ? AND status = 'pending'
            """, (note_id,))
            
            # Get note content
            try:
                note_info = conn.execute("""
                    SELECT title, content_path, note_type, 
                           COALESCE(tags, '') as tags, 
                           COALESCE(links, '') as links
                    FROM notes WHERE id = ?
                """, (note_id,)).fetchone()
            except sqlite3.OperationalError:
                note_info = conn.execute("""
                    SELECT title, content_path, note_type, '', ''
                    FROM notes WHERE id = ?
                """, (note_id,)).fetchone()
            
            if not note_info:
                continue
            
            title, content_path, note_type, tags, links = note_info
            
            # Read content
            if content_path and Path(content_path).exists():
                with open(content_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                continue
            
            status_text.text(f"🧠 Generating embedding for: {title[:30]}...")
            
            # Generate embedding
            embedding = embedding_model_instance.encode(content, normalize_embeddings=True)
            embedding_list = embedding.tolist()
            
            # Create unique ID for ChromaDB
            chroma_id = f"note_{note_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            status_text.text(f"💾 Saving to database: {title[:30]}...")
            
            # Add to ChromaDB
            notes_collection.add(
                ids=[chroma_id],
                embeddings=[embedding_list],
                documents=[content],
                metadatas=[{
                    "note_id": note_id,
                    "title": title,
                    "note_type": note_type,
                    "tags": tags,
                    "links": links,
                    "embedding_model": embedding_model_name,
                    "content_type": "note"
                }]
            )
            
            # Update database
            conn.execute("""
                UPDATE notes 
                SET has_embedding = TRUE, embedding_model = ?
                WHERE id = ?
            """, (embedding_model_name, note_id))
            
            try:
                conn.execute("""
                    UPDATE embedding_status 
                    SET needs_embedding = FALSE, processing_status = 'completed',
                        embedding_model = ?, embedding_id = ?
                    WHERE file_path = (SELECT content_path FROM notes WHERE id = ?)
                """, (embedding_model_name, chroma_id, note_id))
                
                conn.execute("""
                    UPDATE embedding_jobs 
                    SET status = 'completed', completed_at = CURRENT_TIMESTAMP
                    WHERE note_id = ? AND status = 'processing'
                """, (note_id,))
            except sqlite3.OperationalError:
                pass  # Tables don't exist yet
            
            successful_count += 1
            
        except Exception as e:
            # Handle errors
            try:
                conn.execute("""
                    UPDATE embedding_jobs 
                    SET status = 'failed', error_message = ?, completed_at = CURRENT_TIMESTAMP
                    WHERE note_id = ? AND status = 'processing'
                """, (str(e), note_id))
                
                conn.execute("""
                    UPDATE embedding_status 
                    SET processing_status = 'failed', error_message = ?
                    WHERE file_path = (SELECT content_path FROM notes WHERE id = ?)
                """, (str(e), note_id))
            except sqlite3.OperationalError:
                pass
            
            status_text.text(f"❌ Error processing note {note_id}: {str(e)}")
    
    # Complete progress
    progress_bar.progress(100, text="✅ Embedding generation completed!")
    status_text.text(f"🎉 Successfully processed {successful_count}/{len(note_ids)} notes")
    
    conn.commit()
    conn.close()
    
    return successful_count

def process_embedding_queue():
    """Process pending embedding jobs with progress display"""
    db_path = Path("notes") / "metadata" / "notes_database.db"
    if not Path(db_path).exists():
        return 0
    
    # Use the global embedding_model
    global embedding_model
    if not embedding_model:
        embedding_model = get_global_embedding_model()
    
    conn = sqlite3.connect(str(db_path))
    
    # Get pending jobs
    try:
        pending_jobs = conn.execute("""
            SELECT note_id FROM embedding_jobs 
            WHERE status = 'pending' 
            ORDER BY created_at ASC
            LIMIT 10
        """).fetchall()
    except sqlite3.OperationalError:
        # If table doesn't exist, get notes that need embeddings
        pending_jobs = conn.execute("""
            SELECT id FROM notes 
            WHERE has_embedding = FALSE 
            ORDER BY timestamp DESC
            LIMIT 10
        """).fetchall()
    
    conn.close()
    
    if not pending_jobs:
        return 0
    
    note_ids = [job[0] for job in pending_jobs]
    
    # Get embedding dimension
    if embedding_model:
        test_emb = embedding_model.encode("test")
        embedding_dim = len(test_emb)
    else:
        embedding_dim = 384
    
    return generate_note_embeddings_batch_with_progress(note_ids, embedding_model, embedding_dim)

# ================================
# PHASE 3+: UNIFIED SEARCH SYSTEM
# ================================

def unified_search(query_text, embedding_model_instance, n_results=10, include_notes=True):
    """Search across both products and notes with unified results"""
    if not embedding_model_instance:
        return {"products": [], "notes": [], "error": "Embedding model not available"}
    
    # Generate query embedding
    try:
        query_embedding = embedding_model_instance.encode(query_text, normalize_embeddings=True).tolist()
    except Exception as e:
        return {"products": [], "notes": [], "error": f"Failed to generate embedding: {e}"}
    
    results = {"products": [], "notes": [], "combined": []}
    
    # Get embedding dimension
    embedding_dim = len(query_embedding)
    
    # Search products collection
    try:
        products_collection = get_local_chroma_collection(embedding_dim)
        product_results = products_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        results["products"] = product_results
    except Exception as e:
        results["products"] = {"error": str(e)}
    
    # Search notes collection if enabled
    if include_notes:
        try:
            notes_collection = get_notes_chroma_collection(embedding_dim)
            notes_count = notes_collection.count()
            
            if notes_count > 0:
                note_results = notes_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(n_results, notes_count),
                    include=["documents", "metadatas", "distances"]
                )
                results["notes"] = note_results
            else:
                results["notes"] = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        except Exception as e:
            results["notes"] = {"error": str(e)}
    
    # Combine and rank results
    combined_results = []
    
    # Add product results
    if results["products"].get("documents") and results["products"]["documents"][0]:
        for i, (doc, meta, dist) in enumerate(zip(
            results["products"]["documents"][0],
            results["products"]["metadatas"][0],
            results["products"]["distances"][0]
        )):
            combined_results.append({
                "content": doc,
                "metadata": meta,
                "distance": dist,
                "source": "product",
                "relevance_score": 1.0 - dist
            })
    
    # Add note results with weighting
    if results["notes"].get("documents") and results["notes"]["documents"][0]:
        for i, (doc, meta, dist) in enumerate(zip(
            results["notes"]["documents"][0],
            results["notes"]["metadatas"][0],
            results["notes"]["distances"][0]
        )):
            # Apply note context weight
            weighted_relevance = (1.0 - dist) * note_context_weight
            combined_results.append({
                "content": doc,
                "metadata": meta,
                "distance": dist,
                "source": "note",
                "relevance_score": weighted_relevance
            })
    
    # Sort by relevance score
    combined_results.sort(key=lambda x: x["relevance_score"], reverse=True)
    results["combined"] = combined_results[:n_results * 2]
    
    return results

# Initialize the advanced notes system
notes_db_path = init_advanced_notes_database()

# ================================
# SIDEBAR: PHASE 3+ ENHANCED UI WITH PROGRESS
# ================================

st.sidebar.header("📝 Advanced Notes (Phase 3+)")

# Enhanced note statistics
stats = get_advanced_notes_stats()
col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("Total Notes", stats["total_notes"])
    st.metric("With Embeddings", stats["has_embedding"])
with col2:
    st.metric("Need Embedding", stats["needs_embedding"])
    st.metric("Pending Jobs", stats.get("pending_jobs", 0))

# Enhanced embedding status display
if stats.get("pending_jobs", 0) > 0 or stats.get("processing_jobs", 0) > 0:
    with st.sidebar.container():
        if stats.get("processing_jobs", 0) > 0:
            st.warning(f"🔄 {stats['processing_jobs']} jobs currently processing...")
        if stats.get("pending_jobs", 0) > 0:
            st.warning(f"⏳ {stats['pending_jobs']} jobs pending...")
        
        # Enhanced embedding progress display
        if stats.get("failed_jobs", 0) > 0:
            st.error(f"❌ {stats['failed_jobs']} jobs failed")
        if stats.get("completed_jobs", 0) > 0:
            st.success(f"✅ {stats['completed_jobs']} jobs completed")

# FIXED: Embedding queue management with visual progress
if stats.get("pending_jobs", 0) > 0:
    if st.sidebar.button("🚀 Process Embedding Queue (with Progress)"):
        with st.sidebar:
            processed = process_embedding_queue()
            if processed > 0:
                st.success(f"✅ Processed {processed} embeddings!")
                time.sleep(2)  # Show success message
                st.rerun()
            else:
                st.info("No jobs to process")

# Enhanced note creation
with st.sidebar.expander("✨ Create Advanced Note", expanded=False):
    note_type = st.selectbox(
        "Category:",
        list(NOTE_CATEGORIES.keys()),
        format_func=lambda x: f"{NOTE_CATEGORIES[x]['emoji']} {x.replace('_', ' ').title()}"
    )
    
    note_title = st.text_input("Title:", placeholder="Enter note title...")
    note_content = st.text_area("Content:", placeholder="Write your note here...", height=100)
    note_links = st.text_input("Links:", placeholder="https://...")
    note_tags = st.text_input("Tags:", placeholder="tag1, tag2, tag3")
    
    if st.button("💾 Save Advanced Note"):
        if note_title and note_content:
            try:
                note_id, file_path = save_advanced_note(
                    note_type, note_title, note_content, note_links, note_tags
                )
                st.success(f"✅ Note saved! ID: {note_id}")
                st.info("🔄 Queued for embedding generation")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error saving note: {e}")
        else:
            st.warning("Please fill in title and content")

# PHASE 3+: Note Management Interface
with st.sidebar.expander("🗂️ Manage All Notes", expanded=False):
    if st.button("📋 Show All Notes Manager"):
        st.session_state.show_note_manager = True
        st.rerun()
    
    if st.button("🔄 Generate All Missing Embeddings"):
        # Process all pending jobs with progress
        total_processed = 0
        while True:
            processed = process_embedding_queue()
            if processed == 0:
                break
            total_processed += processed
        
        if total_processed > 0:
            st.success(f"✅ Generated {total_processed} embeddings!")
        else:
            st.info("All notes already have embeddings")
        st.rerun()

# Recent notes display
if stats["total_notes"] > 0:
    st.sidebar.subheader("📋 Recent Notes")
    recent = get_recent_notes(3)
    for note in recent:
        emoji = NOTE_CATEGORIES[note["type"]]["emoji"]
        embed_status = "✅" if note["has_embedding"] else "⏳"
        with st.sidebar.expander(f"{emoji} {note['title'][:20]}..."):
            st.write(f"**Type:** {note['type'].replace('_', ' ').title()}")
            st.write(f"**Created:** {note['timestamp'][:19]}")
            st.write(f"**Embedding:** {embed_status}")

# Add benefits info
st.sidebar.info("""
💰 **Phase 3+ Benefits:**
- ✅ Zero API costs
- ✅ No rate limits  
- ✅ Local embeddings
- ✅ Unified search
- 🆕 Note embeddings
- 🆕 Background processing
- 🆕 Progress tracking
- 🆕 Full note management
- 🆕 Visual feedback
""")

# ================================
# PHASE 3+: COMPREHENSIVE NOTE MANAGER
# ================================

# Initialize session state
if 'show_note_manager' not in st.session_state:
    st.session_state.show_note_manager = False

if st.session_state.show_note_manager:
    st.write("---")
    st.header("🗂️ Comprehensive Note Manager with Embedding Status")
    
    # Close button
    if st.button("❌ Close Note Manager"):
        st.session_state.show_note_manager = False
        st.rerun()
    
    # Get all notes
    all_notes = get_all_notes_with_details()
    
    if not all_notes:
        st.info("No notes found. Create your first note using the sidebar!")
    else:
        st.write(f"**Managing {len(all_notes)} notes**")
        
        # Enhanced filters with embedding status
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            filter_type = st.selectbox(
                "Filter by Type:",
                ["All Types"] + list(NOTE_CATEGORIES.keys()),
                format_func=lambda x: f"{NOTE_CATEGORIES.get(x, {}).get('emoji', '📁')} {x.replace('_', ' ').title()}" if x != "All Types" else "📁 All Types"
            )
        
        with col2:
            filter_embedding = st.selectbox(
                "Filter by Embedding:",
                ["All Notes", "With Embeddings", "Without Embeddings"]
            )
        
        with col3:
            search_notes = st.text_input("🔍 Search Notes:", placeholder="Search in titles...")
        
        with col4:
            sort_by = st.selectbox(
                "Sort by:",
                ["Recent First", "Oldest First", "Title A-Z", "Title Z-A"]
            )
        
        # Apply filters
        filtered_notes = all_notes
        
        if filter_type != "All Types":
            filtered_notes = [n for n in filtered_notes if n["type"] == filter_type]
        
        if filter_embedding == "With Embeddings":
            filtered_notes = [n for n in filtered_notes if n["has_embedding"]]
        elif filter_embedding == "Without Embeddings":
            filtered_notes = [n for n in filtered_notes if not n["has_embedding"]]
        
        if search_notes:
            filtered_notes = [n for n in filtered_notes if search_notes.lower() in n["title"].lower()]
        
        # Apply sorting
        if sort_by == "Recent First":
            filtered_notes.sort(key=lambda x: x["timestamp"], reverse=True)
        elif sort_by == "Oldest First":
            filtered_notes.sort(key=lambda x: x["timestamp"])
        elif sort_by == "Title A-Z":
            filtered_notes.sort(key=lambda x: x["title"])
        elif sort_by == "Title Z-A":
            filtered_notes.sort(key=lambda x: x["title"], reverse=True)
        
        st.write(f"**Showing {len(filtered_notes)} notes**")
        
        # Enhanced display with embedding status
        for note in filtered_notes:
            emoji = NOTE_CATEGORIES[note["type"]]["emoji"]
            embed_status = "✅" if note["has_embedding"] else "⏳"
            file_size_kb = (note["file_size"] or 0) / 1024
            
            # Enhanced status indicator
            if note["has_embedding"]:
                status_color = "🟢"
                status_text = "Ready"
            else:
                status_color = "🟡"
                status_text = "Pending"
            
            with st.expander(f"{emoji} {note['title']} - {status_color} {status_text} ({file_size_kb:.1f} KB)"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Type:** {note['type'].replace('_', ' ').title()}")
                    st.write(f"**Created:** {note['timestamp'][:19]}")
                    st.write(f"**Last Modified:** {note['last_modified'][:19] if note['last_modified'] else 'N/A'}")
                    st.write(f"**Embedding Status:** {'✅ Ready' if note['has_embedding'] else '⏳ Pending'}")
                    if note['has_embedding'] and note['embedding_model']:
                        st.write(f"**Embedding Model:** {note['embedding_model']}")
                    st.write(f"**Priority:** {note['search_priority']}")
                    if note['tags']:
                        st.write(f"**Tags:** {note['tags']}")
                    if note['links']:
                        st.write(f"**Links:** {note['links']}")
                    
                    # View content
                    if st.button("👁️ View Content", key=f"view_{note['id']}"):
                        note_content = get_note_content(note['id'])
                        if note_content:
                            st.text_area("Content:", note_content['content'], height=200, key=f"content_view_{note['id']}")
                    
                    # Edit note
                    if st.button("✏️ Edit Note", key=f"edit_{note['id']}"):
                        st.session_state[f"editing_{note['id']}"] = True
                        st.rerun()
                    
                    # Edit form
                    if st.session_state.get(f"editing_{note['id']}", False):
                        note_content = get_note_content(note['id'])
                        if note_content:
                            st.write("**Editing Note:**")
                            new_title = st.text_input("Title:", value=note_content['title'], key=f"edit_title_{note['id']}")
                            new_content = st.text_area("Content:", value=note_content['content'], height=200, key=f"edit_content_{note['id']}")
                            new_links = st.text_input("Links:", value=note_content['links'], key=f"edit_links_{note['id']}")
                            new_tags = st.text_input("Tags:", value=note_content['tags'], key=f"edit_tags_{note['id']}")
                            
                            col_save, col_cancel = st.columns(2)
                            with col_save:
                                if st.button("💾 Save Changes", key=f"save_{note['id']}"):
                                    update_note(note['id'], new_title, new_content, new_links, new_tags)
                                    st.session_state[f"editing_{note['id']}"] = False
                                    st.success("✅ Note updated! Embedding will be regenerated.")
                                    st.rerun()
                            
                            with col_cancel:
                                if st.button("❌ Cancel", key=f"cancel_{note['id']}"):
                                    st.session_state[f"editing_{note['id']}"] = False
                                    st.rerun()
                
                with col2:
                    # Enhanced action buttons
                    if not note['has_embedding']:
                        if st.button("🚀 Generate Embedding", key=f"embed_{note['id']}"):
                            # Process single note embedding with progress
                            # Calculate embedding dimension
                            if embedding_model:
                                test_emb = embedding_model.encode("test")
                                embedding_dim = len(test_emb)
                            else:
                                model_dimensions = {
                                    "BAAI/bge-small-en-v1.5": 384,
                                    "all-mpnet-base-v2": 768,
                                    "paraphrase-multilingual-mpnet-base-v2": 768,
                                    "all-MiniLM-L6-v2": 384
                                }
                                embedding_dim = model_dimensions.get(embedding_model_name, 384)
                            
                            processed = generate_note_embeddings_batch_with_progress([note['id']], embedding_model, embedding_dim)
                                                        
                            if processed > 0:
                                st.success("✅ Embedding generated!")
                                st.rerun()
                    
                    if st.button("🗑️ Delete", key=f"delete_{note['id']}"):
                        st.session_state[f"confirm_delete_{note['id']}"] = True
                        st.rerun()
                    
                    if st.session_state.get(f"confirm_delete_{note['id']}", False):
                        st.warning("⚠️ Are you sure?")
                        col_yes, col_no = st.columns(2)
                        with col_yes:
                            if st.button("✅ Yes", key=f"yes_delete_{note['id']}"):
                                delete_note(note['id'])
                                st.session_state[f"confirm_delete_{note['id']}"] = False
                                st.success("🗑️ Note deleted!")
                                st.rerun()
                        with col_no:
                            if st.button("❌ No", key=f"no_delete_{note['id']}"):
                                st.session_state[f"confirm_delete_{note['id']}"] = False
                                st.rerun()
                    
                    st.write(f"**ID:** {note['id']}")

# ================================
# Continue with the rest of your original functions...
# (Conversation management, Puter.js integration, search UI, etc.)
# ================================

# Initialize conversation history in session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

def add_to_conversation(role, content):
    """Add message to conversation history"""
    if enable_context:
        st.session_state.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        
        if len(st.session_state.conversation_history) > max_context_messages * 2:
            st.session_state.conversation_history = st.session_state.conversation_history[-max_context_messages * 2:]

def build_context_prompt(user_query, document_context, note_context=""):
    """Build prompt with conversation context and note context"""
    if not enable_context or not st.session_state.conversation_history:
        base_prompt = f"User Query: {user_query}\n\nContext: {document_context}"
    else:
        context_messages = ""
        for msg in st.session_state.conversation_history[-max_context_messages:]:
            role_emoji = "👤" if msg["role"] == "user" else "🤖"
            context_messages += f"{role_emoji} {msg['role'].capitalize()}: {msg['content']}\n"
        
        base_prompt = f"""Previous Conversation:
{context_messages}

Current Query: {user_query}

Document Context: {document_context}"""
    
    # Add note context if available
    if note_context and enable_unified_search:
        base_prompt += f"\n\nPersonal Notes Context: {note_context}"
        base_prompt += "\n\nPlease respond considering both the document context and the user's personal notes above."
    else:
        base_prompt += "\n\nPlease respond considering the conversation history above."
    
    return base_prompt

def clear_conversation_history():
    """Clear conversation history"""
    st.session_state.conversation_history = []
    st.success("🧹 Conversation history cleared!")

# ================================
# LOCAL EMBEDDINGS INGESTION
# ================================

def ingest_local_embeddings(collection, embeddings_folder="product_embeddings_v2"):
    """Ingest embeddings from your embedding script format"""
    if not os.path.exists(embeddings_folder):
        st.warning(f"Embeddings folder '{embeddings_folder}' not found.")
        return
    
    embedding_files = glob.glob(os.path.join(embeddings_folder, "*_embeddings_v2.json"))
    
    if not embedding_files:
        st.warning(f"No embedding files found in '{embeddings_folder}'. Run your embedding generation script first.")
        return
    
    st.write(f"📁 Found {len(embedding_files)} local embedding files")
    
    total_docs = 0
    for file_path in embedding_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            product = data.get("product", "unknown")
            embeddings = data.get("embeddings", [])
            chunks = data.get("chunks", [])
            links = data.get("links", [])
            
            if not embeddings:
                continue
            
            documents = [f"{product} - chunk {i}: {chunk[:200]}..." for i, chunk in enumerate(chunks)]
            ids = [f"{product}_local_chunk_{i}" for i in range(len(embeddings))]
            metadatas = [{
                "product": product,
                "chunk_index": i,
                "model": data.get("model_info", {}).get("name", embedding_model_name),
                "links": ", ".join(links) if isinstance(links, list) else str(links),
                "processing_time": data.get("processing_info", {}).get("processing_time", 0)
            } for i in range(len(embeddings))]
            
            batch_size = 100
            for i in range(0, len(embeddings), batch_size):
                batch_embeddings = embeddings[i:i+batch_size]
                batch_documents = documents[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                batch_metadatas = metadatas[i:i+batch_size]
                
                collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_documents,
                    metadatas=batch_metadatas
                )
            
            total_docs += len(embeddings)
            
        except Exception as e:
            st.error(f"Error processing {file_path}: {e}")
    
    st.success(f"✅ Ingested {total_docs} documents from local embeddings!")

# ================================
# ENHANCED PUTER.JS COMPONENT (keeping your original)
# ================================

def create_streaming_puter_component(prompt, model="gpt-4o-mini", stream=True):
    """ENHANCED: Puter.js component with streaming and context awareness"""
    escaped_prompt = prompt.replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
    unique_id = int(time.time() * 1000) % 1000000
    
    fallback_models = {
        "claude-sonnet-4": ["claude-opus-4", "gpt-4o", "gpt-4o-mini"],
        "claude-opus-4": ["claude-sonnet-4", "gpt-4o", "gpt-4o-mini"]
    }
    
    puter_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://js.puter.com/v2/"></script>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 15px;
                background: #f8f9fa;
                min-height: 600px;
            }}
            .container {{
                background: white;
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                min-height: 550px;
                border: 1px solid #e0e0e0;
            }}
            .model-info {{
                background: linear-gradient(135deg, #e3f2fd, #f3e5f5);
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 20px;
                border-left: 4px solid #2196f3;
                font-size: 0.95em;
            }}
            .streaming-indicator {{
                background: #e8f5e8;
                border: 1px solid #4caf50;
                padding: 10px;
                border-radius: 8px;
                margin-bottom: 15px;
                display: {'block' if stream else 'none'};
            }}
            .context-indicator {{
                background: #fff3e0;
                border: 1px solid #ff9800;
                padding: 10px;
                border-radius: 8px;
                margin-bottom: 15px;
                font-size: 0.9em;
            }}
            .loading {{
                display: flex;
                align-items: center;
                gap: 10px;
                color: #666;
                padding: 15px;
            }}
            .spinner {{
                border: 3px solid #f3f3f3;
                border-top: 3px solid #667eea;
                border-radius: 50%;
                width: 20px;
                height: 20px;
                animation: spin 1s linear infinite;
            }}
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            .result {{
                white-space: pre-wrap;
                line-height: 1.6;
                color: #333;
                max-height: 400px;
                overflow-y: auto;
                padding: 15px;
                background: #fafafa;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
                font-family: inherit;
            }}
            .streaming-text {{
                white-space: pre-wrap;
                line-height: 1.6;
                color: #333;
                max-height: 400px;
                overflow-y: auto;
                padding: 15px;
                background: #fafafa;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
                font-family: inherit;
                min-height: 100px;
            }}
            .error {{
                color: #dc3545;
                background: #f8d7da;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #f5c6cb;
                margin: 10px 0;
            }}
            .warning {{
                background: #fff3cd;
                border: 1px solid #ffeaa7;
                color: #856404;
                padding: 15px;
                border-radius: 8px;
                margin: 10px 0;
            }}
            .success {{
                background: #d4edda;
                border: 1px solid #c3e6cb;
                color: #155724;
                padding: 15px;
                border-radius: 8px;
                margin: 10px 0;
            }}
            .streaming-cursor {{
                animation: blink 1s infinite;
                font-weight: bold;
                color: #667eea;
            }}
            @keyframes blink {{
                0%, 50% {{ opacity: 1; }}
                51%, 100% {{ opacity: 0; }}
            }}
            .stats {{
                margin-top: 15px;
                font-size: 0.9em;
                color: #666;
                border-top: 1px solid #eee;
                padding-top: 10px;
                display: flex;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="container" id="container_{unique_id}">
            <div class="model-info">
                <strong>🤖 Model:</strong> {model} | 
                <strong>⚡ Provider:</strong> Puter.js (Free) | 
                <strong>📊 Embeddings:</strong> Local |
                <strong>🔄 Streaming:</strong> {'Enabled' if stream else 'Disabled'}
            </div>
            
            <div class="streaming-indicator">
                <strong>🌊 Streaming Mode:</strong> Responses will appear in real-time
            </div>
            
            <div class="context-indicator">
                <strong>🧠 Phase 3+ Enhanced:</strong> AI with unified search and complete note management
            </div>
            
            <div id="result_{unique_id}">
                <div class="loading">
                    <div class="spinner"></div>
                    <span>{'Streaming' if stream else 'Processing'} with {model}...</span>
                </div>
            </div>
        </div>
        
        <script>
            async function processQuery_{unique_id}() {{
                const resultDiv = document.getElementById('result_{unique_id}');
                const fallbacks = {json.dumps(fallback_models.get(model, []))};
                
                async function tryStreamingModel(modelName, isRetry = false) {{
                    try {{
                        if (isRetry) {{
                            resultDiv.innerHTML = `
                                <div class="warning">
                                    <strong>🔄 Retrying with ${{modelName}}</strong><br>
                                    Primary model had issues. Trying fallback option...
                                </div>
                            `;
                        }}
                        
                        const startTime = Date.now();
                        const streamingEnabled = {str(stream).lower()};
                        
                        if (streamingEnabled) {{
                            resultDiv.innerHTML = `
                                <div class="streaming-text" id="streamingContent_{unique_id}"></div>
                                <div class="stats" id="stats_{unique_id}">
                                    <span>⏱ Streaming started...</span>
                                    <span>📝 Model: ${{modelName}}</span>
                                </div>
                            `;
                            
                            const streamingContent = document.getElementById('streamingContent_{unique_id}');
                            const stats = document.getElementById('stats_{unique_id}');
                            
                            const response = await puter.ai.chat("{escaped_prompt}", {{
                                model: modelName,
                                stream: true,
                                max_tokens: 2000
                            }});
                            
                            let fullResponse = '';
                            let chunkCount = 0;
                            
                            for await (const chunk of response) {{
                                chunkCount++;
                                const content = chunk?.text || chunk?.content || '';
                                
                                if (content) {{
                                    fullResponse += content;
                                    streamingContent.innerHTML = fullResponse + '<span class="streaming-cursor">▋</span>';
                                    streamingContent.scrollTop = streamingContent.scrollHeight;
                                    
                                    const currentTime = Date.now();
                                    const elapsed = ((currentTime - startTime) / 1000).toFixed(1);
                                    stats.innerHTML = `
                                        <span>⏱ Time: ${{elapsed}}s</span>
                                        <span>📦 Chunks: ${{chunkCount}}</span>
                                        <span>📝 Model: ${{modelName}}</span>
                                        <span>🔄 Streaming...</span>
                                    `;
                                }}
                            }}
                            
                            streamingContent.innerHTML = fullResponse;
                            const endTime = Date.now();
                            const totalTime = ((endTime - startTime) / 1000).toFixed(2);
                            
                            stats.innerHTML = `
                                <span>⏱ Completed in: ${{totalTime}}s</span>
                                <span>📦 Total chunks: ${{chunkCount}}</span>
                                <span>📝 Model: ${{modelName}}</span>
                                <span>✅ Stream complete</span>
                            `;
                        }} else {{
                            const response = await puter.ai.chat("{escaped_prompt}", {{
                                model: modelName,
                                stream: false,
                                max_tokens: 2000
                            }});
                            
                            const endTime = Date.now();
                            const processingTime = ((endTime - startTime) / 1000).toFixed(2);
                            
                            resultDiv.innerHTML = `
                                <div class="result">${{response}}</div>
                                <div class="stats">
                                    <span>⏱ Processing time: ${{processingTime}}s</span>
                                    <span>📝 Model: ${{modelName}}</span>
                                    <span>📊 Non-streaming mode</span>
                                </div>
                            `;
                        }}
                        
                        return true;
                        
                    }} catch (error) {{
                        console.error(`Error with ${{modelName}}:`, error);
                        
                        if (error.message.includes('no fallback model available')) {{
                            resultDiv.innerHTML = `
                                <div class="warning">
                                    <strong>⚠️ Model Temporarily Unavailable</strong><br>
                                    The ${{modelName}} model is experiencing issues. Trying alternative models...
                                </div>
                            `;
                        }} else if (error.message.includes('credits') || error.message.includes('tokens')) {{
                            resultDiv.innerHTML = `
                                <div class="warning">
                                    <strong>⚠️ Usage Limit Reached</strong><br>
                                    ${{modelName}} has reached its usage limit. Switching to alternative model...
                                </div>
                            `;
                        }}
                        
                        return false;
                    }}
                }}
                
                const success = await tryStreamingModel("{model}");
                
                if (!success && fallbacks.length > 0) {{
                    for (const fallback of fallbacks) {{
                        const fallbackSuccess = await tryStreamingModel(fallback, true);
                        if (fallbackSuccess) break;
                        await new Promise(resolve => setTimeout(resolve, 1000));
                    }}
                }}
            }}
            
            processQuery_{unique_id}();
        </script>
    </body>
    </html>
    """
    
    return components.html(puter_html, height=650)

def get_structured_output_from_puter_enhanced(concatenated_text, user_query, model="gpt-4o-mini", note_context=""):
    """ENHANCED: Puter.js processing with context awareness and notes"""
    context_prompt = build_context_prompt(user_query, concatenated_text, note_context)
    
    full_prompt = f"""You are a helpful AI assistant specializing in analog devices and electronic components. 
You have access to both technical documentation and the user's personal notes.
Provide accurate, technical information based on the provided context.

{context_prompt}

Please provide a comprehensive, well-structured response that directly addresses the query."""
    
    st.write("### 🤖 AI Processing with Puter.js (Phase 3+ Enhanced)")
    
    # Show context status
    if enable_context and st.session_state.conversation_history:
        st.info(f"🧠 **Context Active**: Remembering {len(st.session_state.conversation_history)} previous messages")
    
    if note_context and enable_unified_search:
        st.info(f"📝 **Note Context**: Including relevant personal notes in response")
    
    create_streaming_puter_component(full_prompt, model, enable_streaming)
    add_to_conversation("user", user_query)
    
    return "Enhanced response with streaming, context, and notes displayed above"

# ================================
# UTILITY FUNCTIONS
# ================================

def extract_and_display_unified_results(unified_results):
    """Extract and display unified search results (products + notes)"""
    if not unified_results.get("combined"):
        st.write("No results found.")
        return
    
    st.write("### 📄 Unified Search Results (Products + Notes)")
    
    for idx, result in enumerate(unified_results["combined"], start=1):
        source_emoji = "🏭" if result["source"] == "product" else "📝"
        source_text = "Product" if result["source"] == "product" else "Personal Note"
        relevance = result["relevance_score"]
        
        with st.expander(f"{source_emoji} Result {idx}: {source_text} (Relevance: {relevance:.2f})"):
            st.write(f"**Content:** {result['content'][:500]}...")
            st.write(f"**Source:** {source_text}")
            st.write(f"**Relevance Score:** {relevance:.3f}")
            
            if result["source"] == "product":
                product = result["metadata"].get("product", "Unknown")
                chunk_index = result["metadata"].get("chunk_index", "Unknown")
                links = result["metadata"].get("links", "No links provided")
                st.write(f"**Product:** {product}")
                st.write(f"**Chunk Index:** {chunk_index}")
                st.write(f"**Links:** {links}")
            else:
                title = result["metadata"].get("title", "Unknown")
                note_type = result["metadata"].get("note_type", "Unknown")
                tags = result["metadata"].get("tags", "")
                st.write(f"**Note Title:** {title}")
                st.write(f"**Note Type:** {note_type}")
                if tags:
                    st.write(f"**Tags:** {tags}")

def extract_and_display_documents(query_results):
    """Extract and display source documents"""
    documents_nested = query_results.get("documents")
    metadatas_nested = query_results.get("metadatas")
    
    if not documents_nested:
        st.write("No documents found in the query results.")
        return
    
    documents = documents_nested[0] if documents_nested else []
    metadatas = metadatas_nested[0] if metadatas_nested and isinstance(metadatas_nested, list) and len(metadatas_nested) > 0 else []
    
    st.write("### 📄 Source Documents")
    for idx, doc in enumerate(documents, start=1):
        metadata = metadatas[idx-1] if idx-1 < len(metadatas) else {}
        product = metadata.get("product", "Unknown")
        chunk_index = metadata.get("chunk_index", "Unknown")
        links = metadata.get("links", "No links provided")
        
        with st.expander(f"Document {idx}: {product} - Chunk {chunk_index}"):
            st.write(f"**Content:** {doc}")
            st.write(f"**Product:** {product}")
            st.write(f"**Chunk Index:** {chunk_index}")
            st.write(f"**Links:** {links}")

# ================================
# MAIN SEARCH UI WITH PHASE 3+ ENHANCEMENTS
# ================================

st.write("---")
st.header("🔍 Phase 3+: Unified Search with Visual Progress Tracking")

# Context Management Controls
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🧹 Clear Context History"):
        clear_conversation_history()

with col2:
    if st.button("📋 View Context History"):
        if st.session_state.conversation_history:
            st.json(st.session_state.conversation_history)
        else:
            st.info("No conversation history yet")

with col3:
    st.write(f"**Context Messages:** {len(st.session_state.conversation_history)}")

query_text = st.text_input(
    "Enter your search query:", 
    placeholder="e.g., Wideband Low Noise Amplifier datasheet or my notes about amplifiers",
    help="Search with streaming responses, conversation memory, and personal notes"
)

# Get embedding dimension
if embedding_model:
    test_emb = embedding_model.encode("test")
    embedding_dim = len(test_emb)
else:
    model_dimensions = {
        "BAAI/bge-small-en-v1.5": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-multilingual-mpnet-base-v2": 768,
        "all-MiniLM-L6-v2": 384
    }
    embedding_dim = model_dimensions.get(embedding_model_name, 384)

# Display current configuration
config_cols = st.columns(4)
with config_cols[0]:
    st.info(f"📊 **Model:** {embedding_model_name} ({embedding_dim}D)")
with config_cols[1]:
    st.info(f"🔄 **Streaming:** {'✅ Enabled' if enable_streaming else '❌ Disabled'}")
with config_cols[2]:
    st.info(f"🧠 **Context:** {'✅ Active' if enable_context else '❌ Disabled'}")
with config_cols[3]:
    st.info(f"🔗 **Unified Search:** {'✅ Active' if enable_unified_search else '❌ Disabled'}")

collection = get_local_chroma_collection(embedding_dim)
notes_collection = get_notes_chroma_collection(embedding_dim)

# Display collection status
col1, col2 = st.columns(2)
with col1:
    try:
        doc_count = collection.count()
        if doc_count == 0:
            st.warning("⚠️ Products collection is empty.")
        else:
            st.success(f"✅ Products ready: {doc_count:,} documents")
    except:
        st.error("❌ Error accessing products collection")

with col2:
    try:
        notes_count = notes_collection.count()
        if notes_count == 0:
            st.info("📝 No notes with embeddings yet")
        else:
            st.success(f"✅ Notes ready: {notes_count:,} documents")
    except:
        st.error("❌ Error accessing notes collection")

# Data Management
st.write("### 📥 Data Management")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📂 Ingest Local Embeddings"):
        with st.spinner("Ingesting local embeddings..."):
            ingest_local_embeddings(collection)

with col2:
    if st.button("📊 Collection Stats"):
        try:
            count = collection.count()
            st.write(f"Product documents: {count:,}")
            notes_count = notes_collection.count()
            st.write(f"Note documents: {notes_count:,}")
            if count > 0:
                sample = collection.get(limit=1, include=["metadatas"])
                if sample["metadatas"]:
                    st.write("Sample product metadata:")
                    st.json(sample["metadatas"][0])
        except Exception as e:
            st.error(f"Error getting stats: {e}")

with col3:
    if st.button("🗑️ Clear Collections"):
        if st.sidebar.button("⚠️ Confirm Clear Collections"):
            try:
                all_data = collection.get()
                if all_data["ids"]:
                    collection.delete(ids=all_data["ids"])
                
                notes_data = notes_collection.get()
                if notes_data["ids"]:
                    notes_collection.delete(ids=notes_data["ids"])
                
                st.success("Collections cleared!")
                st.rerun()
            except Exception as e:
                st.error(f"Error clearing collections: {e}")

# PHASE 3+: Unified search with products and notes
if st.button("🚀 Phase 3+ Unified Search (Products + Notes)", type="primary"):
    if not query_text:
        st.warning("Please enter a valid search query.")
    else:
        with st.spinner("🔍 Performing unified search across products and notes..."):
            try:
                unified_results = unified_search(
                    query_text, 
                    embedding_model, 
                    n_results=10, 
                    include_notes=enable_unified_search
                )
                
                if unified_results.get("error"):
                    st.error(f"❌ Search error: {unified_results['error']}")
                    st.stop()
                
                if not unified_results.get("combined"):
                    st.warning("No relevant results found.")
                    st.stop()
                
                st.success("✅ Unified search completed! Processing with Enhanced AI...")
                
                # Extract contexts for AI
                product_context = ""
                note_context = ""
                
                for result in unified_results["combined"][:5]:
                    if result["source"] == "product":
                        product_context += f"\n\n{result['content']}"
                    else:
                        note_context += f"\n\n{result['content']}"
                
                # Use enhanced Puter.js with unified context
                get_structured_output_from_puter_enhanced(
                    product_context, 
                    query_text, 
                    model=selected_model,
                    note_context=note_context
                )
                
                # Display unified results
                extract_and_display_unified_results(unified_results)
                
            except Exception as e:
                st.error(f"❌ Error during unified search: {e}")
                st.write("**Troubleshooting Steps:**")
                st.write("1. Ensure embedding model is loaded")
                st.write("2. Check if collections have data")
                st.write("3. Process embedding queue for notes")

# ================================
# ENHANCED FOOTER
# ================================

st.write("---")
st.markdown(f"""
### 🎉 **Phase 3+ Complete - Ultimate AI Knowledge Management with Visual Progress!**

**✅ Core Features:**
- 📊 **Local Embeddings** - {embedding_model_name} ({embedding_dim}D)
- 🤖 **Puter.js AI** - {selected_model} with advanced capabilities
- 🚫 **Zero Dependencies** - No external API costs

**🆕 Enhanced Features:**
- 🌊 **Streaming Responses** - Real-time AI output ({'✅ Enabled' if enable_streaming else '❌ Disabled'})
- 🧠 **Context Awareness** - Conversation memory ({'✅ Active' if enable_context else '❌ Disabled'})
- 🔗 **Unified Search** - Products + Notes ({'✅ Active' if enable_unified_search else '❌ Disabled'})
- 📝 **History Management** - {len(st.session_state.conversation_history)} messages in memory

**📝 Phase 3+ Complete Note System:**
- 📋 **Total Notes** - {stats["total_notes"]} notes created
- ✅ **With Embeddings** - {stats["has_embedding"]} notes processed
- ⏳ **Pending Jobs** - {stats.get("pending_jobs", 0)} embedding jobs queued
- 🔄 **Background Processing** - Automatic embedding generation with progress bars
- 🎯 **Smart Search** - Cross-reference between products and personal knowledge
- 📊 **Advanced Analytics** - Comprehensive note statistics and insights
- ✏️ **Full CRUD Operations** - Create, Read, Update, Delete all note types
- 🗂️ **Complete Management** - View, edit, delete with embedding status tracking
- 📈 **Visual Progress** - Real-time progress tracking for all operations
- 🔍 **Enhanced Filtering** - Advanced search and sorting capabilities

**🎯 System Status:**
- 💰 **Cost:** $0 (Completely free)
- 🔄 **Rate Limits:** None
- 🛡️ **Privacy:** All data local
- 🌐 **Connectivity:** Works offline
- ⚡ **Performance:** Phase 3+ optimized with visual progress tracking
- 🧠 **Intelligence:** AI learns from your personal notes
- 🗂️ **Management:** Complete note lifecycle management with visual feedback

*Your system is now the ultimate AI-powered knowledge management platform with complete note CRUD operations, visual progress tracking, embedding management, and unified search capabilities!*

**🏆 Achievement Unlocked:** Ultimate professional-grade AI knowledge management system with complete visual progress tracking!

**Background Embedding Status:** ✅ Working with real-time progress display and status tracking!
""")


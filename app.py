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

def queue_synced_notes_for_embedding():
    """
    Scans the database for notes without embeddings and queues them.
    This is essential for processing notes synced from Google Drive.
    """
    db_path = Path("notes") / "metadata" / "notes_database.db"
    if not db_path.exists():
        return 0

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        # Find notes that need an embedding but don't have a pending job
        cursor.execute("""
            SELECT id FROM notes n
            WHERE n.has_embedding = FALSE AND NOT EXISTS (
                SELECT 1 FROM embedding_jobs j
                WHERE j.note_id = n.id AND (j.status = 'pending' OR j.status = 'completed')
            )
        """)
        notes_to_queue = cursor.fetchall()

        if not notes_to_queue:
            return 0

        # Add a pending job for each missing note
        for note_id_tuple in notes_to_queue:
            note_id = note_id_tuple[0]
            cursor.execute("""
                INSERT INTO embedding_jobs (note_id, status)
                VALUES (?, 'pending')
            """, (note_id,))
        
        conn.commit()
        st.info(f"✅ Queued {len(notes_to_queue)} synced notes for embedding generation.")
        return len(notes_to_queue)

    except sqlite3.OperationalError as e:
        st.warning(f"Could not queue synced notes (database might be old): {e}")
        return 0
    finally:
        conn.close()

# Function to run the initial sync from Google Drive
@st.cache_resource(show_spinner="Connecting to Google Drive and syncing data...")
def initial_sync():
    """
    Authenticates with Google Drive, downloads data, and queues synced notes.
    """
    try:
        drive = gds.authenticate_gdrive()
        
        gds.sync_directory_from_drive(drive, "notes")
        gds.sync_directory_from_drive(drive, "product_embeddings_v2")
        
        # Queue synced notes for embedding
        queue_synced_notes_for_embedding()
        
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
        st.rerun() 
    else:
        st.sidebar.error("❌ Google Drive sync failed. App cannot continue.")
        st.stop()
# --- END GOOGLE DRIVE SYNC ---


# Add migration success notice
st.sidebar.success("🎉 Phase 3+: Complete Knowledge Management!")
try:
    st.sidebar.image("logo.png", width=200)
except:
    pass

# Model selection for Puter.js
st.sidebar.header("🤖 AI Model Selection")
selected_model = st.sidebar.selectbox(
    "Choose AI Model:",
    ["gpt-4o-mini", "gpt-4o", "claude-sonnet-4", "claude-opus-4", "o1-mini", "o1", "o3-mini", "o3", "gpt-4.1"],
    index=0, help="Select the AI model for processing queries"
)

# Embedding model selection
st.sidebar.header("📊 Embedding Model")
embedding_model_name = st.sidebar.selectbox(
    "Choose Embedding Model:",
    ["BAAI/bge-small-en-v1.5", "all-mpnet-base-v2", "paraphrase-multilingual-mpnet-base-v2", "all-MiniLM-L6-v2"],
    index=0, help="Select the local embedding model"
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
# LOCAL EMBEDDINGS SETUP
# ================================
@st.cache_resource(show_spinner=True)
def load_embedding_model(model_name="BAAI/bge-small-en-v1.5"):
    try:
        with st.spinner(f"Loading embedding model: {model_name}..."):
            model = SentenceTransformer(model_name)
        st.success(f"✅ Local embedding model '{model_name}' loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading embedding model '{model_name}': {e}")
        return None

@st.cache_resource
def get_global_embedding_model():
    return load_embedding_model(embedding_model_name)

embedding_model = get_global_embedding_model()

# ================================
# PHASE 3+: NOTE MANAGEMENT & DATABASE
# ================================
NOTE_CATEGORIES = {
    "personal": {"emoji": "📝", "needs_embedding": True, "audio": False, "priority": 1},
    "achievement": {"emoji": "🏆", "needs_embedding": True, "audio": False, "priority": 2},
    "mistake": {"emoji": "⚠️", "needs_embedding": True, "audio": False, "priority": 3},
    "youtube_summary": {"emoji": "📺", "needs_embedding": True, "audio": True, "priority": 4},
    "youtube_mindmap": {"emoji": "🧠", "needs_embedding": True, "audio": False, "priority": 5},
    "lecture_notes": {"emoji": "📚", "needs_embedding": True, "audio": True, "priority": 6},
    "lecture_mindmap": {"emoji": "🗺️", "needs_embedding": True, "audio": False, "priority": 7}
}

# <<< START SOLUTION MODIFICATION >>>
# Restored the missing function
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

# <<< END SOLUTION MODIFICATION >>>

def migrate_database_to_phase3():
    db_path = Path("notes") / "metadata" / "notes_database.db"
    if not db_path.exists(): return
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT search_priority FROM notes LIMIT 1")
        conn.close()
        return
    except sqlite3.OperationalError:
        st.info("🔄 Migrating database to Phase 3+ schema...")
    
    new_columns = [
        ("tags", "TEXT DEFAULT ''"), ("last_modified", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
        ("embedding_dimension", "INTEGER DEFAULT 384"), ("search_priority", "INTEGER DEFAULT 1")
    ]
    for col_name, col_def in new_columns:
        try:
            cursor.execute(f"ALTER TABLE notes ADD COLUMN {col_name} {col_def}")
        except sqlite3.OperationalError: pass
    
    for note_type, config in NOTE_CATEGORIES.items():
        cursor.execute("UPDATE notes SET search_priority = ? WHERE note_type = ?", (config["priority"], note_type))

    embedding_columns = [
        ("processing_status", "TEXT DEFAULT 'pending'"), ("error_message", "TEXT"), ("embedding_id", "TEXT")
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
    conn.commit()
    conn.close()
    st.success("🎉 Database successfully migrated!")

@st.cache_resource
def init_advanced_notes_database():
    base_dir = Path("notes")
    base_dir.mkdir(exist_ok=True)
    (base_dir / "metadata").mkdir(exist_ok=True)
    db_path = base_dir / "metadata" / "notes_database.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT, note_type TEXT NOT NULL, title TEXT NOT NULL,
            content_path TEXT, audio_path TEXT, links TEXT, tags TEXT DEFAULT '',
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, last_modified DATETIME DEFAULT CURRENT_TIMESTAMP,
            has_embedding BOOLEAN DEFAULT FALSE, embedding_model TEXT, file_hash TEXT, file_size INTEGER,
            embedding_dimension INTEGER DEFAULT 384, search_priority INTEGER DEFAULT 1
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embedding_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, note_id INTEGER, status TEXT DEFAULT 'pending',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP, started_at DATETIME, completed_at DATETIME,
            error_message TEXT, FOREIGN KEY (note_id) REFERENCES notes (id)
        )
    """)
    conn.commit()
    conn.close()
    migrate_database_to_phase3()
    return str(db_path)

def save_advanced_note(note_type, title, content, links="", tags="", audio_file=None):
    if note_type not in NOTE_CATEGORIES:
        raise ValueError(f"Invalid note type: {note_type}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')[:50]
    content_filename = f"{timestamp}_{safe_title}.txt"
    content_path = Path("notes") / note_type / content_filename
    formatted_content = f"Title: {title}\nCategory: {note_type}\nTags: {tags}\nCreated: {datetime.now().isoformat()}\nLinks: {links}\n{' - ' * 25}\n\n{content}"
    with open(content_path, 'w', encoding='utf-8') as f: f.write(formatted_content)
    
    content_hash = hashlib.md5(formatted_content.encode()).hexdigest()
    file_size = content_path.stat().st_size
    search_priority = NOTE_CATEGORIES[note_type]["priority"]
    
    db_path = Path("notes") / "metadata" / "notes_database.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute("""
        INSERT INTO notes (note_type, title, content_path, links, tags, has_embedding, file_hash, file_size, search_priority)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (note_type, title, str(content_path), links, tags, False, content_hash, file_size, search_priority))
    note_id = cursor.lastrowid
    
    if NOTE_CATEGORIES[note_type]["needs_embedding"]:
        conn.execute("INSERT INTO embedding_jobs (note_id, status) VALUES (?, ?)", (note_id, 'pending'))
    
    conn.commit()
    conn.close()

    if 'drive_instance' in st.session_state and st.session_state.drive_instance:
        with st.spinner("Syncing new note to Google Drive..."):
            gds.sync_directory_to_drive(st.session_state.drive_instance, "notes")
    
    return note_id, str(content_path)

# ================================
# EMBEDDING & SEARCH LOGIC
# ================================
def generate_note_embeddings_batch_with_progress(note_ids: List[int], embedding_model_instance, embedding_dimension=384):
    if not embedding_model_instance or not note_ids: return 0
    
    db_path = Path("notes") / "metadata" / "notes_database.db"
    conn = sqlite3.connect(str(db_path))
    notes_collection = get_notes_chroma_collection(embedding_dimension)
    
    progress_bar = st.progress(0, text="Starting embedding generation...")
    successful_count = 0
    
    for idx, note_id in enumerate(note_ids):
        progress_percent = int(((idx + 1) / len(note_ids)) * 100)
        progress_bar.progress(progress_percent, text=f"Processing note {idx + 1}/{len(note_ids)} (ID: {note_id})")
        
        try:
            conn.execute("UPDATE embedding_jobs SET status = 'processing', started_at = CURRENT_TIMESTAMP WHERE note_id = ?", (note_id,))
            conn.commit()
            
            note_info = conn.execute("SELECT title, content_path, note_type, tags, links FROM notes WHERE id = ?", (note_id,)).fetchone()
            if not note_info: continue

            title, content_path, note_type, tags, links = note_info
            with open(content_path, 'r', encoding='utf-8') as f: content = f.read()

            embedding = embedding_model_instance.encode(content, normalize_embeddings=True).tolist()
            chroma_id = f"note_{note_id}"
            
            notes_collection.add(
                ids=[chroma_id], embeddings=[embedding], documents=[content],
                metadatas=[{"note_id": note_id, "title": title, "note_type": note_type, "tags": tags, "links": links, "content_type": "note"}]
            )
            
            conn.execute("UPDATE notes SET has_embedding = TRUE, embedding_model = ? WHERE id = ?", (embedding_model_name, note_id))
            conn.execute("UPDATE embedding_jobs SET status = 'completed', completed_at = CURRENT_TIMESTAMP WHERE note_id = ?", (note_id,))
            conn.commit()
            successful_count += 1
            
        except Exception as e:
            conn.execute("UPDATE embedding_jobs SET status = 'failed', error_message = ? WHERE note_id = ?", (str(e), note_id))
            conn.commit()
            st.error(f"Error processing note {note_id}: {e}")
            
    conn.close()
    return successful_count

def process_all_pending_embeddings():
    """Fetches ALL pending notes and processes their embeddings."""
    db_path = Path("notes") / "metadata" / "notes_database.db"
    if not db_path.exists():
        st.info("No notes database found.")
        return 0

    global embedding_model
    if not embedding_model:
        embedding_model = get_global_embedding_model()

    conn = sqlite3.connect(str(db_path))
    try:
        # Fetch ALL pending jobs without a limit
        pending_jobs = conn.execute("SELECT note_id FROM embedding_jobs WHERE status = 'pending'").fetchall()
    except sqlite3.OperationalError:
        pending_jobs = conn.execute("SELECT id FROM notes WHERE has_embedding = FALSE").fetchall()
    conn.close()

    if not pending_jobs:
        st.info("No pending notes to embed.")
        return 0

    note_ids = [job[0] for job in pending_jobs]
    
    if embedding_model:
        embedding_dim = len(embedding_model.encode("test"))
    else:
        embedding_dim = 384 # Default dimension

    return generate_note_embeddings_batch_with_progress(note_ids, embedding_model, embedding_dim)

def unified_search(query_text, embedding_model_instance, n_results=10, include_notes=True):
    if not embedding_model_instance:
        return {"products": [], "notes": [], "error": "Embedding model not available"}
    
    try:
        query_embedding = embedding_model_instance.encode(query_text, normalize_embeddings=True).tolist()
    except Exception as e:
        return {"products": [], "notes": [], "error": f"Failed to generate embedding: {e}"}
    
    results = {"products": [], "notes": [], "combined": []}
    embedding_dim = len(query_embedding)
    
    try:
        products_collection = get_local_chroma_collection(embedding_dim)
        product_results = products_collection.query(
            query_embeddings=[query_embedding], n_results=n_results, include=["documents", "metadatas", "distances"]
        )
        results["products"] = product_results
    except Exception as e: results["products"] = {"error": str(e)}
    
    if include_notes:
        try:
            notes_collection = get_notes_chroma_collection(embedding_dim)
            if notes_collection.count() > 0:
                note_results = notes_collection.query(
                    query_embeddings=[query_embedding], n_results=min(n_results, notes_collection.count()), include=["documents", "metadatas", "distances"]
                )
                results["notes"] = note_results
            else:
                results["notes"] = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        except Exception as e: results["notes"] = {"error": str(e)}
    
    combined_results = []
    if results["products"].get("documents") and results["products"]["documents"][0]:
        for doc, meta, dist in zip(results["products"]["documents"][0], results["products"]["metadatas"][0], results["products"]["distances"][0]):
            combined_results.append({"content": doc, "metadata": meta, "distance": dist, "source": "product", "relevance_score": 1.0 - dist})
    
    if results["notes"].get("documents") and results["notes"]["documents"][0]:
        for doc, meta, dist in zip(results["notes"]["documents"][0], results["notes"]["metadatas"][0], results["notes"]["distances"][0]):
            weighted_relevance = (1.0 - dist) * (1.0 + note_context_weight)
            combined_results.append({"content": doc, "metadata": meta, "distance": dist, "source": "note", "relevance_score": weighted_relevance})
            
    combined_results.sort(key=lambda x: x["relevance_score"], reverse=True)
    results["combined"] = combined_results[:n_results * 2]
    
    return results

# Initialize the advanced notes system
notes_db_path = init_advanced_notes_database()

# ================================
# SIDEBAR UI
# ================================
st.sidebar.header("📝 Advanced Notes (Phase 3+)")
stats = get_advanced_notes_stats()
col1, col2 = st.sidebar.columns(2)
col1.metric("Total Notes", stats["total_notes"])
col1.metric("With Embeddings", stats["has_embedding"])
col2.metric("Need Embedding", stats["needs_embedding"])
col2.metric("Pending Jobs", stats.get("pending_jobs", 0))

if stats.get("pending_jobs", 0) > 0:
    if st.sidebar.button("🚀 Process Embedding Queue"):
        processed = process_all_pending_embeddings()
        if processed > 0:
            st.success(f"✅ Processed {processed} embeddings!")
            time.sleep(2)
            st.rerun()

with st.sidebar.expander("🗂️ Manage All Notes", expanded=False):
    if st.button("📋 Show All Notes Manager"):
        st.session_state.show_note_manager = True
        st.rerun()
    
    if st.button("🔄 Generate All Missing Embeddings"):
        processed = process_all_pending_embeddings()
        if processed > 0:
            st.success(f"✅ Generated {processed} embeddings!")
            time.sleep(2)
            st.rerun()
        else:
            st.info("All notes already have embeddings")

with st.sidebar.expander("✨ Create Advanced Note", expanded=False):
    note_type = st.selectbox(
        "Category:", list(NOTE_CATEGORIES.keys()),
        format_func=lambda x: f"{NOTE_CATEGORIES[x]['emoji']} {x.replace('_', ' ').title()}"
    )
    note_title = st.text_input("Title:", placeholder="Enter note title...")
    note_content = st.text_area("Content:", placeholder="Write your note here...", height=100)
    note_links = st.text_input("Links:", placeholder="https://...")
    note_tags = st.text_input("Tags:", placeholder="tag1, tag2, tag3")
    if st.button("💾 Save Advanced Note"):
        if note_title and note_content:
            save_advanced_note(note_type, note_title, note_content, note_links, note_tags)
            st.success("✅ Note saved and queued for embedding!")
            st.rerun()
        else:
            st.warning("Please fill in title and content")

# ================================
# MAIN PAGE UI (Placeholder)
# ================================
if 'show_note_manager' not in st.session_state:
    st.session_state.show_note_manager = False

if st.session_state.show_note_manager:
    st.header("Note Manager would be displayed here...")
    if st.button("Close Manager"):
        st.session_state.show_note_manager = False
        st.rerun()

st.header("🔍 Unified Search")
query_text = st.text_input(
    "Enter your search query:", 
    placeholder="Search across products and your personal notes...",
    key="main_search"
)

if st.button("🚀 Search", type="primary"):
    if query_text:
        st.info(f"Searching for: '{query_text}'")
        # Unified search logic would be here
    else:
        st.warning("Please enter a query.")

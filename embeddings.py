# embeddings.py

import streamlit as st
from sentence_transformers import SentenceTransformer
import torch
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# --- Local Embedding Model Management ---

@st.cache_resource(show_spinner=True)
def load_embedding_model(model_name: str = "BAAI/bge-small-en-v1.5") -> Optional[SentenceTransformer]:
    """Loads and caches the sentence transformer model."""
    try:
        with st.spinner(f"Loading embedding model: {model_name}..."):
            model = SentenceTransformer(model_name)
        st.success(f"✅ Local embedding model '{model_name}' loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading embedding model '{model_name}': {e}")
        return None

# --- Note Embedding Generation (REFACTORED for reliability) ---

def _process_single_note(note_id: int, conn: sqlite3.Connection, embedding_model_instance: SentenceTransformer, embedding_model_name: str, notes_collection):
    """
    Processes a single note to generate and store its embedding.
    This function is designed to be called within an existing DB connection.
    """
    cursor = conn.cursor()
    try:
        note_info = cursor.execute(
            "SELECT title, content_path, note_type, COALESCE(tags, '') as tags, COALESCE(links, '') as links FROM notes WHERE id = ?", (note_id,)
        ).fetchone()

        if not note_info:
            print(f"Warning: Note ID {note_id} not found. Skipping.")
            return False

        title, content_path, note_type, tags, links = note_info
        
        with open(content_path, 'r', encoding='utf-8') as f:
            full_content = f.read()
        
        clean_content = full_content.split("-" * 50, 1)[-1].strip()
        
        if not clean_content:
            print(f"Warning: Note ID {note_id} ('{title}') has no content. Skipping.")
            return False

        embedding = embedding_model_instance.encode(clean_content, normalize_embeddings=True).tolist()
        chroma_id = f"note_{note_id}"

        notes_collection.add(
            ids=[chroma_id],
            embeddings=[embedding],
            documents=[clean_content],
            metadatas=[{"note_id": note_id, "title": title, "note_type": note_type, 
                        "tags": tags, "links": links, "content_type": "note"}]
        )
        
        cursor.execute("UPDATE notes SET has_embedding = TRUE, embedding_model = ? WHERE id = ?", (embedding_model_name, note_id))
        cursor.execute("UPDATE embedding_jobs SET status = 'completed', completed_at = CURRENT_TIMESTAMP WHERE note_id = ?", (note_id,))
        
        print(f"Successfully processed embedding for note ID {note_id} ('{title}').")
        return True

    except Exception as e:
        error_msg = f"Error on note {note_id}: {e}"
        print(error_msg)
        cursor.execute("UPDATE embedding_jobs SET status = 'failed', error_message = ? WHERE note_id = ?", (str(e), note_id))
        return False

def process_embedding_queue(embedding_model_instance: SentenceTransformer, embedding_model_name: str, notes_collection):
    """
    Processes all pending notes from the embedding queue one by one for reliability.
    """
    db_path = Path("notes") / "metadata" / "notes_database.db"
    if not db_path.exists():
        return 0

    conn = sqlite3.connect(str(db_path))
    try:
        pending_jobs = conn.execute("SELECT note_id FROM embedding_jobs WHERE status = 'pending' ORDER BY created_at ASC").fetchall()
    except sqlite3.OperationalError:
        pending_jobs = []
    
    if not pending_jobs:
        conn.close()
        return 0
        
    note_ids_to_process = [job[0] for job in pending_jobs]
    successful_count = 0
    
    # Use a UI placeholder for progress updates
    progress_text = st.empty()
    
    for i, note_id in enumerate(note_ids_to_process):
        progress_text.info(f"⚙️ Processing note {i + 1}/{len(note_ids_to_process)}...")
        if _process_single_note(note_id, conn, embedding_model_instance, embedding_model_name, notes_collection):
            successful_count += 1
    
    conn.commit()
    conn.close()
    
    progress_text.empty() # Clear the progress text
    return successful_count

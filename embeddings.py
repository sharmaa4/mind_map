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
    """
    Loads a local Sentence Transformer model and caches it.

    Args:
        model_name (str): The name of the model to load from Hugging Face.

    Returns:
        Optional[SentenceTransformer]: The loaded model instance, or None if an error occurs.
    """
    try:
        with st.spinner(f"Loading embedding model: {model_name}..."):
            model = SentenceTransformer(model_name)
        st.success(f"✅ Local embedding model '{model_name}' loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading embedding model '{model_name}': {e}")
        st.write("Please ensure 'sentence-transformers' is installed: `pip install sentence-transformers`")
        return None

def get_query_embedding_local(query_text: str, model: SentenceTransformer) -> Optional[list]:
    """
    Generates a vector embedding for a given query text using a local model.

    Args:
        query_text (str): The text to be embedded.
        model (SentenceTransformer): The loaded sentence transformer model.

    Returns:
        Optional[list]: The generated embedding as a list of floats, or None on error.
    """
    if model is None:
        st.error("❌ Embedding model is not available for query processing.")
        return None
    
    try:
        embedding = model.encode(query_text, normalize_embeddings=True)
        return embedding.tolist()
    except Exception as e:
        st.error(f"❌ Error generating local embedding: {e}")
        return None

# --- Note Embedding Generation and Queue Management ---

def generate_note_embeddings_batch_with_progress(
    note_ids: List[int], 
    embedding_model_instance: SentenceTransformer, 
    notes_collection
):
    """
    Generates and stores embeddings for a batch of notes with a visual progress bar.

    Args:
        note_ids (List[int]): A list of note IDs to process.
        embedding_model_instance (SentenceTransformer): The loaded embedding model.
        notes_collection (chromadb.Collection): The ChromaDB collection to store note embeddings.

    Returns:
        int: The number of successfully processed notes.
    """
    if not embedding_model_instance:
        st.error("Embedding model not available. Cannot process notes.")
        return 0
    
    db_path = Path("notes") / "metadata" / "notes_database.db"
    conn = sqlite3.connect(str(db_path))
    
    progress_bar = st.progress(0, text="Starting embedding generation...")
    status_text = st.empty()
    successful_count = 0
    
    for idx, note_id in enumerate(note_ids):
        try:
            progress_bar.progress((idx + 1) / len(note_ids), text=f"Processing note {idx + 1}/{len(note_ids)}...")
            
            note_info = conn.execute(
                "SELECT title, content_path, note_type, COALESCE(tags, '') as tags, COALESCE(links, '') as links FROM notes WHERE id = ?", (note_id,)
            ).fetchone()

            if not note_info:
                status_text.warning(f"Note ID {note_id} not found. Skipping.")
                continue
            
            title, content_path, note_type, tags, links = note_info
            
            with open(content_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            embedding = embedding_model_instance.encode(content, normalize_embeddings=True).tolist()
            chroma_id = f"note_{note_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            notes_collection.add(
                ids=[chroma_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[{"note_id": note_id, "title": title, "note_type": note_type, "tags": tags, "links": links, "content_type": "note"}]
            )
            
            conn.execute("UPDATE notes SET has_embedding = TRUE, embedding_model = ? WHERE id = ?", (embedding_model_instance.get_name(), note_id))
            conn.execute("UPDATE embedding_jobs SET status = 'completed', completed_at = CURRENT_TIMESTAMP WHERE note_id = ?", (note_id,))
            successful_count += 1
            
        except Exception as e:
            conn.execute("UPDATE embedding_jobs SET status = 'failed', error_message = ? WHERE note_id = ?", (str(e), note_id))
            status_text.error(f"Error on note {note_id}: {e}")
            
    conn.commit()
    conn.close()
    
    progress_bar.progress(100, text="✅ Embedding generation complete!")
    status_text.success(f"Successfully processed {successful_count}/{len(note_ids)} notes.")
    return successful_count

def process_embedding_queue(embedding_model_instance, notes_collection):
    """
    Checks for and processes any pending notes that need embeddings.
    
    Args:
        embedding_model_instance (SentenceTransformer): The model to use for embedding.
        notes_collection (chromadb.Collection): The collection where embeddings are stored.

    Returns:
        int: The number of notes for which embeddings were successfully generated.
    """
    db_path = Path("notes") / "metadata" / "notes_database.db"
    if not db_path.exists():
        return 0
    
    conn = sqlite3.connect(str(db_path))
    try:
        pending_jobs = conn.execute("SELECT note_id FROM embedding_jobs WHERE status = 'pending' ORDER BY created_at ASC LIMIT 10").fetchall()
    finally:
        conn.close()
        
    if not pending_jobs:
        return 0
        
    note_ids_to_process = [job[0] for job in pending_jobs]
    return generate_note_embeddings_batch_with_progress(note_ids_to_process, embedding_model_instance, notes_collection)

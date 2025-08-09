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
    try:
        with st.spinner(f"Loading embedding model: {model_name}..."):
            model = SentenceTransformer(model_name)
        st.success(f"✅ Local embedding model '{model_name}' loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading embedding model '{model_name}': {e}")
        return None

def get_query_embedding_local(query_text: str, model: SentenceTransformer) -> Optional[list]:
    if model is None:
        st.error("❌ Embedding model is not available for query processing.")
        return None
    try:
        embedding = model.encode(query_text, normalize_embeddings=True)
        return embedding.tolist()
    except Exception as e:
        st.error(f"❌ Error generating local embedding: {e}")
        return None

# --- Note Embedding Generation and Queue Management (REFACTORED) ---

def generate_note_embeddings_batch_with_progress(
    note_ids: List[int],
    embedding_model_instance: SentenceTransformer,
    embedding_model_name: str,
    notes_collection
):
    if not embedding_model_instance:
        st.error("Embedding model not available. Cannot process notes.")
        return 0
    if not note_ids:
        return 0

    db_path = Path("notes") / "metadata" / "notes_database.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    progress_bar = st.progress(0, text="Starting embedding generation...")
    status_text = st.empty()
    successful_count = 0

    # Data lists for batch processing
    chroma_ids = []
    chroma_embeddings = []
    chroma_documents = []
    chroma_metadatas = []
    completed_note_ids = []

    try:
        for idx, note_id in enumerate(note_ids):
            progress_text = f"Processing note {idx + 1}/{len(note_ids)}..."
            progress_bar.progress((idx + 1) / len(note_ids), text=progress_text)
            print(progress_text)

            try:
                note_info = cursor.execute(
                    "SELECT title, content_path, note_type, COALESCE(tags, '') as tags, COALESCE(links, '') as links FROM notes WHERE id = ?", (note_id,)
                ).fetchone()

                if not note_info:
                    status_text.warning(f"Note ID {note_id} not found. Skipping.")
                    continue
                
                title, content_path, note_type, tags, links = note_info
                
                with open(content_path, 'r', encoding='utf-8') as f:
                    full_content_with_header = f.read()
                
                clean_content = full_content_with_header.split("-" * 50, 1)[-1].strip()
                
                if not clean_content:
                    status_text.warning(f"Note ID {note_id} has no content after header. Skipping.")
                    continue

                embedding = embedding_model_instance.encode(clean_content, normalize_embeddings=True).tolist()
                
                # Prepare data for batch ChromaDB insertion
                chroma_ids.append(f"note_{note_id}")
                chroma_embeddings.append(embedding)
                chroma_documents.append(clean_content)
                chroma_metadatas.append({
                    "note_id": note_id, "title": title, "note_type": note_type, 
                    "tags": tags, "links": links, "content_type": "note"
                })
                completed_note_ids.append(note_id)
                
            except Exception as e:
                error_msg = f"Error processing note {note_id}: {e}"
                print(error_msg)
                status_text.error(error_msg)
                cursor.execute("UPDATE embedding_jobs SET status = 'failed', error_message = ? WHERE note_id = ?", (str(e), note_id))

        # Perform batch operations after the loop
        if chroma_ids:
            notes_collection.add(
                ids=chroma_ids,
                embeddings=chroma_embeddings,
                documents=chroma_documents,
                metadatas=chroma_metadatas
            )
            print(f"Successfully added {len(chroma_ids)} embeddings to ChromaDB.")

            for note_id in completed_note_ids:
                cursor.execute("UPDATE notes SET has_embedding = TRUE, embedding_model = ? WHERE id = ?", (embedding_model_name, note_id))
                cursor.execute("UPDATE embedding_jobs SET status = 'completed', completed_at = CURRENT_TIMESTAMP WHERE note_id = ?", (note_id,))
            
            successful_count = len(completed_note_ids)

        conn.commit()

    except Exception as e:
        conn.rollback()
        st.error(f"A critical error occurred during batch processing: {e}")
    finally:
        conn.close()
    
    progress_bar.progress(100, text="✅ Embedding generation complete!")
    status_text.success(f"Successfully processed {successful_count}/{len(note_ids)} notes.")
    return successful_count


def process_embedding_queue(embedding_model_instance: SentenceTransformer, embedding_model_name: str, notes_collection):
    db_path = Path("notes") / "metadata" / "notes_database.db"
    if not db_path.exists():
        return 0
    
    conn = sqlite3.connect(str(db_path))
    try:
        pending_jobs = conn.execute("SELECT note_id FROM embedding_jobs WHERE status = 'pending' ORDER BY created_at ASC").fetchall()
    finally:
        conn.close()
        
    if not pending_jobs:
        return 0
        
    note_ids_to_process = [job[0] for job in pending_jobs]
    return generate_note_embeddings_batch_with_progress(note_ids_to_process, embedding_model_instance, embedding_model_name, notes_collection)

# embeddings.py

import streamlit as st
from sentence_transformers import SentenceTransformer
import torch
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import traceback

# Keep model loading cached so repeated calls are cheap
@st.cache_resource(show_spinner=True)
def load_embedding_model(model_name: str = "BAAI/bge-small-en-v1.5") -> Optional[SentenceTransformer]:
    """Loads and caches the sentence transformer model."""
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        print(f"[embeddings] Error loading embedding model '{model_name}': {e}")
        traceback.print_exc()
        return None


def _process_single_note(note_id: int, conn: sqlite3.Connection, embedding_model_instance, embedding_model_name: str, notes_collection) -> bool:
    """
    Internal single-note embedding/ingest implementation.
    Assumes notes_collection is an already-obtained Chroma collection instance.
    """
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT content_path, title FROM notes WHERE id = ?", (note_id,))
        row = cursor.fetchone()
        if not row:
            print(f"[embeddings] Note id {note_id} not found.")
            return False
        content_path, title = row
        if not Path(content_path).exists():
            print(f"[embeddings] Content file missing for note {note_id}: {content_path}")
            # mark job failed
            cursor.execute("UPDATE embedding_jobs SET status = 'failed', last_error = ? WHERE note_id = ?", ("file missing", note_id))
            return False

        with open(content_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Simple chunk logic: split into paragraphs up to 500 tokens approx by char
        chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
        if not chunks:
            chunks = [content[:1000]]

        # embed each chunk
        texts = chunks
        embeddings = embedding_model_instance.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        # create ids & metadatas
        ids = [f"note_{note_id}_chunk_{i}" for i in range(len(embeddings))]
        metadatas = [{"note_id": note_id, "title": title, "chunk_index": i} for i in range(len(embeddings))]
        documents = [t[:500] for t in texts]

        # add to Chroma collection (notes_collection is expected to be a chroma collection instance)
        try:
            notes_collection.add(ids=ids, embeddings=embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings, metadatas=metadatas, documents=documents)
        except Exception as e:
            # fallback: try batch-add in smaller chunks
            try:
                batch_size = 50
                for i in range(0, len(ids), batch_size):
                    notes_collection.add(ids=ids[i:i+batch_size], embeddings=(embeddings[i:i+batch_size].tolist() if hasattr(embeddings, "tolist") else embeddings[i:i+batch_size]), metadatas=metadatas[i:i+batch_size], documents=documents[i:i+batch_size])
            except Exception as e2:
                print(f"[embeddings] Failed to add embeddings to collection: {e2}")
                cursor.execute("UPDATE embedding_jobs SET status = 'failed', last_error = ? WHERE note_id = ?", (str(e2), note_id))
                conn.commit()
                return False

        # mark note has_embedding true
        cursor.execute("UPDATE notes SET has_embedding = TRUE WHERE id = ?", (note_id,))
        cursor.execute("UPDATE embedding_jobs SET status = 'completed', updated_at = CURRENT_TIMESTAMP WHERE note_id = ?", (note_id,))
        conn.commit()
        print(f"[embeddings] Successfully processed embedding for note ID {note_id} ('{title}').")
        return True

    except Exception as e:
        print(f"[embeddings] Error processing note {note_id}: {e}")
        traceback.print_exc()
        try:
            cursor.execute("UPDATE embedding_jobs SET status = 'failed', last_error = ? WHERE note_id = ?", (str(e), note_id))
            conn.commit()
        except:
            pass
        return False


def process_embedding_queue(embedding_model_instance, embedding_model_name: str, notes_collection, limit: int = 50) -> int:
    """
    Processes pending embedding jobs found in the SQLite DB and writes vectors to the provided notes_collection.

    MUST be passed the same notes_collection instance that the rest of the app is using (from vector_db.get_notes_chroma_collection()).
    Returns number of successful embeddings processed.
    """
    if embedding_model_instance is None:
        print("[embeddings] embedding_model_instance is None; cannot process embeddings.")
        return 0

    db_path = Path.cwd() / "notes" / "metadata" / "notes_database.db"
    if not db_path.exists():
        print("[embeddings] DB not present; nothing to process.")
        return 0

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT id, note_id FROM embedding_jobs WHERE status = 'pending' ORDER BY created_at ASC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    if not rows:
        conn.close()
        print("[embeddings] No pending embedding jobs found.")
        return 0

    note_ids_to_process = [r[1] for r in rows]
    successful_count = 0

    # Progress logging
    print(f"[embeddings] Processing {len(note_ids_to_process)} pending embedding jobs...")

    for i, note_id in enumerate(note_ids_to_process):
        print(f"[embeddings] Processing ({i+1}/{len(note_ids_to_process)}) note_id={note_id}")
        ok = _process_single_note(note_id, conn, embedding_model_instance, embedding_model_name, notes_collection)
        if ok:
            successful_count += 1

    conn.close()
    print(f"[embeddings] Finished. Successfully processed {successful_count} embeddings.")
    return successful_count


# vector_db.py

# --- Inject pysqlite3 as sqlite3 if available (must be before chromadb import) ---
import sys
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
    print("[vector_db] Using pysqlite3 as sqlite3 backend (injected in vector_db).")
except Exception as e:
    print(f"[vector_db] pysqlite3 injection unavailable: {e} (continuing, chromadb may error if sqlite is too old)")

import streamlit as st

# chromadb import follows after we tried the injection
import chromadb
from chromadb.config import Settings
import os
from pathlib import Path
from typing import Optional

# Centralized constants (use same collection name everywhere)
PERSIST_DIRECTORY = str(Path.cwd() / "db_local")
NOTES_COLLECTION_NAME = "notes_collection_v1"  # change if you intentionally version collections
DEFAULT_EMBEDDING_DIM = 768  # adjust if your embedding dims differ

# --- ChromaDB Client and Collection Management ---


@st.cache_resource(show_spinner=False)
def get_chroma_client(persist_directory: str = PERSIST_DIRECTORY) -> chromadb.PersistentClient:
    os.makedirs(persist_directory, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_directory, settings=Settings())
    return client


def get_notes_chroma_collection(collection_name: Optional[str] = None, dim: int = DEFAULT_EMBEDDING_DIM, persist_directory: str = PERSIST_DIRECTORY):
    """
    Returns a Chroma collection for notes. Always uses a deterministic collection name by default.
    This function intentionally does NOT cache the collection itself via st.cache to avoid subtle
    cache-key mismatches; the client is cached via st.cache_resource above.
    """
    client = get_chroma_client(persist_directory=persist_directory)
    if collection_name is None:
        collection_name = NOTES_COLLECTION_NAME

    # Use get_collection with create_if_not_exists semantics (Chroma SDK differs across versions;
    # the pattern below is safe for most versions)
    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        # If get_collection fails because it doesn't exist, create it
        collection = client.create_collection(name=collection_name)

    return collection


def list_collections(persist_directory: str = PERSIST_DIRECTORY):
    client = get_chroma_client(persist_directory=persist_directory)
    try:
        return client.list_collections()
    except Exception as e:
        print(f"[vector_db] Error listing collections: {e}")
        return []


def delete_collection(collection_name: str, persist_directory: str = PERSIST_DIRECTORY):
    client = get_chroma_client(persist_directory=persist_directory)
    try:
        client.delete_collection(name=collection_name)
        return True
    except Exception as e:
        print(f"[vector_db] Error deleting collection {collection_name}: {e}")
        return False


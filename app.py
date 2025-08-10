import streamlit as st
import streamlit.components.v1 as components
import pysqlite3
import sys
sys.modules["sqlite3"] = pysqlite3

import os
from pathlib import Path
from datetime import datetime

from database import (
    migrate_database,
    get_all_notes_with_details,
    # other DB functions like insert_note, update_note, delete_note...
)
from google_drive_sync import sync_from_gdrive, sync_to_gdrive
from embeddings import load_embedding_model, process_embedding_queue
from vector_db import get_notes_chroma_collection, NOTES_COLLECTION_NAME
from search import search_notes

# ─────────────────────────────────────────
# Run schema migration before anything else
# ─────────────────────────────────────────
migrate_database()

# ─────────────────────────────────────────
# App Config
# ─────────────────────────────────────────
st.set_page_config(page_title="Mind Map Notes", layout="wide")

# ─────────────────────────────────────────
# Sidebar - Actions
# ─────────────────────────────────────────
st.sidebar.header("Actions")

if st.sidebar.button("Sync from Google Drive"):
    try:
        sync_from_gdrive()
        migrate_database()  # Ensure DB schema is up-to-date after sync
        st.sidebar.success("Synced from Google Drive successfully!")
    except Exception as e:
        st.sidebar.error(f"Sync failed: {e}")

if st.sidebar.button("Sync to Google Drive"):
    try:
        sync_to_gdrive()
        st.sidebar.success("Synced to Google Drive successfully!")
    except Exception as e:
        st.sidebar.error(f"Sync failed: {e}")

st.sidebar.markdown("---")

# ─────────────────────────────────────────
# Sidebar - New Note
# ─────────────────────────────────────────
st.sidebar.subheader("Create New Note")
new_title = st.sidebar.text_input("Title")
new_content = st.sidebar.text_area("Content")

if st.sidebar.button("Save Note"):
    if new_title.strip() == "" or new_content.strip() == "":
        st.sidebar.warning("Please enter both title and content.")
    else:
        from database import insert_note
        note_id = insert_note(
            note_type="text",
            title=new_title.strip(),
            content=new_content.strip(),
            created_at=datetime.now()
        )
        st.sidebar.success(f"Note '{new_title}' created!")
        process_embedding_queue()  # Keep embeddings updated

st.sidebar.markdown("---")

# ─────────────────────────────────────────
# Search Section
# ─────────────────────────────────────────
st.subheader("Search Notes")
query = st.text_input("Enter search query")
if st.button("Search"):
    if query.strip():
        results = search_notes(query)
        if results:
            st.write("### Search Results")
            for r in results:
                st.write(f"- **{r['title']}**: {r['content'][:100]}...")
        else:
            st.info("No matching notes found.")

st.markdown("---")

# ─────────────────────────────────────────
# All Notes Table
# ─────────────────────────────────────────
def display_notes_table(limit: int = 200):
    rows = get_all_notes_with_details(limit=limit)
    if not rows:
        st.info("No notes found.")
        return

    st.write("### All Notes")
    for note_id, note_type, title, content_path, links, created_at in rows:
        st.markdown(f"**{title}** ({created_at})")
        if content_path and Path(content_path).exists():
            with open(content_path, "r") as f:
                preview = f.read(200)
                st.code(preview)
        else:
            st.code("[Content missing]")

display_notes_table(limit=200)


# app.py

# --- FIX: Apply the pysqlite3 patch BEFORE any other imports ---
import pysqlite3
import sys
sys.modules["sqlite3"] = pysqlite3
# ----------------------------------------------------------------

import streamlit as st
import time
from pathlib import Path

# Import the newly created modules for backend logic
import database as db
import vector_db as vdb
import embeddings as emb
import search
import ai_services
import utils
import google_drive_sync as gds

# --- Page Configuration ---
st.set_page_config(page_title="AI Knowledge Management System", layout="wide")
st.title("🚀 AI Knowledge Management System (Phase 3+)")

# --- Google Drive Synchronization ---
if 'drive_synced' not in st.session_state:
    st.session_state.drive_synced = False
if 'drive_instance' not in st.session_state:
    st.session_state.drive_instance = None

@st.cache_resource(show_spinner="Connecting to Google Drive and syncing data...")
def initial_sync():
    """Authenticates with Google Drive and performs the initial data sync."""
    try:
        drive = gds.authenticate_gdrive()
        gds.sync_directory_from_drive(drive, "notes")
        gds.sync_directory_from_drive(drive, "product_embeddings_v2")
        return drive
    except Exception as e:
        st.error(f"Fatal Error: Could not sync with Google Drive. Details: {e}")
        return None

if not st.session_state.drive_synced:
    drive_instance = initial_sync()
    if drive_instance:
        st.session_state.drive_synced = True
        st.session_state.drive_instance = drive_instance
        st.sidebar.success("✅ Synced with Google Drive!")
        st.rerun()
    else:
        st.sidebar.error("❌ Google Drive sync failed. App cannot continue.")
        st.stop()

# --- Sidebar UI ---
st.sidebar.success("🎉 Phase 3+: Complete Knowledge Management!")
try:
    st.sidebar.image("logo.png", width=200)
except Exception:
    pass

st.sidebar.header("🤖 AI Model Selection")
selected_model = st.sidebar.selectbox("Choose AI Model:", ["gpt-4o-mini", "gpt-4o"], index=0)
st.sidebar.header("📊 Embedding Model")
embedding_model_name = st.sidebar.selectbox("Choose Embedding Model:", ["BAAI/bge-small-en-v1.5", "all-MiniLM-L6-v2"], index=0)
st.sidebar.header("⚡ Enhanced Features")
enable_streaming = st.sidebar.checkbox("🔄 Enable Streaming Output", value=True)
enable_context = st.sidebar.checkbox("🧠 Enable Context Awareness", value=True)
max_context_messages = st.sidebar.slider("📝 Context History Length", 1, 10, 5)
st.sidebar.header("🔍 Advanced Search")
enable_unified_search = st.sidebar.checkbox("🔗 Unified Search (Products + Notes)", value=True)
note_context_weight = st.sidebar.slider("📝 Note Context Weight", 0.0, 1.0, 0.3)

# --- Initialization ---
if 'conversation_history' not in st.session_state: st.session_state.conversation_history = []
if 'show_note_manager' not in st.session_state: st.session_state.show_note_manager = False
notes_db_path = db.init_advanced_notes_database()
embedding_model = emb.load_embedding_model(embedding_model_name)
if embedding_model:
    embedding_dim = embedding_model.get_sentence_embedding_dimension()
    products_collection = vdb.get_local_chroma_collection("analog_products_local", embedding_dim)
    notes_collection = vdb.get_notes_chroma_collection("user_notes_local", embedding_dim)

# --- Sidebar: Advanced Notes UI ---
st.sidebar.header("📝 Advanced Notes")
stats = db.get_advanced_notes_stats()
if stats:
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Total Notes", stats.get("total_notes", 0))
    col2.metric("Need Embedding", stats.get("needs_embedding", 0))
    if stats.get("pending_jobs", 0) > 0 and st.sidebar.button("🚀 Process Embedding Queue"):
        with st.sidebar:
            processed_count = emb.process_embedding_queue(embedding_model, embedding_model_name, notes_collection)
            st.success(f"Processed {processed_count} notes.")
            st.rerun()

with st.sidebar.expander("✨ Create Advanced Note"):
    note_type = st.selectbox("Category:", list(db.NOTE_CATEGORIES.keys()), format_func=lambda x: f"{db.NOTE_CATEGORIES[x]['emoji']} {x.replace('_', ' ').title()}")
    note_title = st.text_input("Title:", placeholder="Enter note title...")
    note_content = st.text_area("Content:", placeholder="Write your note here...", height=100)
    note_links = st.text_input("Links:", placeholder="https://...")
    note_tags = st.text_input("Tags:", placeholder="tag1, tag2")
    if st.button("💾 Save Advanced Note"):
        if note_title and note_content:
            note_id, _ = db.save_advanced_note(note_type, note_title, note_content, note_links, note_tags)
            st.success(f"Note saved! ID: {note_id}")
            gds.sync_directory_to_drive(st.session_state.drive_instance, "notes")
            st.rerun()
        else:
            st.warning("Please provide a title and content.")

# Note Management Expander
with st.sidebar.expander("🗂️ Manage All Notes"):
    if st.button("📋 Show All Notes Manager"):
        st.session_state.show_note_manager = True
        st.rerun()

# --- Main Page UI ---

# Note Manager Display Logic
if st.session_state.show_note_manager:
    st.header("🗂️ Comprehensive Note Manager")
    if st.button("❌ Close Note Manager"):
        st.session_state.show_note_manager = False
        st.rerun()

    all_notes = db.get_all_notes_with_details()
    if not all_notes:
        st.info("No notes found. Create one from the sidebar!")
    else:
        for note in all_notes:
            with st.expander(f"{db.NOTE_CATEGORIES[note['type']]['emoji']} {note['title']}"):
                st.write(f"**Type:** {note['type'].replace('_', ' ').title()}")
                st.write(f"**Last Modified:** {note['last_modified'][:19]}")
                st.write(f"**Embedding Status:** {'✅ Ready' if note['has_embedding'] else '⏳ Pending'}")

                if st.button("🗑️ Delete Note", key=f"delete_{note['id']}"):
                    db.delete_note(note['id'])
                    vdb.delete_note_embedding([note['id']], notes_collection)
                    gds.sync_directory_to_drive(st.session_state.drive_instance, "notes")
                    st.success(f"Note '{note['title']}' and its embeddings have been deleted.")
                    st.rerun()

st.header("🔍 Unified Search (Products & Notes)")
query_text = st.text_input("Enter your search query:", placeholder="Search across datasheets and your personal notes...")

if st.button("🚀 Unified Search", type="primary"):
    if query_text and embedding_model:
        with st.spinner("Performing unified search..."):
            results = search.unified_search(
                query_text, 
                embedding_model, 
                products_collection, 
                notes_collection, 
                note_context_weight, 
                n_results=20,
                include_notes=enable_unified_search
            )
            
            if results.get("error"):
                st.error(results["error"])
            elif not results.get("combined"):
                st.warning("No relevant results found.")
            else:
                st.success("Search complete! Generating AI summary...")
                
                all_products = [r for r in results['combined'] if r['source'] == 'product']
                all_notes = [r for r in results['combined'] if r['source'] == 'note']
                
                product_context = "\n\n".join([p['content'] for p in all_products])
                note_context = "\n\n".join([n['content'] for n in all_notes[:15]])

                ai_services.get_ai_response(
                    user_query=query_text, document_context=product_context, note_context=note_context,
                    conversation_history=st.session_state.conversation_history, model=selected_model,
                    enable_streaming=enable_streaming, enable_context=enable_context,
                    max_context_messages=max_context_messages
                )
                utils.add_to_conversation("user", query_text)
                utils.extract_and_display_unified_results(results)

with st.expander("Conversation Management"):
    if st.button("🧹 Clear Conversation History"): utils.clear_conversation_history()
    if st.button("📋 View Conversation History"): st.json(st.session_state.conversation_history)

with st.expander("🗂️ Data Management"):
    if st.button("📂 Ingest Local Product Embeddings"):
        vdb.ingest_local_embeddings(products_collection)

# app.py
import streamlit as st
import pandas as pd
from pathlib import Path
import time

# ---- imports from your modules ----
from google_drive_sync import authenticate_gdrive, sync_directory_from_drive
from database import (
    init_advanced_notes_database,
    scan_and_queue_new_notes,
    save_advanced_note,
    get_all_notes_with_details,
    get_pending_embedding_jobs,
)
from vector_db import get_notes_chroma_collection, NOTES_COLLECTION_NAME, PERSIST_DIRECTORY
from embeddings import load_embedding_model, process_embedding_queue

# ---- App constants & startup ----
st.set_page_config(page_title="Notes (Sync + Search) — Streamlit", layout="wide")

# Ensure DB exists and schema present right away (prevents race conditions)
db_path = init_advanced_notes_database()

# Cached single Chroma collection instance used everywhere
@st.cache_resource(show_spinner=False)
def _get_collection():
    return get_notes_chroma_collection(collection_name=NOTES_COLLECTION_NAME, persist_directory=PERSIST_DIRECTORY)

notes_collection = _get_collection()

# Cached model load (embeddings.py already caches, but this provides a local handle)
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)


# ---- Helper UI functions ----
def display_notes_table(limit: int = 200):
    rows = get_all_notes_with_details(limit=limit)
    if not rows:
        st.info("No notes found.")
        return
    df = pd.DataFrame(rows, columns=["id", "note_type", "title", "content_path", "links", "tags", "has_embedding"])
    st.dataframe(df.sort_values("id", ascending=False).reset_index(drop=True), use_container_width=True)


def run_sync_and_scan(drive=None, parent_folder_id=None):
    # If drive provided, sync; else just scan (useful for manual file drops)
    if drive is not None:
        st.info("Syncing from Google Drive...")
        try:
            local_path = sync_directory_from_drive(drive)
            st.success(f"Drive sync extracted to: {local_path}")
        except Exception as e:
            st.error(f"Drive sync failed: {e}")
            return False
    # Register files into DB (guarded init inside function)
    st.info("Scanning local notes and queueing embeddings (if needed)...")
    try:
        scan_and_queue_new_notes()
        st.success("Scan completed.")
        return True
    except Exception as e:
        st.error(f"Scan failed: {e}")
        return False


def run_process_embedding_queue(limit: int = 50):
    if embedding_model is None:
        st.error("Embedding model not loaded; cannot process embeddings.")
        return 0
    st.info("Processing embedding queue (this may take some time)...")
    t0 = time.time()
    count = process_embedding_queue(embedding_model, EMBEDDING_MODEL_NAME, notes_collection, limit=limit)
    t1 = time.time()
    st.success(f"Processed {count} embedding jobs in {t1 - t0:.1f}s")
    return count


def semantic_search(query: str, top_k: int = 5):
    if embedding_model is None:
        st.error("Embedding model not loaded; cannot embed query.")
        return []
    if query.strip() == "":
        return []

    qvec = embedding_model.encode([query], show_progress_bar=False, convert_to_numpy=True)[0]
    try:
        # chroma collection query - ask for documents/metadatas/distances
        results = notes_collection.query(
            embeddings=[qvec.tolist() if hasattr(qvec, "tolist") else qvec],
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
        )
        # results is a dict-like structure
        hits = []
        if results:
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]
            for doc, meta, dist in zip(docs, metas, dists):
                hits.append({"document": doc, "metadata": meta, "distance": dist})
        return hits
    except Exception as e:
        st.error(f"Search failed: {e}")
        return []


# ---- Layout ----
st.title("Notes — Drive Sync, DB & Semantic Search")
left_col, right_col = st.columns([3, 1])

with right_col:
    st.header("Actions")

    # Option: Automatic sync on start toggle
    auto_sync = st.checkbox("Sync from Drive on start", value=False, help="If checked the app will attempt to download notes.zip from Drive on startup (requires configured secrets).")

    # Manual Drive sync button & flow
    if st.button("Authenticate & Sync from Drive"):
        try:
            drive = authenticate_gdrive()
            run_sync_and_scan(drive=drive)
        except Exception as e:
            st.error(f"Drive auth/sync error: {e}")

    # Manual scan (useful if you drop files into notes/ folder locally)
    if st.button("Scan local notes & queue"):
        run_sync_and_scan(drive=None)

    # Process embeddings now
    if st.button("Process embedding queue (run)"):
        run_process_embedding_queue(limit=100)

    # Show pending jobs
    pending = get_pending_embedding_jobs(limit=10)
    st.write("Pending embedding jobs (top 10):")
    if pending:
        st.table(pd.DataFrame(pending, columns=["job_id", "note_id", "status"]))
    else:
        st.write("No pending jobs.")

    # Info about collection
    st.markdown("**Chroma collection**")
    st.write(f"Collection name: `{NOTES_COLLECTION_NAME}`")
    try:
        collections = [c['name'] for c in notes_collection._client.list_collections()]  # introspect client
        st.write("Persist dir:", PERSIST_DIRECTORY)
        st.write("Collections found in persist dir:", collections)
    except Exception:
        st.write("Collections: (could not list, but collection object exists)")

with left_col:
    # On-first-load actions: optionally auto-sync then scan
    if auto_sync and "startup_synced" not in st.session_state:
        st.session_state["startup_synced"] = True
        try:
            drive = authenticate_gdrive()
            run_sync_and_scan(drive=drive)
        except Exception as e:
            st.warning(f"Auto-sync failed (you can try manual sync): {e}")

    # Sidebar-like note creation area (left pane top)
    st.subheader("Create a new note")
    with st.form("create_note_form", clear_on_submit=False):
        title = st.text_input("Title", "")
        from database import NOTE_CATEGORIES  # read categories from database module
        category = st.selectbox("Category", options=list(NOTE_CATEGORIES.keys()), index=0)
        tags = st.text_input("Tags (comma-separated)", "")
        links = st.text_input("Links (comma-separated)", "")
        content = st.text_area("Content", height=180)
        embed_now = st.checkbox("Embed now after create (runs embedding queue)", value=False)
        submitted = st.form_submit_button("Save note")
        if submitted:
            if not title and not content.strip():
                st.error("Note needs at least a title or content.")
            else:
                try:
                    note_id = save_advanced_note(category, title or "Untitled", content, links=links, tags=tags)
                    st.success(f"Saved note (id={note_id}).")
                    # optionally run scan to be safe (but save already inserted)
                    scan_and_queue_new_notes()
                    if embed_now:
                        run_process_embedding_queue(limit=10)
                except Exception as e:
                    st.error(f"Failed to save note: {e}")

    st.markdown("---")
    # Simple semantic search
    st.subheader("Semantic search")
    q = st.text_input("Query", "")
    top_k = st.slider("Top K", min_value=1, max_value=20, value=5)
    if st.button("Search"):
        if q.strip() == "":
            st.warning("Enter a query.")
        else:
            with st.spinner("Running semantic search..."):
                hits = semantic_search(q, top_k)
            if not hits:
                st.info("No matches found or collection empty.")
            else:
                for i, h in enumerate(hits, start=1):
                    meta = h.get("metadata", {})
                    st.markdown(f"**#{i}** — Title: {meta.get('title')} — note_id: {meta.get('note_id')} — distance: {h.get('distance'):.4f}")
                    st.write(h.get("document"))

    st.markdown("---")
    st.subheader("All notes (recent)")
    display_notes_table(limit=200)


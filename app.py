# app.py

# --- Apply the pysqlite3 patch and disable telemetry ---
import os
os.environ["CHROMADB_DISABLE_TELEMETRY"] = "true"
import pysqlite3
import sys
sys.modules["sqlite3"] = pysqlite3
# -------------------------------------------------------------------------------------

import streamlit as st

# --- Backend Modules ---
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

# --- Single, Unified Initialization Function ---
@st.cache_resource(show_spinner="Initializing Knowledge Base...")
def initialize_system(model_name: str):
    """
    Handles the entire system initialization: GDrive sync, DB setup, model loading,
    and initial embedding of synced notes. This is the single source of truth for all
    core components and runs only once.
    """
    # 1. Sync with Google Drive
    drive = gds.authenticate_gdrive()
    gds.sync_directory_from_drive(drive, "notes")
    gds.sync_directory_from_drive(drive, "product_embeddings_v2")

    # 2. Initialize SQLite Database
    db.init_advanced_notes_database()

    # 3. Load the specified embedding model
    model = emb.load_embedding_model(model_name)
    if not model:
        raise Exception(f"Fatal: Could not load the embedding model '{model_name}'.")
    
    embedding_dim = model.get_sentence_embedding_dimension()

    # 4. Create model-namespaced ChromaDB collections
    model_name_safe = model_name.replace('/', '__')
    notes_collection_name = f"user_notes_local__{model_name_safe}"
    products_collection_name = f"analog_products_local__{model_name_safe}"
    
    notes_collection = vdb.get_notes_chroma_collection(notes_collection_name, embedding_dim)
    products_collection = vdb.get_local_chroma_collection(products_collection_name, embedding_dim)

    # 5. Ingest product data into its collection
    vdb.ingest_local_embeddings(products_collection)

    # 6. Scan synced notes and process the embedding queue immediately
    db.scan_and_queue_new_notes()
    processed_count = emb.process_embedding_queue(model, model_name, notes_collection)
    if processed_count > 0:
        # Use st.toast for a temporary message that doesn't require a rerun
        st.toast(f"✅ Automatically processed {processed_count} synced notes.")

    return {
        "drive_instance": drive,
        "embedding_model": model,
        "notes_collection": notes_collection,
        "products_collection": products_collection
    }

# --- Application Entry Point ---
# Set the desired model name (can be changed later in the UI)
if 'embedding_model_name' not in st.session_state:
    st.session_state.embedding_model_name = "BAAI/bge-small-en-v1.5"

try:
    # Initialize the system and get all required components.
    # This will run once and the result will be cached.
    system_resources = initialize_system(st.session_state.embedding_model_name)
    drive_instance = system_resources["drive_instance"]
    embedding_model = system_resources["embedding_model"]
    notes_collection = system_resources["notes_collection"]
    products_collection = system_resources["products_collection"]
except Exception as e:
    st.error(f"❌ A critical error occurred during startup: {e}")
    st.stop()


# --- Main Application Logic ---
def main():
    """Main function to run the Streamlit App."""
    
    st.sidebar.success("🎉 Phase 3+: Complete Knowledge Management!")
    try:
        st.sidebar.image("logo.png", width=200)
    except Exception:
        pass

    st.sidebar.header("🤖 AI Model Selection")
    selected_model = st.sidebar.selectbox("Choose AI Model:", ["gpt-4o-mini", "gpt-4o"], index=0)
    
    st.sidebar.header("📊 Embedding Model")
    # This selectbox now only serves to update the session state, which will trigger a full rerun
    # with the new model if changed. The key links it to our single source of truth.
    st.sidebar.selectbox(
        "Choose Embedding Model:",
        ["BAAI/bge-small-en-v1.5", "all-MiniLM-L6-v2"],
        key="embedding_model_name"
    )
    
    st.sidebar.header("⚡ Enhanced Features")
    enable_streaming = st.sidebar.checkbox("🔄 Enable Streaming Output", value=True)
    enable_context = st.sidebar.checkbox("🧠 Enable Context Awareness", value=True)
    max_context_messages = st.sidebar.slider("📝 Context History Length", 1, 10, 5)
    st.sidebar.header("🔍 Advanced Search")
    enable_unified_search = st.sidebar.checkbox("🔗 Unified Search (Products + Notes)", value=True)
    note_context_weight = st.sidebar.slider("📝 Note Context Weight", 0.0, 1.0, 0.3)

    if 'conversation_history' not in st.session_state: st.session_state.conversation_history = []
    if 'show_note_manager' not in st.session_state: st.session_state.show_note_manager = False

    # --- Sidebar: Notes UI ---
    st.sidebar.header("📝 Advanced Notes")
    stats = db.get_advanced_notes_stats()
    if stats:
        col1, col2 = st.sidebar.columns(2)
        col1.metric("Total Notes", stats.get("total_notes", 0))
        # This now accurately reflects the state after the initial processing run
        col2.metric("Need Embedding", stats.get("pending_jobs", 0))
        if stats.get("failed_jobs", 0) > 0:
            st.sidebar.metric("❌ Failed Jobs", stats.get("failed_jobs", 0))

    with st.sidebar.expander("✨ Create Advanced Note"):
        note_type = st.selectbox("Category:", list(db.NOTE_CATEGORIES.keys()), format_func=lambda x: f"{db.NOTE_CATEGORIES[x]['emoji']} {x.replace('_', ' ').title()}")
        note_title = st.text_input("Title:", placeholder="Enter note title...")
        note_content = st.text_area("Content:", placeholder="Write your note here...", height=100)
        note_links = st.text_input("Links:", placeholder="https://...")
        note_tags = st.text_input("Tags:", placeholder="tag1, tag2")
        if st.button("💾 Save Advanced Note"):
            if note_title and note_content:
                note_id, _ = db.save_advanced_note(note_type, note_title, note_content, note_links, note_tags)
                st.toast(f"Note saved! ID: {note_id}. Processing...")
                
                processed_count = emb.process_embedding_queue(embedding_model, st.session_state.embedding_model_name, notes_collection)
                if processed_count > 0:
                    st.toast("✅ Embeddings generated for new note!")
                
                gds.sync_directory_to_drive(drive_instance, "notes")
                st.rerun()
            else:
                st.warning("Please provide a title and content.")

    with st.sidebar.expander("🗂️ Manage All Notes"):
        if st.button("📋 Show All Notes Manager"):
            st.session_state.show_note_manager = True
            st.rerun()

    # --- Main Page UI ---
    if st.session_state.show_note_manager:
        st.header("🗂️ Comprehensive Note Manager")
        if st.button("❌ Close Note Manager"):
            st.session_state.show_note_manager = False
            st.rerun()

        all_notes = db.get_all_notes_with_details()
        if not all_notes:
            st.info("No notes found.")
        else:
            for note in all_notes:
                with st.expander(f"{db.NOTE_CATEGORIES.get(note['type'], {}).get('emoji', '📝')} {note['title']}"):
                    st.write(f"**Type:** {note['type'].replace('_', ' ').title()}")
                    st.write(f"**Last Modified:** {note['last_modified'][:19]}")
                    st.write(f"**Embedding Status:** {'✅ Ready' if note['has_embedding'] else '⏳ Pending'}")

                    if st.button("🗑️ Delete Note", key=f"delete_{note['id']}"):
                        db.delete_note(note['id'])
                        vdb.delete_note_embedding([note['id']], notes_collection)
                        gds.sync_directory_to_drive(drive_instance, "notes")
                        st.success(f"Note '{note['title']}' deleted.")
                        st.rerun()

    st.header("🔍 Unified Search (Products & Notes)")
    query_text = st.text_input("Enter your search query:", placeholder="Search across datasheets and your personal notes...")

    if st.button("🚀 Unified Search", type="primary"):
        if query_text and embedding_model:
            with st.spinner("Performing unified search..."):
                results = search.unified_search(
                    query_text, embedding_model, products_collection, notes_collection, 
                    note_context_weight, n_results=20, include_notes=enable_unified_search
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

if __name__ == "__main__":
    main()

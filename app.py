import streamlit as st
import streamlit.components.v1 as components
import pysqlite3
import sys
sys.modules["sqlite3"] = pysqlite3

from database import migrate_database, get_all_notes_with_details
# Import the rest of your modules here...
# from vector_db import ...
# from embeddings import ...

# ---- Run schema migration before anything else ----
migrate_database()

# ---- Helper UI functions ----
def display_notes_table(limit: int = 200):
    rows = get_all_notes_with_details(limit=limit)
    if not rows:
        st.info("No notes found.")
        return
    for row in rows:
        st.write(row)

# ---- Main app ----
st.title("Mind Map Notes")
st.markdown("---")
st.subheader("All notes (recent)")
display_notes_table(limit=200)


# utils.py

import streamlit as st
import time
from typing import List, Dict, Any

# --- Conversation and State Management ---

def add_to_conversation(role: str, content: str):
    """
    Adds a message to the conversation history stored in Streamlit's session state.
    It also ensures the history does not exceed a predefined maximum length.

    Args:
        role (str): The role of the message sender ('user' or 'assistant').
        content (str): The content of the message.
    """
    # max_context_messages should be retrieved from session_state or a config,
    # but we can default it here for modularity.
    max_messages = st.session_state.get('max_context_messages', 5)

    st.session_state.conversation_history.append({
        "role": role,
        "content": content,
        "timestamp": time.time()
    })
    
    # Prune the history to maintain the desired context length
    if len(st.session_state.conversation_history) > max_messages * 2: # user + assistant messages
        st.session_state.conversation_history = st.session_state.conversation_history[-(max_messages * 2):]

def clear_conversation_history():
    """Clears the conversation history from the session state."""
    st.session_state.conversation_history = []
    st.success("🧹 Conversation history cleared!")

# --- UI Display Helpers ---

def extract_and_display_unified_results(unified_results: Dict[str, Any]):
    """
    Extracts and displays unified search results (products + notes) in a formatted way.

    Args:
        unified_results (Dict[str, Any]): The combined results from the search function.
    """
    if not unified_results.get("combined"):
        st.info("No source documents were found for this query.")
        return

    st.write("### 📄 Source Documents (Products & Notes)")
    
    for idx, result in enumerate(unified_results["combined"], start=1):
        source_emoji = "🏭" if result["source"] == "product" else "📝"
        source_text = "Product Document" if result["source"] == "product" else "Personal Note"
        relevance = result.get("relevance_score", 0.0)
        
        # Determine title or document identifier
        if result["source"] == "product":
            title = result["metadata"].get("product", "Unknown Product")
        else: # 'note'
            title = result["metadata"].get("title", "Untitled Note")

        with st.expander(f"{source_emoji} **{title}** | Source: {source_text} | Relevance: {relevance:.2f}"):
            st.markdown(f"**Content Snippet:**\n> {result['content'][:500]}...")
            st.markdown(f"---")
            
            if result["source"] == "product":
                st.markdown(f"**Product:** `{result['metadata'].get('product', 'N/A')}`")
                st.markdown(f"**Chunk Index:** `{result['metadata'].get('chunk_index', 'N/A')}`")
                links = result["metadata"].get("links")
                if links and links != "No links provided":
                    st.markdown(f"**Links:** {links}")
            else: # 'note'
                st.markdown(f"**Note Type:** `{result['metadata'].get('note_type', 'N/A')}`")
                tags = result["metadata"].get("tags")
                if tags:
                    st.markdown(f"**Tags:** `{tags}`")

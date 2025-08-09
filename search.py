# search.py

import streamlit as st
from sentence_transformers import SentenceTransformer
import vector_db  # Import the module for ChromaDB interactions
from typing import List, Dict, Any, Optional

def unified_search(
    query_text: str,
    embedding_model_instance: SentenceTransformer,
    products_collection,
    notes_collection,
    note_context_weight: float = 0.3,
    n_results: int = 10,
    include_notes: bool = True
) -> Dict[str, Any]:
    """
    Searches across both product and note collections, combining and ranking the results.

    Args:
        query_text (str): The user's search query.
        embedding_model_instance (SentenceTransformer): The loaded embedding model.
        products_collection (chromadb.Collection): The ChromaDB collection for products.
        notes_collection (chromadb.Collection): The ChromaDB collection for notes.
        note_context_weight (float): The weight to apply to the relevance of note results.
        n_results (int): The number of results to fetch from each collection.
        include_notes (bool): Flag to include notes in the search.

    Returns:
        Dict[str, Any]: A dictionary containing raw and combined search results.
    """
    if not embedding_model_instance:
        return {"products": [], "notes": [], "combined": [], "error": "Embedding model not available"}

    # 1. Generate Query Embedding
    try:
        query_embedding = embedding_model_instance.encode(query_text, normalize_embeddings=True).tolist()
    except Exception as e:
        return {"products": [], "notes": [], "combined": [], "error": f"Failed to generate query embedding: {e}"}
    
    # 2. Search Product Collection
    product_results = {}
    try:
        product_results = products_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        st.error(f"Error querying product collection: {e}")

    # 3. Search Notes Collection
    note_results = {}
    if include_notes:
        try:
            notes_count = notes_collection.count()
            if notes_count > 0:
                note_results = notes_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(n_results, notes_count),
                    include=["documents", "metadatas", "distances"]
                )
        except Exception as e:
            st.error(f"Error querying notes collection: {e}")

    # 4. Combine and Rank Results
    combined_results = []
    
    # Process product results
    if product_results.get("documents") and product_results["documents"][0]:
        for doc, meta, dist in zip(product_results["documents"][0], product_results["metadatas"][0], product_results["distances"][0]):
            combined_results.append({
                "content": doc, "metadata": meta, "distance": dist,
                "source": "product", "relevance_score": 1.0 - dist
            })

    # Process note results
    if note_results.get("documents") and note_results["documents"][0]:
        for doc, meta, dist in zip(note_results["documents"][0], note_results["metadatas"][0], note_results["distances"][0]):
            combined_results.append({
                "content": doc, "metadata": meta, "distance": dist,
                "source": "note", "relevance_score": (1.0 - dist) * (1 + note_context_weight) # Boost note relevance
            })

    # Sort combined results by the final relevance score
    combined_results.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    return {
        "products": product_results,
        "notes": note_results,
        "combined": combined_results[:n_results * 2] # Return a generous number of combined results
    }

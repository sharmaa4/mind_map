# vector_db.py

import streamlit as st
import chromadb
from chromadb.config import Settings
import os
import glob
import json
from typing import Optional

# --- ChromaDB Client and Collection Management ---

@st.cache_resource(show_spinner=False)
def get_chroma_client(persist_directory: str = "./db_local/") -> chromadb.PersistentClient:
    """
    Creates and returns a persistent ChromaDB client.

    Args:
        persist_directory (str): The directory to store the database files.

    Returns:
        chromadb.PersistentClient: An instance of the ChromaDB client.
    """
    os.makedirs(persist_directory, exist_ok=True)
    client = chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings()
    )
    return client

@st.cache_resource(show_spinner=False)
def get_local_chroma_collection(collection_name: str, embedding_dimension: int) -> Optional[chromadb.Collection]:
    """
    Gets or creates a ChromaDB collection for product embeddings.

    Args:
        collection_name (str): The name of the collection.
        embedding_dimension (int): The dimension of the embeddings to be stored.

    Returns:
        Optional[chromadb.Collection]: The ChromaDB collection object, or None if an error occurs.
    """
    try:
        client = get_chroma_client()
        collection = client.get_or_create_collection(
            name=f"{collection_name}_{embedding_dimension}d",
            metadata={"embedding_dimension": embedding_dimension}
        )
        print(f"INFO: Using local product collection: {collection.name}")
        return collection
    except Exception as e:
        st.error(f"Error getting Chroma product collection: {e}")
        return None

@st.cache_resource(show_spinner=False)
def get_notes_chroma_collection(collection_name: str, embedding_dimension: int) -> Optional[chromadb.Collection]:
    """
    Gets or creates a ChromaDB collection specifically for user notes.

    Args:
        collection_name (str): The name of the collection for notes.
        embedding_dimension (int): The dimension of the embeddings.

    Returns:
        Optional[chromadb.Collection]: The ChromaDB collection object, or None if an error occurs.
    """
    try:
        client = get_chroma_client()
        collection = client.get_or_create_collection(
            name=f"{collection_name}_{embedding_dimension}d",
            metadata={"embedding_dimension": embedding_dimension}
        )
        print(f"INFO: Using notes collection: {collection.name}")
        return collection
    except Exception as e:
        st.error(f"Error getting notes collection: {e}")
        return None

# --- Data Ingestion Logic ---

def ingest_local_embeddings(collection: chromadb.Collection, embeddings_folder: str = "product_embeddings_v2"):
    """
    Ingests pre-computed product embeddings from JSON files into a ChromaDB collection.

    Args:
        collection (chromadb.Collection): The collection to ingest data into.
        embeddings_folder (str): The folder containing the JSON embedding files.
    """
    if not os.path.exists(embeddings_folder):
        st.warning(f"Embeddings folder '{embeddings_folder}' not found.")
        return

    embedding_files = glob.glob(os.path.join(embeddings_folder, "*_embeddings_v2.json"))

    if not embedding_files:
        st.warning(f"No embedding files found in '{embeddings_folder}'. Run embedding generation script first.")
        return

    st.write(f"📁 Found {len(embedding_files)} local embedding files to process.")
    
    total_docs_ingested = 0
    with st.spinner("Ingesting local embeddings into ChromaDB..."):
        for file_path in embedding_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                product = data.get("product", "unknown")
                embeddings = data.get("embeddings", [])
                chunks = data.get("chunks", [])
                links = data.get("links", [])
                
                if not embeddings or not chunks:
                    continue

                documents = [f"{product} - chunk {i}: {chunk[:250]}..." for i, chunk in enumerate(chunks)]
                ids = [f"{product}_local_chunk_{i}" for i in range(len(embeddings))]
                metadatas = [{
                    "product": product,
                    "chunk_index": i,
                    "model": data.get("model_info", {}).get("name", "unknown"),
                    "links": ", ".join(links) if isinstance(links, list) else str(links)
                } for i in range(len(embeddings))]

                # Ingest in batches to avoid overwhelming the database
                batch_size = 100
                for i in range(0, len(embeddings), batch_size):
                    collection.add(
                        ids=ids[i:i+batch_size],
                        embeddings=embeddings[i:i+batch_size],
                        documents=documents[i:i+batch_size],
                        metadatas=metadatas[i:i+batch_size]
                    )
                
                total_docs_ingested += len(embeddings)
                
            except Exception as e:
                st.error(f"Error processing file {file_path}: {e}")

    if total_docs_ingested > 0:
        st.success(f"✅ Ingested {total_docs_ingested} documents from local embeddings!")
    else:
        st.info("No new documents were ingested.")

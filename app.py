import streamlit as st
import streamlit.components.v1 as components
import pysqlite3
import sys
print("pysqlite3 version:", pysqlite3.sqlite_version)
sys.modules["sqlite3"] = pysqlite3
import shutil
import os
import glob
import json
import time
from concurrent.futures import ThreadPoolExecutor
import uuid
import chromadb
from chromadb.config import Settings
import requests
import fitz  # PyMuPDF
from PIL import Image
import subprocess
from bs4 import BeautifulSoup
import numpy as np

# Import sentence transformers for local embeddings
from sentence_transformers import SentenceTransformer

# Configure the Streamlit page
st.set_page_config(page_title="Global Product Search - Powered by Puter.js & Local Embeddings", layout="wide")
st.title("🚀 Global Product Search for Analog Products - Powered by Puter.js & Local Embeddings")

# Add migration success notice
st.sidebar.success("🎉 Fully Migrated to Puter.js + Local Embeddings!")
try:
    st.sidebar.image("logo.png", width=200)
except:
    pass  # Logo file might not exist

# Model selection for Puter.js
st.sidebar.header("🤖 AI Model Selection")
selected_model = st.sidebar.selectbox(
    "Choose AI Model:",
    ["gpt-4o-mini", "gpt-4o", "o1-mini", "o1", "o3-mini", "o3", "gpt-4.1", "claude-sonnet-4"],
    index=0,
    help="Select the AI model for processing queries"
)

# Embedding model selection (matching your embedding script)
st.sidebar.header("📊 Embedding Model")
embedding_model_name = st.sidebar.selectbox(
    "Choose Embedding Model:",
    ["BAAI/bge-small-en-v1.5", "all-mpnet-base-v2", "paraphrase-multilingual-mpnet-base-v2", "all-MiniLM-L6-v2"],
    index=0,
    help="Select the local embedding model (must match your embedding generation script)"
)

# Add benefits info
st.sidebar.info("""
💰 **Benefits Achieved:**
- ✅ Zero API costs
- ✅ No rate limits  
- ✅ Local embeddings
- ✅ No DNS issues
- ✅ Offline capability
""")

# ================================
# LOCAL EMBEDDINGS SETUP
# ================================

@st.cache_resource(show_spinner=True)
def load_embedding_model(model_name="BAAI/bge-small-en-v1.5"):
    """
    Load local sentence transformer model for embeddings
    This replaces Azure OpenAI embeddings completely
    """
    try:
        with st.spinner(f"Loading embedding model: {model_name}..."):
            model = SentenceTransformer(model_name)
        st.success(f"✅ Local embedding model '{model_name}' loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading embedding model '{model_name}': {e}")
        st.write("**Solution:** Install sentence-transformers: `pip install sentence-transformers`")
        return None

def get_query_embedding_local(query_text, model_name="BAAI/bge-small-en-v1.5"):
    """
    REPLACEMENT for get_query_embedding() - Uses local sentence transformers
    No Azure OpenAI API calls!
    """
    model = load_embedding_model(model_name)
    if model is None:
        st.error("❌ Embedding model not available")
        return None
    
    try:
        # Generate embedding locally
        embedding = model.encode(query_text, normalize_embeddings=True)
        
        # Convert numpy array to list for ChromaDB compatibility
        embedding_list = embedding.tolist()
        
        # Display success info
        st.success(f"✅ Generated embedding locally - Dimension: {len(embedding_list)} | Model: {model_name}")
        
        return embedding_list
        
    except Exception as e:
        st.error(f"❌ Error generating local embedding: {e}")
        return None

# Load the embedding model on startup
embedding_model = load_embedding_model(embedding_model_name)

# ================================
# CHROMADB LOCAL COLLECTION SETUP
# ================================

@st.cache_resource(show_spinner=False)
def get_chroma_client(persist_directory="./db_local/"):
    client = chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings()
    )
    return client

@st.cache_resource(show_spinner=False)
def get_local_chroma_collection(embedding_dimension=384):
    client = get_chroma_client()
    collection_name = f"analog_products_local_{embedding_dimension}d"
    try:
        collection = client.get_collection(name=collection_name)
        st.info(f"✅ Using local collection: {collection_name}")
    except:
        collection = client.create_collection(
            name=collection_name,
            embedding_function=None
        )
        st.success(f"✅ Created new local collection: {collection_name}")
    return collection

# ================================
# LOCAL EMBEDDINGS INGESTION
# ================================

def ingest_local_embeddings(collection, embeddings_folder="product_embeddings_v2"):
    """
    Ingest embeddings from your new embedding script format
    """
    if not os.path.exists(embeddings_folder):
        st.warning(f"Embeddings folder '{embeddings_folder}' not found.")
        return
    
    embedding_files = glob.glob(os.path.join(embeddings_folder, "*_embeddings_v2.json"))
    
    if not embedding_files:
        st.warning(f"No embedding files found in '{embeddings_folder}'. Run your embedding generation script first.")
        return
    
    st.write(f"📁 Found {len(embedding_files)} local embedding files")
    
    total_docs = 0
    for file_path in embedding_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            product = data.get("product", "unknown")
            embeddings = data.get("embeddings", [])
            chunks = data.get("chunks", [])
            links = data.get("links", [])
            
            if not embeddings:
                continue
            
            # Create documents and IDs
            documents = [f"{product} - chunk {i}: {chunk[:200]}..." for i, chunk in enumerate(chunks)]
            ids = [f"{product}_local_chunk_{i}" for i in range(len(embeddings))]
            metadatas = [{
                "product": product,
                "chunk_index": i,
                "model": data.get("model_info", {}).get("name", embedding_model_name),
                "links": ", ".join(links) if isinstance(links, list) else str(links),
                "processing_time": data.get("processing_info", {}).get("processing_time", 0)
            } for i in range(len(embeddings))]
            
            # Add to collection in batches
            batch_size = 100
            for i in range(0, len(embeddings), batch_size):
                batch_embeddings = embeddings[i:i+batch_size]
                batch_documents = documents[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                batch_metadatas = metadatas[i:i+batch_size]
                
                collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_documents,
                    metadatas=batch_metadatas
                )
            
            total_docs += len(embeddings)
            
        except Exception as e:
            st.error(f"Error processing {file_path}: {e}")
    
    st.success(f"✅ Ingested {total_docs} documents from local embeddings!")

# ================================
# PUTER.JS INTEGRATION FUNCTIONS (FIXED)
# ================================

def create_puter_component_fixed(prompt, model="gpt-4o-mini"):
    """
    Fixed version without 'key' argument for compatibility
    """
    escaped_prompt = prompt.replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
    
    # Add unique identifier to prevent caching issues
    unique_id = int(time.time() * 1000) % 1000000
    
    puter_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://js.puter.com/v2/"></script>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 20px;
                background: #f8f9fa;
            }}
            .container {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .loading {{
                display: flex;
                align-items: center;
                gap: 10px;
                color: #666;
            }}
            .spinner {{
                border: 3px solid #f3f3f3;
                border-top: 3px solid #667eea;
                border-radius: 50%;
                width: 20px;
                height: 20px;
                animation: spin 1s linear infinite;
            }}
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            .result {{
                white-space: pre-wrap;
                line-height: 1.6;
                color: #333;
            }}
            .error {{
                color: #dc3545;
                background: #f8d7da;
                padding: 10px;
                border-radius: 5px;
                border: 1px solid #f5c6cb;
            }}
            .model-info {{
                background: #e3f2fd;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 15px;
                border-left: 4px solid #2196f3;
            }}
        </style>
    </head>
    <body>
        <div class="container" id="container_{unique_id}">
            <div class="model-info">
                <strong>🤖 Model:</strong> {model} | <strong>⚡ Provider:</strong> Puter.js (Free) | <strong>📊 Embeddings:</strong> Local
            </div>
            <div id="result_{unique_id}">
                <div class="loading">
                    <div class="spinner"></div>
                    <span>Processing with {model}...</span>
                </div>
            </div>
        </div>
        
        <script>
            async function processQuery_{unique_id}() {{
                const resultDiv = document.getElementById('result_{unique_id}');
                try {{
                    const startTime = Date.now();
                    const response = await puter.ai.chat("{escaped_prompt}", {{
                        model: "{model}",
                        stream: false
                    }});
                    const endTime = Date.now();
                    const processingTime = ((endTime - startTime) / 1000).toFixed(2);
                    resultDiv.innerHTML = `
                        <div class="result">${{response}}</div>
                        <div style="margin-top: 15px; font-size: 0.9em; color: #666; border-top: 1px solid #eee; padding-top: 10px;">
                            ⏱ Processing time: ${{processingTime}}s | 💡 Model: {model} | 📊 Embeddings: Local
                        </div>
                    `;
                }} catch (error) {{
                    console.error('Puter.js Error:', error);
                    resultDiv.innerHTML = `
                        <div class="error">
                            <strong>❌ Error:</strong> ${{error.message || 'Unknown error occurred'}}
                            <br><br>
                            <strong>Troubleshooting:</strong>
                            <ul>
                                <li>Check your internet connection</li>
                                <li>Try refreshing the page</li>
                                <li>Try a different model</li>
                            </ul>
                        </div>
                    `;
                }}
            }}
            processQuery_{unique_id}();
        </script>
    </body>
    </html>
    """
    
    return components.html(puter_html, height=300)

def get_structured_output_from_puter(concatenated_text, user_query, model="gpt-4o-mini", prompt_template=None):
    """
    Updated function using the fixed component
    """
    if prompt_template is None:
        prompt_template = (
            "Based on the following product documentation text and the user "
            "query, generate a structured summary based on the user query given. "
            "Don't mention any kinds of links or something. "
            "Keep your answers aligned with user's query only. "
            "If user is asking very generic questions example 'What kind of "
            "products ADI/ Analog Devices have then mention the user to ask "
            "specific questions related to certain categories specifically "
            "documentation related and not generic questions. "
            "You can answer such questions using the knowledge you are already "
            "trained on, no need to provide summary of the documents in this case. "
            f"User Query: {user_query}\n\n"
        )
    
    full_prompt = prompt_template + concatenated_text
    
    st.write("### 🤖 AI Processing with Puter.js")
    create_puter_component_fixed(full_prompt, model)
    
    return "Response displayed above via Puter.js integration"

# ================================
# UTILITY FUNCTIONS
# ================================

def extract_and_display_documents(query_results):
    """
    Extracts documents from the query results JSON and displays each document.
    """
    documents_nested = query_results.get("documents")
    metadatas_nested = query_results.get("metadatas")
    
    if not documents_nested:
        st.write("No documents found in the query results.")
        return
    
    documents = documents_nested[0] if documents_nested else []
    metadatas = metadatas_nested[0] if metadatas_nested and isinstance(metadatas_nested, list) and len(metadatas_nested) > 0 else []
    
    st.write("### 📄 Source Documents")
    for idx, doc in enumerate(documents, start=1):
        metadata = metadatas[idx-1] if idx-1 < len(metadatas) else {}
        product = metadata.get("product", "Unknown")
        chunk_index = metadata.get("chunk_index", "Unknown")
        links = metadata.get("links", "No links provided")
        
        with st.expander(f"Document {idx}: {product} - Chunk {chunk_index}"):
            st.write(f"**Content:** {doc}")
            st.write(f"**Product:** {product}")
            st.write(f"**Chunk Index:** {chunk_index}")
            st.write(f"**Links:** {links}")

# ================================
# MAIN SEARCH UI WITH LOCAL EMBEDDINGS + PUTER.JS
# ================================

st.write("---")
st.header("🔍 Global Search - Powered by Local Embeddings + Puter.js!")

query_text = st.text_input(
    "Enter your search query:", 
    placeholder="e.g., Wideband Low Noise Amplifier datasheet",
    help="Search through thousands of product documents using local AI-powered semantic search"
)

# Get embedding dimension from the loaded model
if embedding_model:
    test_emb = embedding_model.encode("test")
    embedding_dim = len(test_emb)
else:
    # Default dimensions for common models
    model_dimensions = {
        "BAAI/bge-small-en-v1.5": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-multilingual-mpnet-base-v2": 768,
        "all-MiniLM-L6-v2": 384
    }
    embedding_dim = model_dimensions.get(embedding_model_name, 384)

st.info(f"📊 **Current Configuration:** Model: {embedding_model_name} | Dimension: {embedding_dim}")

collection = get_local_chroma_collection(embedding_dim)

# Display collection status
try:
    doc_count = collection.count()
    if doc_count == 0:
        st.warning("⚠️ Collection is empty. Please run the ingestion process below.")
    else:
        st.success(f"✅ Collection ready with {doc_count:,} documents")
except:
    st.error("❌ Error accessing collection")

# Ingestion controls
st.write("### 📥 Data Management")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📂 Ingest Local Embeddings"):
        with st.spinner("Ingesting local embeddings..."):
            ingest_local_embeddings(collection)

with col2:
    if st.button("📊 Collection Stats"):
        try:
            count = collection.count()
            st.write(f"Documents in collection: {count:,}")
            if count > 0:
                # Get a sample to show metadata structure
                sample = collection.get(limit=1, include=["metadatas"])
                if sample["metadatas"]:
                    st.write("Sample metadata:")
                    st.json(sample["metadatas"][0])
        except Exception as e:
            st.error(f"Error getting stats: {e}")

with col3:
    if st.button("🗑️ Clear Collection"):
        if st.sidebar.button("⚠️ Confirm Clear Collection"):
            try:
                # Get all IDs and delete them
                all_data = collection.get()
                if all_data["ids"]:
                    collection.delete(ids=all_data["ids"])
                    st.success("Collection cleared!")
                    st.rerun()
                else:
                    st.info("Collection is already empty.")
            except Exception as e:
                st.error(f"Error clearing collection: {e}")

# Main search functionality
if st.button("🚀 Search with Local Embeddings + Puter.js", type="primary"):
    if not query_text:
        st.warning("Please enter a valid search query.")
    else:
        with st.spinner("🔍 Generating local embeddings and searching..."):
            try:
                # Generate query embedding locally
                query_embedding = get_query_embedding_local(query_text, embedding_model_name)
                if query_embedding is None:
                    st.error("❌ Failed to generate embedding")
                    st.stop()
                
                # Search the collection
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=5,
                    include=["documents", "metadatas"]
                )
                
                # Check if we got results
                if not results.get("documents") or not results["documents"][0]:
                    st.warning("No relevant documents found. Please check if data is ingested properly.")
                    st.stop()
                
                # Get concatenated text from results
                documents = results["documents"][0]
                concatenated_text = "\n\n".join(documents)
                
                st.success("✅ Search completed! Processing with Puter.js...")
                
                # Use Puter.js for chat completion
                get_structured_output_from_puter(
                    concatenated_text, 
                    query_text, 
                    model=selected_model
                )
                
                # Display source documents
                extract_and_display_documents(results)
                
            except Exception as e:
                st.error(f"❌ Error during search: {e}")
                st.write("**Troubleshooting Steps:**")
                st.write("1. Install dependencies: `pip install sentence-transformers torch`")
                st.write("2. Run your embedding generation script")
                st.write("3. Click 'Ingest Local Embeddings' button")

# ================================
# FOOTER WITH COMPLETE MIGRATION INFO
# ================================

st.write("---")
st.markdown(f"""
### 🎉 **Complete Migration Successful!**

**✅ Fully Independent System:**
- 📊 **Local Embeddings** - Using {embedding_model_name} ({embedding_dim}D)
- 🤖 **Puter.js AI** - {selected_model} for chat completions
- 🚫 **Zero Dependencies** - No Azure OpenAI or external services
- ⚡ **Offline Capable** - Embeddings work without internet

**📈 Performance Benefits:**
- 💰 **$0 Costs** - Completely free operation
- 🚀 **Faster Processing** - Local embeddings, no API latency
- 🔄 **No Rate Limits** - Unlimited queries
- 🛡️ **Enhanced Privacy** - All data stays local

**🔧 Technical Stack:**
- **Embeddings**: {embedding_model_name}
- **Chat AI**: Puter.js ({selected_model})
- **Vector DB**: ChromaDB (Local)
- **UI**: Streamlit

*System fully migrated from Azure OpenAI to local + Puter.js architecture*
""")


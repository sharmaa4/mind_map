import pysqlite3
import sys
print("pysqlite3 version:", pysqlite3.sqlite_version)  # Verify version is >= 3.35.0
sys.modules["sqlite3"] = pysqlite3

import chromadb
from chromadb.config import Settings

import os
import glob
import time
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import fitz  # PyMuPDF for PDF text extraction
import tiktoken  # For token counting

# Set environment variables
os.environ["CHROMADB_DISABLE_TELEMETRY"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -----------------------------
# API Configuration
# -----------------------------
# Text Embedding API (e.g., Azure OpenAI)
AZURE_OPENAI_ENDPOINT = "https://engassist-eus-dev-aais.openai.azure.com/openai/deployments/hackathon-emb-emb3l-team-21-cgcwn/embeddings?api-version=2023-05-15"
AZURE_OPENAI_API_KEY = "de7a14848db7462c9783adbcfbb0b430"

# Chat API for Query Enrichment (unused in this pipeline)
CHAT_API_ENDPOINT = "https://engassist-eus-dev-aais.openai.azure.com/openai/deployments/hackathon-gpt4omini-team-21-cgcwn/chat/completions?api-version=2024-10-21"
CHAT_API_KEY = "de7a14848db7462c9783adbcfbb0b430"

# -----------------------------
# Pipeline Parameters
# -----------------------------
MAX_PDF_PAGES = 50       # Skip PDFs with more than 50 pages.
MAX_PDFS = 50            # Process up to 50 PDFs.
BATCH_SIZE = 16          # Number of text chunks to process per batch.
THREAD_WORKERS = 40       # Number of threads for batch inference.
REFERENCES_FOLDER = "./references"  # Folder to store dumped JSON embeddings

# Global variable to hold aggregated TOC (for documentation purposes)
global_toc = ""

# -----------------------------
# RateLimiter Implementation
# -----------------------------
import threading
class RateLimiter:
    def __init__(self, max_requests_per_minute, max_tokens_per_minute):
        self.max_requests = max_requests_per_minute
        self.max_tokens = max_tokens_per_minute
        self.lock = threading.Lock()
        self.reset_time = time.time() + 60
        self.requests_made = 0
        self.tokens_used = 0

    def wait_if_needed(self, tokens_in_chunk):
        with self.lock:
            current_time = time.time()
            if current_time >= self.reset_time:
                self.reset_time = current_time + 60
                self.requests_made = 0
                self.tokens_used = 0
            if self.requests_made + 1 > self.max_requests or self.tokens_used + tokens_in_chunk > self.max_tokens:
                sleep_time = self.reset_time - current_time
                print(f"Rate limit reached (reqs: {self.requests_made}, tokens: {self.tokens_used}). Sleeping for {sleep_time:.2f} sec.")
                time.sleep(sleep_time)
                self.reset_time = time.time() + 60
                self.requests_made = 0
                self.tokens_used = 0
            self.requests_made += 1
            self.tokens_used += tokens_in_chunk

rate_limiter = RateLimiter(1200, 200000)

import os
import shutil

def clear_directory(directory):
    """
    Deletes the given directory and its contents if it exists, then recreates it.
    
    Args:
        directory (str): The path to the directory to clear.
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)
        print(f"Cleared directory: {directory}")
    os.makedirs(directory)
    print(f"Recreated directory: {directory}")

def clear_required_directories():
    """
    Clears all required directories before processing.
    Modify the list below to include any directories you want to clear.
    """
    # Update these paths as necessary. In your code, downloaded PDFs are saved in "downloaded_pdfs",
    # and your persistent ChromaDB is stored in "chroma_db" (and possibly "chroma_db_images").
    directories_to_clear = ["downloaded_pdfs", "chroma_db", "references"]
    for directory in directories_to_clear:
        clear_directory(directory)


# -----------------------------
# Token Counting using tiktoken
# -----------------------------
def count_tokens(text, model="text-embedding-ada-002"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

# -----------------------------
# OpenAI API Call for Text Embedding (Batch)
# -----------------------------
def openai_embed_batch(chunks, retries=3, backoff_factor=2):
    total_tokens = sum(count_tokens(chunk) for chunk in chunks)
    rate_limiter.wait_if_needed(total_tokens)
    
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY
    }
    payload = {"input": chunks}
    
    for attempt in range(retries):
        try:
            response = requests.post(AZURE_OPENAI_ENDPOINT, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                embeddings = [item["embedding"] for item in result["data"]]
                return embeddings
            else:
                print(f"Embedding API call failed (status {response.status_code}): {response.text}")
        except Exception as e:
            print(f"Exception during embedding API call: {e}")
        sleep_time = backoff_factor ** attempt
        print(f"Retrying after {sleep_time} seconds...")
        time.sleep(sleep_time)
    return [None] * len(chunks)

# -----------------------------
# Text Chunking Function
# -----------------------------
def chunk_text(text, max_length=2000, overlap=200):
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + max_length
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= text_length:
            break
        start = end - overlap
    return chunks

# -----------------------------
# PDF Text Extraction with Page-wise Chunking & TOC Extraction
# -----------------------------
def extract_toc(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        toc = doc.get_toc()  # Each entry: [level, title, page number]
        doc.close()
        return toc
    except Exception as e:
        print(f"Error extracting TOC from {pdf_path}: {e}")
        return []

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        pages_text = []
        for page in doc:
            pages_text.append(page.get_text())
        doc.close()
        return pages_text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return []

def process_pdf_text_with_toc(pdf_path, max_length=2000, overlap=200):
    pdf_name = os.path.basename(pdf_path)
    pages_text = extract_text_from_pdf(pdf_path)
    all_chunks = []
    for page_idx, page_text in enumerate(pages_text):
        chunks = chunk_text(page_text, max_length=max_length, overlap=overlap)
        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append((pdf_name, page_idx, chunk_idx, chunk))
    toc = extract_toc(pdf_path)
    return pdf_name, all_chunks, toc

def extract_texts_from_pdfs(pdf_folder, max_pdfs=50):
    all_chunks = []
    pdf_times = {}
    toc_dict = {}
    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))[:max_pdfs]
    print(f"Found {len(pdf_files)} PDF(s) for text extraction.", flush=True)
    total_time = 0.0
    for pdf_path in pdf_files:
        start = time.time()
        pdf_name, chunks, toc = process_pdf_text_with_toc(pdf_path)
        extraction_time = time.time() - start
        pdf_times[pdf_name] = extraction_time
        total_time += extraction_time
        print(f"{pdf_name}: {len(chunks)} chunk(s) extracted in {extraction_time:.2f} sec.", flush=True)
        all_chunks.extend(chunks)
        toc_dict[pdf_name] = toc
    print(f"Total text extraction time: {total_time:.2f} sec.", flush=True)
    return all_chunks, pdf_times, toc_dict

# -----------------------------
# Batch Inference Stage for Text Embeddings
# -----------------------------
def batch_data(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

def process_text_batch(batch_data):
    """
    Processes a batch of text chunks.
    batch_data: list of tuples (pdf_name, page_idx, chunk_idx, text_chunk)
    Returns a list of dicts with:
      - 'id': Unique ID (format: pdf_name_page_{page_idx}_chunk_{chunk_idx})
      - 'embedding': Embedding vector as list of floats (from OpenAI API)
      - 'metadata': { 'pdf_name': pdf_name, 'page_idx': page_idx, 'chunk_idx': chunk_idx }
      - 'document': ""
    """
    results = []
    try:
        texts = [item[3] for item in batch_data]
        meta_list = [(item[0], item[1], item[2]) for item in batch_data]
        embeddings = openai_embed_batch(texts)
        for i, emb in enumerate(embeddings):
            if not emb:
                continue
            pdf_name, page_idx, chunk_idx = meta_list[i]
            results.append({
                "id": f"{pdf_name}_page_{page_idx}_chunk_{chunk_idx}",
                "embedding": emb,
                "metadata": {"pdf_name": pdf_name, "page_idx": page_idx, "chunk_idx": chunk_idx},
                "document": ""
            })
    except Exception as e:
        print(f"Error in text batch processing: {e}", flush=True)
    return results

# -----------------------------
# Dump Text Embeddings JSON by PDF (including TOC)
# -----------------------------
def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def dump_text_embeddings_json_with_toc(all_embeddings, toc_dict, output_folder):
    ensure_folder(output_folder)
    grouped = {}
    for item in all_embeddings:
        pdf_name = item["metadata"]["pdf_name"]
        if pdf_name not in grouped:
            grouped[pdf_name] = []
        grouped[pdf_name].append(item)
    
    for pdf_name, embeddings in grouped.items():
        output_file = os.path.join(output_folder, f"{pdf_name}_text_embeddings.json")
        pdf_toc = toc_dict.get(pdf_name, [])
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump({
                    "pdf": pdf_name,
                    "toc": pdf_toc,
                    "embeddings": embeddings
                }, f)
            print(f"Dumped text embeddings and TOC for {pdf_name} to {output_file}", flush=True)
        except Exception as e:
            print(f"Error dumping JSON for {pdf_name}: {e}", flush=True)

# -----------------------------
# Main Pipeline
# -----------------------------
def main():
    overall_start = time.time()
    pdf_folder = "./downloaded_pdfs"  # Folder where your PDFs are stored

    # Extraction Stage: Extract text page-wise and TOC from PDFs.
    all_chunks, pdf_times, toc_dict = extract_texts_from_pdfs(pdf_folder, max_pdfs=50)
    print(f"Total text chunks collected for processing: {len(all_chunks)}", flush=True)
    
    # Aggregate TOCs and store in global variable for reference.
    global global_toc
    global_toc = "\n\n".join([f"{pdf} TOC:\n" + "\n".join([f"Level {entry[0]}: {entry[1]} (Page {entry[2]})" for entry in toc]) 
                               for pdf, toc in toc_dict.items() if toc])
    print("Aggregated TOC for all PDFs:\n", global_toc, flush=True)

    # Batch Inference Stage using ThreadPoolExecutor.
    inference_start = time.time()
    all_embeddings = []
    batches = list(batch_data(all_chunks, BATCH_SIZE))
    print(f"Processing {len(batches)} batches with batch size {BATCH_SIZE}...", flush=True)

    with ThreadPoolExecutor(max_workers=THREAD_WORKERS) as executor:
        futures = {executor.submit(process_text_batch, batch): batch for batch in batches}
        batch_counter = 0
        for future in as_completed(futures):
            batch_counter += 1
            try:
                batch_results = future.result()
                all_embeddings.extend(batch_results)
                print(f"Completed batch {batch_counter}/{len(batches)}: Processed {len(batch_results)} embeddings.", flush=True)
            except Exception as e:
                print(f"Error processing a batch: {e}", flush=True)
    inference_end = time.time()
    print(f"Total inference time (batch processing): {inference_end - inference_start:.2f} sec.", flush=True)

    # Report Per-PDF Extraction Times.
    print("\nPer-PDF Extraction Times:", flush=True)
    for pdf, t in pdf_times.items():
        print(f"  {pdf}: {t:.2f} sec", flush=True)
    print(f"Total extraction time: {sum(pdf_times.values()):.2f} sec.", flush=True)

    # -----------------------------
    # Insert Embeddings into Persistent ChromaDB.
    # -----------------------------
    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings()
    )
    collection_name = "pdf_embeddings"
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Using existing ChromaDB collection: {collection_name}", flush=True)
    except Exception as e:
        collection = client.create_collection(name=collection_name, embedding_function=None)
        print(f"Created new ChromaDB collection: {collection_name}", flush=True)

    if all_embeddings:
        ids = [item["id"] for item in all_embeddings]
        embeddings = [item["embedding"] for item in all_embeddings]
        metadatas = [item["metadata"] for item in all_embeddings]
        documents = [item["document"] for item in all_embeddings]
        print("Inserting embeddings into ChromaDB collection...", flush=True)
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        print(f"Upserted {len(ids)} embeddings into collection '{collection_name}'.", flush=True)
    else:
        print("No embeddings to upsert.", flush=True)

    # -----------------------------
    # Dump Embeddings to JSON Files in a Separate Folder (References)
    # -----------------------------
    ensure_folder("references")
    dump_text_embeddings_json_with_toc(all_embeddings, toc_dict, "references")

    overall_end = time.time()
    print(f"\nOverall pipeline time: {overall_end - overall_start:.2f} sec.", flush=True)

if __name__ == "__main__":
    #clear_required_directories()
    main()


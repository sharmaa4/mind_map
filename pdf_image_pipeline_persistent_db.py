import os
import glob
import time
import sys
import torch
import colpali_engine
import pysqlite3
import sys
print("pysqlite3 version:", pysqlite3.sqlite_version)  # Verify version is >= 3.35.0
sys.modules["sqlite3"] = pysqlite3

import os
import glob
import time
import sys
import torch
from pdf2image import convert_from_bytes
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import chromadb
from chromadb.config import Settings
import shutil

# Set environment variables early
os.environ["CHROMADB_DISABLE_TELEMETRY"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# CORRECTED IMPORT: The models are in the fireworks.models submodule for this version.
from colpali_engine.fireworks.models import ColIdefics3, ColIdefics3Processor
# Import the cythonized batch processing function.
from cython_optimizations import process_image_batch

# ------ Parameters ------
MAX_PDF_PAGES = 300       # Skip PDFs with more than 300 pages.
MAX_PDFS = 1              # Process up to 1 PDF.
BATCH_SIZE = 24           # Number of pages to process per batch.
THREAD_WORKERS = 5        # Number of threads for batch inference.
REFERENCES_FOLDER = "./references_images"  # Folder to store JSON embeddings (for image pipeline)

def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

# --- Directory Clearing Functions ---
def clear_directory(directory):
    """
    Deletes the given directory and its contents if it exists, then recreates it.
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)
        print(f"Cleared directory: {directory}")
    os.makedirs(directory)
    print(f"Recreated directory: {directory}")

def clear_required_directories():
    """
    Clears directories before processing.
    Modify the list to include any directories that need to be cleared.
    """
    directories_to_clear = ["downloaded_pdfs", "chroma_db_images", REFERENCES_FOLDER]
    for directory in directories_to_clear:
        clear_directory(directory)

# -------------------------------
def extract_images_from_pdfs(pdf_folder, max_pdfs=MAX_PDFS, max_pages=MAX_PDF_PAGES):
    """
    Extract images and metadata from PDFs.
    Skips any PDF with more than max_pages.
    Returns:
      - all_pages: list of tuples (pdf_name, page_idx, page_img)
      - pdf_times: dict mapping pdf_name to extraction time.
    """
    all_pages = []
    pdf_times = {}
    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))[:max_pdfs]
    print(f"Found {len(pdf_files)} PDF(s) for extraction.", flush=True)
    total_extraction_time = 0.0
    for pdf_path in pdf_files:
        pdf_name = os.path.basename(pdf_path)
        start = time.time()
        try:
            print(f"Extracting images from {pdf_name}...", flush=True)
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            pages = convert_from_bytes(pdf_bytes, dpi=150)
            extraction_time = time.time() - start
            pdf_times[pdf_name] = extraction_time
            total_extraction_time += extraction_time
            print(f"{pdf_name}: {len(pages)} page(s) extracted in {extraction_time:.2f} sec.", flush=True)
            if len(pages) > max_pages:
                print(f"Skipping {pdf_name} (more than {max_pages} pages).", flush=True)
                continue
            for idx, page_img in enumerate(pages):
                all_pages.append((pdf_name, idx, page_img))
        except Exception as e:
            print(f"Error extracting {pdf_name}: {e}", flush=True)
    print(f"Total extraction time: {total_extraction_time:.2f} sec.", flush=True)
    return all_pages, pdf_times

def batch_data(data, batch_size):
    """Yield successive batches from data."""
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

def dump_embeddings_json(all_embeddings, output_folder):
    """
    Groups embeddings by PDF name and dumps each PDF's embeddings (with metadata)
    to a JSON file in the output folder.
    """
    ensure_folder(output_folder)
    grouped = {}
    for item in all_embeddings:
        pdf_name = item["metadata"]["pdf_name"]
        if pdf_name not in grouped:
            grouped[pdf_name] = []
        grouped[pdf_name].append(item)
    
    for pdf_name, embeddings in grouped.items():
        output_file = os.path.join(output_folder, f"{pdf_name}_embeddings.json")
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump({"pdf": pdf_name, "embeddings": embeddings}, f)
            print(f"Dumped embeddings for {pdf_name} to {output_file}", flush=True)
        except Exception as e:
            print(f"Error dumping JSON for {pdf_name}: {e}", flush=True)

def query_collection(query_text, model, processor, n_results=5):
    """
    Processes query text to compute its embedding using the given model and processor,
    then queries the existing ChromaDB collection for the top n_results.
    Returns the query results.
    """
    print("Processing query text for retrieval...", flush=True)
    batch_query = processor.process_queries([query_text]).to("cpu")
    with torch.no_grad():
        query_embedding = model(**batch_query)
    query_vector = query_embedding.mean(dim=1).squeeze(0)
    query_vector_list = query_vector.tolist()
    print("Computed query embedding.", flush=True)

    client = chromadb.Client(settings=Settings(persist_directory="./chroma_db_images"))
    collection = client.get_collection(name="pdf_images_embeddings")
    
    print(f"Querying the collection for the top {n_results} results...", flush=True)
    results = collection.query(
        query_embeddings=[query_vector_list],
        n_results=n_results,
        include=["metadatas", "documents"]
    )
    return results

def query_loop(model, processor):
    """
    Continuously prompts the user for query text, performs the query, and displays the results.
    """
    print("\nEnter query text to search the database (type 'exit' to quit):", flush=True)
    while True:
        query_text = input("Query: ").strip()
        if query_text.lower() in ("exit", "quit"):
            print("Exiting query loop.", flush=True)
            break
        results = query_collection(query_text, model, processor, n_results=5)
        metadatas = results.get("metadatas", [[]])
        documents = results.get("documents", [[]])
        if metadatas and metadatas[0]:
            print("Query Results:", flush=True)
            for i, metadata in enumerate(metadatas[0]):
                document = documents[0][i] if documents and documents[0] else ""
                print(f"\nResult {i+1}:", flush=True)
                print(f"  Metadata (PDF info): {metadata}", flush=True)
                print(f"  Document: {document}", flush=True)
        else:
            print("No relevant results found.", flush=True)

def main():
    overall_start = time.time()
    pdf_folder = "./downloaded_pdfs"  # Folder where your PDFs are stored

    # ----- Extraction Stage: Extract images from PDFs -----
    all_pages, pdf_times = extract_images_from_pdfs(pdf_folder, max_pdfs=MAX_PDFS, max_pages=MAX_PDF_PAGES)
    print(f"Total pages collected for processing: {len(all_pages)}", flush=True)

    # ----- Load Colpali Model & Processor in the Main Process -----
    custom_cache_dir = "./"
    print("Loading Colpali model and processor in the main process...", flush=True)
    model = ColIdefics3.from_pretrained(
        "vidore/colSmol-256M",
        torch_dtype=torch.float32,
        attn_implementation="eager",
        cache_dir=custom_cache_dir
    ).eval()
    processor = ColIdefics3Processor.from_pretrained(
        "vidore/colSmol-256M",
        cache_dir=custom_cache_dir
    )

    # ----- Batch Inference Stage using ThreadPoolExecutor -----
    inference_start = time.time()
    all_embeddings = []
    batches = list(batch_data(all_pages, BATCH_SIZE))
    print(f"Processing {len(batches)} batches with batch size {BATCH_SIZE}...", flush=True)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=THREAD_WORKERS) as executor:
        futures = {executor.submit(process_image_batch, batch, model, processor): batch for batch in batches}
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

    # ----- Report Per-PDF Extraction Times -----
    print("\nPer-PDF Extraction Times:", flush=True)
    for pdf, t in pdf_times.items():
        print(f"  {pdf}: {t:.2f} sec", flush=True)
    print(f"Total extraction time: {sum(pdf_times.values()):.2f} sec.", flush=True)

    # -----------------------------
    # Insert Embeddings into Persistent ChromaDB (Image Pipeline)
    # -----------------------------
    client = chromadb.PersistentClient(
        path="./chroma_db_images",
        settings=Settings()
    )
    collection_name = "pdf_images_embeddings"
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

    # ----- Dump Embeddings to JSON Files (References) -----
    ensure_folder(REFERENCES_FOLDER)
    dump_embeddings_json(all_embeddings, REFERENCES_FOLDER)

    overall_end = time.time()
    print(f"\nOverall pipeline time: {overall_end - overall_start:.2f} sec.", flush=True)

    # Continuous Query Loop (if needed)
    # query_loop(model, processor)

if __name__ == "__main__":
    # Clear directories before processing.
    #clear_required_directories()
    # Ensure references folder is available.
    ensure_folder(REFERENCES_FOLDER)
    main()

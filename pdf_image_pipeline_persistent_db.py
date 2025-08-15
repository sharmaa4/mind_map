import os
import glob
import time
import sys
import torch
import pysqlite3
print("pysqlite3 version:", pysqlite3.sqlite_version)  # Verify version is >= 3.35.0
sys.modules["sqlite3"] = pysqlite3

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

# Import the corrected model loading function
from query_vector import load_colpali_model_and_processor

# ------ Parameters ------
MAX_PDF_PAGES = 300
MAX_PDFS = 1
BATCH_SIZE = 16  # Reduced batch size for stability with the new model
THREAD_WORKERS = 4
REFERENCES_FOLDER = "./references_images"

def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def process_image_batch_python(batch_data, model, processor):
    """
    Processes a batch of page images using the Hugging Face Transformers model.
    This replaces the old Cython function.
    """
    results = []
    images = [item[2] for item in batch_data]
    meta_list = [(item[0], item[1]) for item in batch_data]

    # Process images with the processor
    inputs = processor(images=images, return_tensors="pt")

    with torch.no_grad():
        # Get image features (embeddings) using the correct API for ColPaliForRetrieval
        image_features = model.get_image_features(**inputs)

    for i, (pdf_name, page_idx) in enumerate(meta_list):
         results.append({
             "id": f"{pdf_name}_page_{page_idx}",
             "embedding": image_features[i].tolist(),
             "metadata": {"pdf_name": pdf_name, "page_idx": page_idx},
             "document": f"Image from page {page_idx + 1} of {pdf_name}"
         })
    return results

def extract_images_from_pdfs(pdf_folder, max_pdfs=MAX_PDFS, max_pages=MAX_PDF_PAGES):
    """
    Extracts images from PDFs.
    """
    all_pages = []
    pdf_times = {}
    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))[:max_pdfs]
    print(f"Found {len(pdf_files)} PDF(s) for extraction.", flush=True)
    
    for pdf_path in pdf_files:
        pdf_name = os.path.basename(pdf_path)
        start = time.time()
        try:
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            pages = convert_from_bytes(pdf_bytes, dpi=150)
            
            if len(pages) > max_pages:
                print(f"Skipping {pdf_name} (has {len(pages)} pages, max is {max_pages}).", flush=True)
                continue

            for idx, page_img in enumerate(pages):
                all_pages.append((pdf_name, idx, page_img))

            extraction_time = time.time() - start
            pdf_times[pdf_name] = extraction_time
            print(f"{pdf_name}: {len(pages)} page(s) extracted in {extraction_time:.2f} sec.", flush=True)

        except Exception as e:
            print(f"Error extracting {pdf_name}: {e}", flush=True)
            
    return all_pages, pdf_times

def batch_data(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

def main():
    overall_start = time.time()
    pdf_folder = "./downloaded_pdfs"
    ensure_folder(pdf_folder)
    ensure_folder(REFERENCES_FOLDER)

    all_pages, pdf_times = extract_images_from_pdfs(pdf_folder)
    print(f"Total pages collected for processing: {len(all_pages)}", flush=True)

    if not all_pages:
        print("No pages to process. Exiting.")
        return

    print("Loading Colpali model and processor from Hugging Face...", flush=True)
    model, processor = load_colpali_model_and_processor()

    inference_start = time.time()
    all_embeddings = []
    batches = list(batch_data(all_pages, BATCH_SIZE))
    print(f"Processing {len(batches)} batches with batch size {BATCH_SIZE}...", flush=True)

    with ThreadPoolExecutor(max_workers=THREAD_WORKERS) as executor:
        futures = {executor.submit(process_image_batch_python, batch, model, processor): batch for batch in batches}
        for i, future in enumerate(as_completed(futures)):
            try:
                batch_results = future.result()
                all_embeddings.extend(batch_results)
                print(f"Completed batch {i+1}/{len(batches)}: Processed {len(batch_results)} embeddings.", flush=True)
            except Exception as e:
                print(f"Error processing a batch: {e}", flush=True)
    
    inference_end = time.time()
    print(f"Total inference time: {inference_end - inference_start:.2f} sec.", flush=True)

    client = chromadb.PersistentClient(path="./chroma_db_images", settings=Settings())
    collection_name = "pdf_images_embeddings"
    
    # Delete the collection if it exists to ensure a fresh start
    try:
        client.delete_collection(name=collection_name)
        print(f"Deleted existing collection: '{collection_name}'")
    except Exception:
        pass # Collection didn't exist, which is fine.

    collection = client.create_collection(name=collection_name)
    print(f"Created new ChromaDB collection: {collection_name}", flush=True)

    if all_embeddings:
        ids = [item["id"] for item in all_embeddings]
        embeddings = [item["embedding"] for item in all_embeddings]
        metadatas = [item["metadata"] for item in all_embeddings]
        documents = [item["document"] for item in all_embeddings]
        
        collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
        print(f"Upserted {len(ids)} embeddings into collection '{collection_name}'.", flush=True)
    else:
        print("No embeddings were generated to upsert.", flush=True)

    overall_end = time.time()
    print(f"\nOverall pipeline time: {overall_end - overall_start:.2f} sec.", flush=True)

if __name__ == "__main__":
    main()

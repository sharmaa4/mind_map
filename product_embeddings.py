import os
import glob
import json
import time
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import tiktoken
from tqdm import tqdm

# -----------------------------
# Configuration and API settings
# -----------------------------
AZURE_OPENAI_ENDPOINT = "https://engassist-eus-dev-aais.openai.azure.com/openai/deployments/hackathon-emb-emb3l-team-21-cgcwn/embeddings?api-version=2023-05-15"
AZURE_OPENAI_API_KEY = "de7a14848db7462c9783adbcfbb0b430"  # Replace with your actual API key

# Maximum tokens per chunk. The model has a limit of 8192 tokens, so we leave some margin.
MAX_TOKENS_PER_CHUNK = 8000  
# Overlap between chunks (in tokens)
OVERLAP_TOKENS = 200  
MODEL_NAME = "text-embedding-ada-002"

# -----------------------------
# Token counting and chunking using tiktoken
# -----------------------------
def count_tokens(text, model=MODEL_NAME):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def chunk_text(text, max_tokens=MAX_TOKENS_PER_CHUNK, overlap_tokens=OVERLAP_TOKENS, model=MODEL_NAME):
    """
    Splits text into overlapping chunks based on token count.
    Each chunk will have at most max_tokens tokens, with an overlap of overlap_tokens tokens.
    """
    encoding = tiktoken.encoding_for_model(model)
    all_tokens = encoding.encode(text)
    chunks = []
    start = 0
    while start < len(all_tokens):
        end = min(start + max_tokens, len(all_tokens))
        chunk_tokens = all_tokens[start:end]
        chunk = encoding.decode(chunk_tokens)
        chunks.append(chunk)
        if end == len(all_tokens):
            break
        start = end - overlap_tokens  # overlap tokens
    return chunks

# -----------------------------
# Function to call Azure OpenAI API for one chunk
# with infinite exponential backoff until a valid (non-empty) embedding is returned.
# -----------------------------
def get_embedding_chunk(chunk, backoff_factor=2):
    attempt = 0
    tokens = count_tokens(chunk)
    while True:
        headers = {
            "Content-Type": "application/json",
            "api-key": AZURE_OPENAI_API_KEY
        }
        payload = {"input": [chunk]}  # API expects a list of strings
        try:
            response = requests.post(AZURE_OPENAI_ENDPOINT, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                embedding = result["data"][0].get("embedding", [])
                if embedding and len(embedding) > 0:
                    return embedding
                else:
                    print("Empty embedding received; retrying...")
            else:
                print(f"API call failed (status {response.status_code}): {response.text}")
        except Exception as e:
            print(f"Exception during API call: {e}")
        sleep_time = backoff_factor ** attempt
        print(f"Retrying after {sleep_time} seconds (attempt {attempt+1})...")
        time.sleep(sleep_time)
        attempt += 1

# -----------------------------
# Process one product file: read text, split into token-based chunks,
# get embeddings for each chunk, and add metadata (including links).
# -----------------------------
def process_product_file(file_path, output_dir, model=MODEL_NAME):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

    # Chunk the text based on token count
    chunks = chunk_text(text, max_tokens=MAX_TOKENS_PER_CHUNK, overlap_tokens=OVERLAP_TOKENS, model=model)
    
    embeddings = []
    # For each chunk, call the API to get its embedding.
    for idx, chunk in enumerate(chunks):
        print(f"Processing chunk {idx+1}/{len(chunks)} for file {file_path} (approx. {count_tokens(chunk)} tokens)...")
        emb = get_embedding_chunk(chunk)
        embeddings.append(emb)
    
    product = os.path.splitext(os.path.basename(file_path))[0]
    # Attempt to load corresponding .links file, if it exists.
    links_file = os.path.join(os.path.dirname(file_path), f"{product}.links")
    links = []
    if os.path.exists(links_file):
        try:
            with open(links_file, "r", encoding="utf-8") as lf:
                content = lf.read().strip()
                try:
                    links = json.loads(content)
                    if not isinstance(links, list):
                        links = [links]
                except json.JSONDecodeError:
                    # Fallback: split by newlines
                    links = [line.strip() for line in content.splitlines() if line.strip()]
        except Exception as e:
            print(f"Error reading links file {links_file}: {e}")
    # Convert links list into a comma-separated string for metadata storage.
    if isinstance(links, list):
        links_str = ", ".join(links)
    else:
        links_str = links

    result = {
        "product": product,
        "embeddings": embeddings,
        "num_chunks": len(chunks),
        "links": links_str
    }
    output_file = os.path.join(output_dir, f"{product}_embeddings.json")
    try:
        with open(output_file, "w", encoding="utf-8") as out_f:
            json.dump(result, out_f)
        print(f"Saved embeddings for {product} with {len(chunks)} chunks to {output_file}")
    except Exception as e:
        print(f"Error saving embeddings for {product}: {e}")
    
    return product

# -----------------------------
# Main pipeline
# -----------------------------
def main():
    input_folder = "extracted_text"      # Folder containing product datasheet text files.
    output_folder = "product_embeddings"  # Folder to store JSON output embeddings.
    os.makedirs(output_folder, exist_ok=True)
    
    product_files = glob.glob(os.path.join(input_folder, "*.txt"))
    print(f"Found {len(product_files)} product text files.")
    
    start_time = time.time()
    processed_products = []
    
    # Use ThreadPoolExecutor to process files concurrently, with a progress bar.
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_product_file, file_path, output_folder): file_path for file_path in product_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Files"):
            product = future.result()
            if product is not None:
                processed_products.append(product)
    
    end_time = time.time()
    print(f"Processed {len(processed_products)} products in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()


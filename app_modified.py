import streamlit as st
# Configure the Streamlit page


import pysqlite3
import sys
print("pysqlite3 version:", pysqlite3.sqlite_version)  # Verify version is >= 3.35.0
sys.modules["sqlite3"] = pysqlite3
import shutil
import os
import glob
import json
import time
from concurrent.futures import ThreadPoolExecutor

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from query_vector import get_query_embedding  # Your Azure/OpenAI-based query embedding function
from query_vector import load_colpali_model_and_processor, get_image_query_embedding  # Your ColPali-based query embedding function

# Configure the Streamlit page
st.set_page_config(page_title="Global Product Search", layout="wide")
st.title("Product Search for Analog Devices Products")

st.sidebar.image("logo.png", width=200)


# Initialize session state for both tabs
if 'text_results' not in st.session_state:
    st.session_state.text_results = None
if 'image_results' not in st.session_state:
    st.session_state.image_results = None


import json
import os

import requests

import os
import fitz  # PyMuPDF

import os
import fitz  # PyMuPDF
from PIL import Image

def feed_images_by_pdf_index(query_results, pdf_folder="downloaded_pdfs"):
    """
    Extracts images from PDFs based on the 'ids' field in the query results.
    
    Each ID is expected to be in the format:
        "MAX2648.pdf_page_7"
    
    This function groups IDs by PDF name, opens the corresponding PDF from pdf_folder,
    and extracts an image of each specified page using PyMuPDF.
    
    Args:
        query_results (dict): The query results JSON as a dictionary.
        pdf_folder (str): Directory where the PDF files are stored.
    
    Returns:
        list: A list of dictionaries containing:
            - "pdf_name": the name of the PDF,
            - "page_number": the extracted page number,
            - "image": the extracted image as a PIL Image object.
    """
    results = []
    ids_nested = query_results.get("ids", [])
    if not ids_nested or not isinstance(ids_nested, list) or len(ids_nested) == 0:
        print("No IDs found in query results.")
        return results

    ids_list = ids_nested[0]  # Take the first list of IDs

    # Group IDs by PDF name
    pdf_page_map = {}
    for id_str in ids_list:
        try:
            # Expecting format: "MAX2648.pdf_page_7"
            if "_page_" not in id_str:
                print(f"Unexpected ID format: {id_str}")
                continue
            pdf_name_with_ext, page_str = id_str.split("_page_")
            pdf_name = pdf_name_with_ext.strip()
            try:
                page_number = int(page_str.strip())
            except ValueError:
                print(f"Invalid page number in ID: {id_str}")
                continue

            pdf_page_map.setdefault(pdf_name, []).append(page_number)
        except Exception as e:
            print(f"Error processing ID '{id_str}': {e}")

    # For each PDF, open it and extract the specified pages as images.
    for pdf_name, page_numbers in pdf_page_map.items():
        pdf_path = os.path.join(pdf_folder, pdf_name)
        if not os.path.exists(pdf_path):
            print(f"PDF file not found: {pdf_path}")
            continue
        try:
            doc = fitz.open(pdf_path)
            for page_number in page_numbers:
                if page_number < len(doc):
                    page = doc.load_page(page_number)
                    pix = page.get_pixmap()  # Render page to an image
                    # Convert the pixmap to a PIL Image object
                    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    results.append({
                        "pdf_name": pdf_name,
                        "page_number": page_number,
                        "image": image
                    })
                    print(f"Extracted image from {pdf_name} at page {page_number}.")
                else:
                    print(f"Page number {page_number} is out of range for {pdf_path}. Total pages: {len(doc)}")
            doc.close()
        except Exception as e:
            print(f"Error processing PDF '{pdf_name}': {e}")

    return results

# Example usage:
if __name__ == "__main__":
    # Simulated query results for PDF image queries.
    query_results = {
        "ids": [[
            "MAX2648.pdf_page_7",
            "MAX2648.pdf_page_4",
            "MAX2648.pdf_page_3",
            "MAX2648.pdf_page_0",
            "MAX2648.pdf_page_1"
        ]]
    }
    
    images_info = feed_images_by_pdf_index(query_results, pdf_folder="downloaded_pdfs")
    for info in images_info:
        print(f"PDF: {info['pdf_name']}, Page: {info['page_number']}")
        # For example, to display the image using PIL:
        # info["image"].show()


def feed_pdf_references(query_results):
    """
    Extracts page references from the query results and returns a concatenated string of references.
    
    Each ID in the query results is expected to be in the format:
        "ADXRS150.pdf_page_3_chunk_0"
        
    This function extracts the PDF file name and the page number from each ID and creates a reference
    string like "Reference: ADXRS150.pdf, Page: 3". All references are concatenated and returned.
    
    Args:
        query_results (dict): The query results JSON as a dictionary.
    
    Returns:
        str: The concatenated page references.
    """
    references = []
    ids_nested = query_results.get("ids", [])
    if not ids_nested or not isinstance(ids_nested, list) or len(ids_nested) == 0:
        print("No IDs found in query results.")
        return ""
    
    ids_list = ids_nested[0]  # Take the first list of IDs
    for id_str in ids_list:
        try:
            # Expected format: "ADXRS150.pdf_page_3_chunk_0"
            parts = id_str.split("_")
            if len(parts) < 5:
                print(f"Unexpected ID format: {id_str}")
                continue
            pdf_name = parts[0]  # e.g., "ADXRS150.pdf"
            # The third element is the page number as a string (e.g., "3")
            page_number = parts[2]
            references.append(f"Reference: {pdf_name}, Page: {page_number}")
        except Exception as e:
            print(f"Error processing ID '{id_str}': {e}")
    
    result_references = "\n".join(references)
    print("Page references generated successfully.")
    return result_references



def feed_documents_to_concat_by_pdf_index_pdf(query_results, pdf_folder="downloaded_pdfs"):
    """
    Extracts text from PDFs using the 'ids' field in the query results.
    
    Each ID is expected to be in the format:
        "ADXRS150.pdf_page_3_chunk_0"
    
    This function opens the corresponding PDF file (e.g., "ADXRS150.pdf") from pdf_folder,
    extracts text from the specified page using PyMuPDF, and concatenates the text.
    
    Args:
        query_results (dict): The query results JSON as a dictionary.
        pdf_folder (str): Directory where the PDF files are stored.
    
    Returns:
        str: The concatenated text extracted from the specified pages.
    """
    concatenated_text_parts = []
    
    # Extract the list of IDs (assumed to be nested as a list of lists)
    ids_nested = query_results.get("ids", [])
    if not ids_nested or not isinstance(ids_nested, list) or len(ids_nested) == 0:
        print("No IDs found in query results.")
        return ""
    
    ids_list = ids_nested[0]  # Taking the first list of IDs
    for id_str in ids_list:
        try:
            # Expecting format like "ADXRS150.pdf_page_3_chunk_0"
            parts = id_str.split("_")
            if len(parts) < 5:
                print(f"Unexpected ID format: {id_str}")
                continue
            
            # First part is the PDF file name (e.g., "ADXRS150.pdf")
            pdf_name_with_ext = parts[0]
            pdf_path = os.path.join(pdf_folder, pdf_name_with_ext)
            
            # Extract page number from the third part (e.g., "3")
            try:
                page_number = int(parts[2])
            except ValueError:
                print(f"Cannot convert page number to int in ID: {id_str}")
                continue
            
            # Open the PDF using PyMuPDF
            if os.path.exists(pdf_path):
                doc = fitz.open(pdf_path)
                if page_number < len(doc):
                    page = doc.load_page(page_number)
                    page_text = page.get_text("text")
                    concatenated_text_parts.append(page_text.strip())
                    print(f"Extracted text from {pdf_name_with_ext} at page {page_number}.")
                else:
                    print(f"Page number {page_number} is out of range for {pdf_path}. Total pages: {len(doc)}")
                doc.close()
            else:
                print(f"PDF file not found: {pdf_path}")
        except Exception as e:
            print(f"Error processing ID '{id_str}': {e}")
    
    result_text = "\n\n".join(concatenated_text_parts)
    print("Concatenated text generated successfully.")
    return result_text

def get_structured_output_from_openai(concatenated_text, user_query, prompt_template=None):
    """
    Sends the concatenated text and a user query to the Azure OpenAI API and returns the structured output.
    
    Args:
        concatenated_text (str): The concatenated text from the extracted text files.
        user_query (str): The query provided by the user.
        prompt_template (str, optional): Additional prompt instructions for structure.
            If None, a default prompt is used.
    
    Returns:
        str: The structured output as returned by the OpenAI API.
    """
    # Use a default prompt template if none is provided.
    if prompt_template is None:
        prompt_template = (
            "Based on the following product documentation text and the user \
            query, generate a structured summary based on the user query\
            given"\
           \
            "Don't mention any kinds of links or something" \
            "Keep your answers aligned with user's query only" \
            "If user is asking very generic questions example 'What kind of \
            products ADI/ Analog Devices have then mention the user to be ask \
            specific questions related to certain categories specifically \
            documentation related and not generic questions"\
            "You can answer such questions using the knowledge you are\
            already,trained on no need to provide summary of the documents in\
            this \
            case"\
            "VERY VERY IMPORTANT !!!!!!******If you are not able find any\
            information\
            or you don't have enough*******Be cautious*********!!!!!!\
            \
            data then please mention you are not able to answer this query" \
            "User Query: {user_query}\n\n"
        )
    
    # Construct the full prompt by inserting the user query and appending the concatenated text.
    prompt = prompt_template.format(user_query=user_query) + concatenated_text

    # Azure OpenAI API endpoint and key information.
    url = "https://engassist-eus-dev-aais.openai.azure.com/openai/deployments/hackathon-gpt4omini-team-21-cgcwn/chat/completions?api-version=2024-10-21"
    headers = {
        "Content-Type": "application/json",
        "api-key": "de7a14848db7462c9783adbcfbb0b430"
    }
    
    # Prepare the payload for a chat completion request.
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that \
            structures product documentation based on user queries. The \
             queries given are of documentation of thousands of products of \
             a semiconductor company so always provide answers with this \
             context"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 16384 
    }
    
    # Send the request to the OpenAI API.
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code != 200:
        raise Exception(f"OpenAI API call failed with status code {response.status_code}: {response.text}")
    
    # Extract the structured response from the API.
    data = response.json()
    structured_output = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    return structured_output


def concatenate_text_files(documents, extracted_text_dir="extracted_text"):
    """
    Given a list of document strings (e.g. "MAX2374 - chunk 0"),
    extract the product name from each document and retrieve the corresponding
    text file (e.g. extracted_text/MAX2374.txt). Concatenate the contents of these
    files and return the combined text.
    
    Args:
        documents (list): List of document strings, e.g. ["MAX2374 - chunk 0", ...]
        extracted_text_dir (str): Directory where the text files are stored.
    
    Returns:
        str: Combined text from all the matching files.
    """
    combined_text = []
    for doc in documents:
        # Extract product name by splitting at " -"
        product_name = doc.split(" -")[0].strip()
        file_name = f"{product_name}.txt"
        file_path = os.path.join(extracted_text_dir, file_name)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                combined_text.append(content)
        else:
            print(f"File not found: {file_path}")
    return "\n\n".join(combined_text)


def feed_documents_to_concat(query_results, extracted_text_dir="extracted_text"):
    """
    Extracts document strings from the query results and feeds them to the
    concatenate_text_files function.
    
    Args:
        query_results (dict): The query results JSON as a dictionary.
        extracted_text_dir (str): Directory where the text files are stored.
    
    Returns:
        str: The concatenated text from all the corresponding files.
    """
    # Assume the "documents" field is a list of lists. Get the first list.
    documents_nested = query_results.get("documents", [])
    if documents_nested and isinstance(documents_nested, list) and len(documents_nested) > 0:
        documents = documents_nested[0]
    else:
        documents = []
    return concatenate_text_files(documents, extracted_text_dir)

import os

def feed_documents_to_concat_by_pdf_index(query_results, extracted_text_dir="extracted_text"):
    """
    Extracts text from PDFs based on the 'ids' field in the query results.
    
    Each ID is expected to be in the format:
        "pdfName.pdf_page_{page_number}_chunk_{chunk_index}"
    
    The corresponding text file is assumed to be stored in the extracted_text directory,
    with a file name derived from the PDF name (e.g., "adh753s.txt" for "adh753s.pdf").
    The text file is assumed to be split into pages using a delimiter (e.g., "\f").
    
    Args:
        query_results (dict): The query results JSON as a dictionary.
        extracted_text_dir (str): Directory where the text files are stored.
    
    Returns:
        str: The concatenated text extracted from the specified pages.
    """
    concatenated_text_parts = []
    
    # Extract the list of IDs from query_results (assumed to be nested as a list of lists)
    ids_nested = query_results.get("ids", [])
    if not ids_nested or not isinstance(ids_nested, list) or len(ids_nested) == 0:
        print("No IDs found in query results.")
        return ""
    
    ids_list = ids_nested[0]  # Taking the first list
    for id_str in ids_list:
        # Expecting format like "adh753s.pdf_page_3_chunk_0"
        try:
            parts = id_str.split("_")
            # parts[0]: "adh753s.pdf"
            # parts[1]: "page"
            # parts[2]: page number (e.g., "3")
            # parts[3]: "chunk"
            # parts[4]: chunk index (e.g., "0")
            pdf_name_with_ext = parts[0]  # e.g., "adh753s.pdf"
            # Derive base name by removing the .pdf extension.
            pdf_base = pdf_name_with_ext.replace(".pdf", "")
            page_number = int(parts[2])
            
            # Construct the expected text file name. Here we assume the file is named like "adh753s.txt".
            file_name = f"{pdf_base}.txt"
            file_path = os.path.join(extracted_text_dir, file_name)
            
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                # Assume pages are separated by the form feed character ("\f")
                pages = content.split("\f")
                if page_number < len(pages):
                    page_text = pages[page_number].strip()
                    concatenated_text_parts.append(page_text)
                else:
                    print(f"Page number {page_number} is out of range for file {file_path}.")
            else:
                print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error processing id '{id_str}': {e}")
    
    return "\n\n".join(concatenated_text_parts)


    


def extract_and_display_documents(query_results):
    """
    Extracts documents from the query results JSON and displays each document 
    along with its corresponding link from the metadatas.
    
    Args:
        query_results (dict): The JSON object containing the query results.
    """
    # Retrieve the nested list of documents.
    documents_nested = query_results.get("documents")
    metadatas_nested = query_results.get("metadatas")
    
    if not documents_nested:
        st.write("No documents found in the query results.")
        return
    
    # Assume the structure is a list of lists; extract the first list.
    documents = documents_nested[0]
    metadatas = metadatas_nested[0] if metadatas_nested and isinstance(metadatas_nested, list) and len(metadatas_nested) > 0 else []
    
    # Loop through each document and its corresponding metadata.
    for idx, doc in enumerate(documents, start=1):
        # Try to get the corresponding metadata; if not present, default to an empty dict.
        metadata = metadatas[idx-1] if idx-1 < len(metadatas) else {}
        link = metadata.get("links", "No link provided")
        st.write(f"**Document {idx}:** {doc}")
        st.write(f"**Link:** {link}")
        st.write("---")

    


# ----------------------------
# Persistent ChromaDB Client & Collection Setup
# ----------------------------
@st.cache_resource(show_spinner=False)
def get_chroma_client(persist_directory="./db/"):
    # Create a persistent Chroma client using the specified persist_directory.
    client = chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings()
    )
    return client

@st.cache_resource(show_spinner=False)
def get_chroma_collection():
    client = get_chroma_client()
    collection_name = "analog_products"
    # Use OpenAI embedding function for retrieval (ensure your OPENAI_API_KEY and OPENAI_API_ENDPOINT are set)
    
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=None
    )
    return collection

collection = get_chroma_collection()

# ----------------------------
# Embedding Ingestion Function (Global Search)
# ----------------------------
def ingest_product_embeddings(collection,embeddings_folder="product_embeddings",max_points=22000):
    points = []
    embedding_files = glob.glob(os.path.join(embeddings_folder, "*.json"))
    st.write(f"Ingestion: Found {len(embedding_files)} embedding files.")
    
    def process_file(file_path):
        local_points = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            st.write(f"Error reading file {file_path}: {e}")
            return local_points

        product = data.get("product", "unknown")
        embeddings = data.get("embeddings", [])
        num_chunks = data.get("num_chunks", len(embeddings))
        links = data.get("links", [])
        if isinstance(links, list):
            links_str = ", ".join(links)
        else:
            links_str = links

        for idx, vector in enumerate(embeddings):
            if not vector or len(vector) == 0:
                st.write(f"Skipping empty embedding at pos {idx} for product {product}.")
                continue
            point_id = f"{product}_{idx}"
            document = f"{product} - chunk {idx}"
            metadata = {
                "product": product,
                "chunk_index": idx,
                "num_chunks": num_chunks,
                "links": links_str
            }
            local_points.append({
                "id": point_id,
                "vector": vector,
                "document": document,
                "metadata": metadata
            })
        return local_points

    load_start = time.time()
    with ThreadPoolExecutor() as executor:
        futures = list(executor.map(process_file, embedding_files))
        for pts in futures:
            points.extend(pts)
            if max_points and len(points) >= max_points:
                points = points[:max_points]
                break
    load_latency = time.time() - load_start
    st.write(f"Ingestion: Loaded {len(points)} points in {load_latency:.2f} seconds.")

    if points:
        upsert_start = time.time()
        collection.add(
            ids=[p["id"] for p in points],
            embeddings=[p["vector"] for p in points],
            documents=[p["document"] for p in points],
            metadatas=[p["metadata"] for p in points]
        )
        upsert_latency = time.time() - upsert_start
        st.write(f"Ingestion: Upserted {len(points)} points in {upsert_latency:.2f} seconds.")
    else:
        st.write("Ingestion: No embeddings found to upsert.")

# Ingest embeddings once per session.
##if "embeddings_ingested" not in st.session_state:
##    with st.spinner("Ingesting product embeddings into the persistent vector DB..."):
##        ingest_product_embeddings(collection)
##    st.session_state["embeddings_ingested"] = True

# ----------------------------
# Global Search UI
# ----------------------------
query_text = st.text_input("Enter your search query:", placeholder="e.g., Wideband Low Noise Amplifier datasheet")
if st.button("Search"):
    if not query_text:
        st.warning("Please enter a valid search query.")
    else:
        with st.spinner("Generating query embedding and searching..."):
            try:
                # Use your OpenAI-based query embedding function.
                query_embedding = get_query_embedding(query_text)
                # Query the collection; Chroma uses its stored embedding function for retrieval.
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=5
                )
                #st.header("Extracted Documents and Links")
                concatenated_text= feed_documents_to_concat(results)
                output = get_structured_output_from_openai(concatenated_text,query_text)
                st.write(output)
                extract_and_display_documents(results)
                st.success("Search completed!")
                
                #st.json(results)
            except Exception as e:
                st.error(f"Error during search: {e}")

# ----------------------------
# Detailed Product Pipeline: Product Detail View
# ----------------------------
# --- Helper Functions for Detailed Product Pipeline ---

import requests
import fitz  # PyMuPDF
import os
import subprocess

def download_pdf(pdf_link, product_name, download_dir="downloaded_pdfs", timeout=30):
    """Download a PDF using curl with a Firefox user-agent."""
    if os.path.exists("downloaded_pdfs"):
        shutil.rmtree("downloaded_pdfs")
    
    print(f"Downloading PDF via curl: {pdf_link}")
    output_path = os.path.join(download_dir, f"{product_name}.pdf")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        result = subprocess.run(
            [
                "curl",
                "-A", "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:101.0) Gecko/20100101 Firefox/101.0",
                pdf_link,
                "-o", output_path
            ],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if result.returncode != 0:
            print(f"Error downloading PDF {pdf_link}: {result.stderr}")
            return False
        print(f"Downloaded PDF to {output_path}")
        return True
    except Exception as e:
        print(f"Exception while downloading PDF {pdf_link}: {e}")
        return False


#def download_pdf(pdf_link, product_name, download_dir="downloaded_pdfs"):
#    if not os.path.exists(download_dir):
#        os.makedirs(download_dir)
#    pdf_path = os.path.join(download_dir, f"{product_name}.pdf")
#    response = requests.get(pdf_link)
#    if response.status_code == 200:
#        with open(pdf_path, "wb") as f:
#            f.write(response.content)
#        return pdf_path
#    else:
#        raise Exception(f"Failed to download PDF: {response.status_code}")
#
#def download_pdf(url, output_path, timeout=30):
#    """Download a PDF using curl with a Firefox user-agent."""
#    print(f"Downloading PDF via curl: {url}")
#    os.makedirs(os.path.dirname(output_path), exist_ok=True)
#    try:
#        result = subprocess.run(
#            [
#                "curl",
#                "-A", "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:101.0) Gecko/20100101 Firefox/101.0",
#                url,
#                "-o", output_path
#            ],
#            capture_output=True,
#            text=True,
#            timeout=timeout
#        )
#        if result.returncode != 0:
#            print(f"Error downloading PDF {url}: {result.stderr}")
#            return False
#        print(f"Downloaded PDF to {output_path}")
#        return True
#    except Exception as e:
#        print(f"Exception while downloading PDF {url}: {e}")
#        return False
    

def get_pdf_page_count(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        count = doc.page_count
        doc.close()
        return count
    except Exception as e:
        st.error(f"Error getting page count: {e}")
        return 0

def process_pdf_text(pdf_path):
    """
    Call your PDF -> text -> OpenAI embeddings pipeline here.
    For this example, we assume a function process_pdf_text_with_toc(pdf_path) exists.
    """
    try:
        from text_processing import process_pdf_text_with_toc
        return process_pdf_text_with_toc(pdf_path)
    except ImportError:
        st.error("Text processing function not found.")
        return ""

def process_pdf_images(pdf_path):
    """
    Call your PDF -> images -> Colpali pipeline here.
    For this example, we assume a function process_pdf_images exists.
    """
    try:
        from image_pipeline import process_pdf_images
        return process_pdf_images(pdf_path)
    except ImportError:
        st.error("Image processing function not found.")
        return []

# Helper Function: Extract PDF links from HTML using BeautifulSoup.
from bs4 import BeautifulSoup

def extract_pdf_links(html):
    """
    Parse the HTML content to extract all unique PDF links.
    If a link starts with '/', prepend 'https://www.analog.com' to form a full URL.
    Returns a list of PDF URLs.
    """
    soup = BeautifulSoup(html, "html.parser")
    pdf_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".pdf"):
            if href.startswith("/"):
                href = "https://www.analog.com" + href
            pdf_links.append(href)
    pdf_links = list(set(pdf_links))
    print(f"Found {len(pdf_links)} PDF links.")
    return pdf_links

# ----------------------------
# Detailed Product Pipeline Section
# ----------------------------
st.write("---")
st.header("Detailed Product Documentation Discussion")

# Let the user enter the product name for a detailed view.
product_to_deep = st.text_input("Enter the product name for detailed view:", placeholder="e.g., Product_XYZ")

if product_to_deep:
    # Locate the downloaded HTML file (assumed to be in "downloaded_html" folder)
    html_file_path = os.path.join("downloaded_html", f"{product_to_deep}.html")
    if os.path.exists(html_file_path):
        with open(html_file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        # Extract PDF links from the HTML content.
        pdf_links = extract_pdf_links(html_content)
        if pdf_links:
            st.write("PDF links found:")
            chosen_pdf_link = st.selectbox("Select a PDF link to download for detailed view", pdf_links)
        else:
            st.error("No PDF links found in the HTML file.")
            chosen_pdf_link = None
    else:
        st.error("HTML file for this product not found.")
        chosen_pdf_link = None

    # Download and process the PDF when the user clicks the button.
    if chosen_pdf_link and st.button("Load PDF for Detailed View"):
        with st.spinner("Loading PDF ..."):
            success = download_pdf(chosen_pdf_link, product_to_deep)
            if success:
                st.success("PDF Loaded! You may start processing PDF for \
                           text-based or image-based queries")
            else:
                st.error("Error downloading PDF.")

 

import subprocess

def run_pdf_text_pipeline():
    """
    Execute the standalone PDF text pipeline (pdf_text_pipeline_persistent_db.py)
    using a subprocess. Captures and returns the output.
    """
    try:
        # Run the external script; adjust timeout if needed
        result = subprocess.run(
            ["python", "pdf_text_pipeline_persistent_db.py"],
            capture_output=False,
            text=False,
            timeout=600  # adjust timeout as needed
        )
        if result.returncode != 0:
            st.error("PDF text pipeline failed:\n" + result.stderr)
        else:
            st.success("PDF text pipeline executed successfully. Persistent Database generated ")
    except Exception as e:
        st.error(f"Error executing PDF text pipeline: {e}")

import subprocess
import streamlit as st

def run_pdf_image_pipeline():
    """
    Execute the standalone PDF image pipeline (pdf_image_pipeline.py)
    using a subprocess. It is expected that this pipeline creates or updates
    a persistent vector DB, so we only display the log output.
    """
    try:
        result = subprocess.run(
            ["python", "pdf_image_pipeline_persistent_db.py"],
            capture_output=False,
            text=False,
            timeout=600  # Adjust timeout as needed.
        )
        if result.returncode != 0:
            st.error("PDF image pipeline failed:\n" + result.stderr)
        else:
            st.success("PDF image pipeline executed successfully!")
    except Exception as e:
        st.error(f"Error executing PDF image pipeline: {e}")

if st.button("Process PDF for Text-related Queries"):
    run_pdf_text_pipeline()

if st.button("Process PDF for Image-related Queries"):
    run_pdf_image_pipeline()

    
    

#import streamlit as st
#import chromadb
#from chromadb.config import Settings
#from query_vector import get_query_embedding  # Your OpenAI-based query embedding function
#import os
#import json
#
#
## ----------------------------
## Load Persistent ChromaDB Client & Collection
## ----------------------------
#def load_persistent_collection():
#    # Create a persistent Chroma client using the "./chroma_db" directory.
#    client = chromadb.PersistentClient(
#        path="./chroma_db",
#        settings=Settings()
#    )
#    collection_name = "pdf_embeddings"
#    try:
#        collection = client.get_collection(name=collection_name)
#        st.write(f"Loaded existing collection: '{collection_name}'.")
#    except Exception as e:
#        st.error(f"Error loading collection '{collection_name}': {e}")
#        collection = None
#    return collection
#
#collection = load_persistent_collection()
#
## ----------------------------
## Streamlit UI for Querying
## ----------------------------
#if collection:
#    query_text = st.text_input("Enter your search query for PDFs:", 
#                               placeholder="e.g., signal conditioning amplifier datasheet")
#    
#    if st.button("Query DB"):
#        if not query_text.strip():
#            st.warning("Please enter a valid query.")
#        else:
#            with st.spinner("Generating query embedding and searching..."):
#                try:
#                    # Generate query embedding using your OpenAI-based function.
#                    query_embedding = get_query_embedding(query_text)
#                    
#                    # Query the collection. The 'include' parameter retrieves metadata and document info.
#                    results = collection.query(
#                        query_embeddings=[query_embedding],
#                        n_results=5,
#                        include=["metadatas", "documents"]
#                    )
#                    
#                    st.success("Query completed!")
#                    st.write("### Query Results:")
#                    st.json(results)
#                except Exception as e:
#                    st.error(f"Error during query: {e}")
#else:
#    st.error("Could not load persistent collection.")



# Create two tabs: one for PDF Text Query and one for PDF Image Query.
tab_text, tab_image = st.tabs(["PDF Text Query", "PDF Image Query"])

# ----------------------------
# Functions to load persistent collections
# ----------------------------
def load_text_collection():
    # Persistent client for PDF text embeddings is stored in "./chroma_db"
    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings()
    )
    collection_name = "pdf_embeddings"
    try:
        collection = client.get_collection(name=collection_name)
        st.write(f"Loaded text collection: '{collection_name}'.")
    except Exception as e:
        st.error(f"Error loading text collection '{collection_name}': {e}")
        collection = None
    return collection

def load_image_collection():
    # Persistent client for PDF image embeddings is stored in "./chroma_db_images"
    client = chromadb.PersistentClient(
        path="./chroma_db_images",
        settings=Settings()
    )
    collection_name = "pdf_images_embeddings"
    try:
        collection = client.get_collection(name=collection_name)
        st.write(f"Loaded image collection: '{collection_name}'.")
    except Exception as e:
        st.error(f"Error loading image collection '{collection_name}': {e}")
        collection = None
    return collection

# ----------------------------
# PDF Text Query UI
# ----------------------------
with tab_text:
    st.header("PDF Text Query")
    text_collection = load_text_collection()
    query_text = st.text_input("Enter your search query for PDF Text:", 
                               placeholder="e.g., signal conditioning amplifier datasheet")
    if st.button("Query Text DB"):
        if not query_text.strip():
            st.warning("Please enter a valid query.")
        elif text_collection:
            with st.spinner("Generating query embedding and searching..."):
                try:
                    query_embedding = get_query_embedding(query_text)
                    results = text_collection.query(
                        query_embeddings=[query_embedding],
                        n_results=5,
                        include=["metadatas", "documents"]
                    )
                    concatenated_text = feed_documents_to_concat_by_pdf_index_pdf(results,pdf_folder="downloaded_pdfs")
                    output = get_structured_output_from_openai(concatenated_text,query_text)
                    st.session_state.text_results = {
                        'output': output,
                        'references': feed_pdf_references(results)
                    }
                    st.write(st.session_state.text_results['output'])
                    st.write(st.session_state.text_results['references'])
                    st.success("Search completed!")
                    st.success("Text Query completed!")
                except Exception as e:
                    st.error(f"Error during text query: {e}")
        else:
            st.error("Text collection not loaded.")

    # Display previous results if they exist
    if st.session_state.text_results:
        st.write(st.session_state.text_results['output'])
        st.write(st.session_state.text_results['references'])

# ----------------------------
# PDF Image Query UI
# ----------------------------
with tab_image:

    st.header("PDF Image Query")
    
    # Load the Colpali model and processor for image queries.
    model, processor = load_colpali_model_and_processor()
    
    # Text input for image query.
    image_query = st.text_input("Enter your query for PDF Images:", placeholder="e.g., image processing details")
    
    if st.button("Query Image DB"):
        if not image_query.strip():
            st.warning("Please enter a valid query.")
        else:
            with st.spinner("Generating image query embedding and searching..."):
                try:
                    # Compute the query embedding using the Colpali model.
                    query_embedding = get_image_query_embedding(image_query, model, processor)
                    # Load the persistent image collection.
                    client = chromadb.PersistentClient(
                        path="./chroma_db_images",
                        settings=Settings()
                    )
                    collection = client.get_collection(name="pdf_images_embeddings")
                    # Query the collection using the computed embedding.
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=5,
                        include=["metadatas", "documents"]
                    )
                    st.session_state.image_results = results
                    st.success("Image query completed!")
                    images_info = feed_images_by_pdf_index(results, pdf_folder="downloaded_pdfs")
                    for info in images_info:
                        st.write(f"PDF: {info['pdf_name']}, Page: {info['page_number']}")
                        st.image(info["image"], caption="Extracted PDF Page")

                    #st.json(results)
                except Exception as e:
                    st.error(f"Error during image query: {e}")

     # Display previous image results if they exist
    if st.session_state.image_results:
        images_info = feed_images_by_pdf_index(st.session_state.image_results, pdf_folder="downloaded_pdfs")
        for info in images_info:
            st.write(f"PDF: {info['pdf_name']}, Page: {info['page_number']}")
            st.image(info["image"], caption="Extracted PDF Page")
                    




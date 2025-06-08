# text_processing.py

import os
import json
import glob
import re
import subprocess
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def sanitize_filename(name):
    # Remove any character that is not alphanumeric, dash, or underscore.
    name = re.sub(r'[^\w\-]', '_', name)
    return name.strip("_") or "unknown_product"

def find_html_links(data):
    """Recursively searches for strings that contain '.html'."""
    links = []
    if isinstance(data, dict):
        for value in data.values():
            links.extend(find_html_links(value))
    elif isinstance(data, list):
        for item in data:
            links.extend(find_html_links(item))
    elif isinstance(data, str):
        if ".html" in data.lower():
            links.append(data)
    return links

def download_html_curl(link, output_filename):
    """Downloads HTML content using curl with a Firefox user-agent."""
    print(f"Downloading HTML from: {link}")
    cmd = [
        "curl",
        "-A", "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:101.0) Gecko/20100101 Firefox/101.0",
        link,
        "-o", output_filename
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error downloading {link}: {result.stderr}")
    else:
        print(f"Downloaded HTML to {output_filename}")

def extract_structured_text(html_content):
    """Extracts text from HTML while preserving sections based on headers."""
    soup = BeautifulSoup(html_content, "html.parser")
    body = soup.body if soup.body else soup
    headers = body.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    sections = []
    
    # Capture text before the first header as "Introduction"
    if headers:
        pre_header_text = []
        for elem in headers[0].previous_siblings:
            if isinstance(elem, str):
                pre_header_text.append(elem.strip())
            elif elem.get_text(strip=True):
                pre_header_text.append(elem.get_text(strip=True))
        pre_text = "\n".join(pre_header_text).strip()
        if pre_text:
            sections.append("Introduction\n" + "-" * 12 + "\n" + pre_text)
    
    # Process each header and the text following it until the next header
    for header in headers:
        header_text = header.get_text(strip=True)
        content_parts = []
        for sibling in header.next_siblings:
            if sibling.name and sibling.name.lower() in ['h1','h2','h3','h4','h5','h6']:
                break
            if isinstance(sibling, str):
                content_parts.append(sibling.strip())
            elif sibling.get_text(strip=True):
                content_parts.append(sibling.get_text(strip=True))
        content = "\n".join(content_parts).strip()
        section = f"{header_text}\n{'-' * len(header_text)}\n{content}"
        sections.append(section)
    
    return "\n\n".join(sections)

def process_single_json_file(json_file, html_folder, text_folder):
    """Processes one JSON file: extracts product name, downloads HTML,
       extracts structured text, and saves outputs."""
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return json_file, f"Error reading file: {e}"
    
    # Get product name or fallback to JSON file name.
    product_name = data.get("Product", {}).get("product", "")
    if not product_name:
        product_name = os.path.splitext(os.path.basename(json_file))[0]
    safe_product_name = sanitize_filename(product_name)
    
    # Extract HTML links and filter for "analog.com"
    html_links = find_html_links(data)
    filtered_links = [link for link in html_links if "analog.com" in link.lower()]
    if not filtered_links:
        return json_file, "No relevant HTML link found."
    
    # Save the extracted links into a .links file.
    links_filename = os.path.join(text_folder, safe_product_name + ".links")
    try:
        with open(links_filename, "w", encoding="utf-8") as f:
            for link in filtered_links:
                f.write(link + "\n")
        print(f"Saved HTML links for {product_name} to {links_filename}")
    except Exception as e:
        print(f"Error saving HTML links for {product_name}: {e}")
    
    # Download the HTML using the first filtered link.
    link = filtered_links[0]
    html_filename = os.path.join(html_folder, safe_product_name + ".html")
    download_html_curl(link, html_filename)
    
    try:
        with open(html_filename, "r", encoding="utf-8") as f:
            html_content = f.read()
    except Exception as e:
        return json_file, f"Error reading downloaded HTML: {e}"
    
    # Extract structured text and save it.
    structured_text = extract_structured_text(html_content)
    text_filename = os.path.join(text_folder, safe_product_name + ".txt")
    try:
        with open(text_filename, "w", encoding="utf-8") as f:
            f.write(structured_text)
        print(f"Saved structured text for {product_name} to {text_filename}")
    except Exception as e:
        print(f"Error saving structured text for {product_name}: {e}")
    
    return json_file, f"Processed successfully for product '{product_name}' with {len(filtered_links)} link(s) extracted."

def process_all_files(json_folder, html_folder, text_folder):
    """Processes all JSON files in the specified folder concurrently."""
    json_files = glob.glob(os.path.join(json_folder, "*.js"))
    if not json_files:
        print("No JSON files found in", json_folder)
        return
    
    results = []
    with ThreadPoolExecutor(max_workers=40) as executor:
        futures = {executor.submit(process_single_json_file, json_file, html_folder, text_folder): json_file 
                   for json_file in json_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing JSON files"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append((futures[future], f"Error: {e}"))
    
    for json_file, message in results:
        print(f"{json_file}: {message}")

if __name__ == "__main__":
    # Adjust these folder paths as needed.
    json_folder = "hackathon_datasets/problem_2_analog_products_data_and_docs/en-js/"
    html_folder = "downloaded_html"
    text_folder = "extracted_text"
    ensure_folder(html_folder)
    ensure_folder(text_folder)
    process_all_files(json_folder, html_folder, text_folder)


# query_vector.py

import requests
import torch
# CORRECTED IMPORT: Using the specific, recommended classes from Hugging Face Transformers.
from transformers import ColPaliForRetrieval, ColPaliProcessor


AZURE_OPENAI_ENDPOINT = "https://engassist-eus-dev-aais.openai.azure.com/openai/deployments/hackathon-emb-emb3l-team-21-cgcwn/embeddings?api-version=2023-05-15"
AZURE_OPENAI_API_KEY = "de7a14848db7462c9783adbcfbb0b430"  # Replace with your actual API key

def get_query_embedding(query_text: str) -> list:
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY
    }
    payload = {"input": [query_text]}
    response = requests.post(AZURE_OPENAI_ENDPOINT, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        return result["data"][0]["embedding"]
    else:
        raise Exception(f"Error obtaining embedding: {response.status_code} - {response.text}")

def load_colpali_model_and_processor(cache_dir="./"):
    """
    Load the Colpali model and its processor from Hugging Face using the correct classes.
    Returns a tuple: (model, processor)
    """
    # CORRECTED MODEL NAME: Using the specific Hugging Face identifier.
    model_name = "vidore/colpali-v1.2-hf"
    
    # CORRECTED CLASS USAGE: Using ColPaliForRetrieval as required.
    model = ColPaliForRetrieval.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    ).eval()
    
    # CORRECTED CLASS USAGE: Using ColPaliProcessor.
    processor = ColPaliProcessor.from_pretrained(model_name)
    
    return model, processor

def get_image_query_embedding(query_text, model, processor):
    """
    Computes a query embedding for the given text query using the Colpali model.
    """
    # Process the text query
    inputs = processor(text=[query_text], return_tensors="pt")
    
    with torch.no_grad():
        # Get the text features (embedding)
        query_embedding = model.get_text_features(**inputs)
        
    # Squeeze to remove batch dimension and convert to list
    return query_embedding.squeeze(0).tolist()

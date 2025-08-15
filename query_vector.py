# query_vector.py

import requests
import torch
# CORRECTED IMPORT: Using the official ColPali classes as per your findings.
from colpali_engine.models import ColPali
from colpali_engine.processor import ColPaliProcessor


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
    Load the Colpali model and its processor.
    Returns a tuple: (model, processor)
    """
    # CORRECTED CLASS NAME: Using ColPali instead of ColIdefics3
    model = ColPali.from_pretrained(
        "vidore/colSmol-256M",
        torch_dtype=torch.float32,
        attn_implementation="eager",
        cache_dir=cache_dir
    ).eval()
    # CORRECTED CLASS NAME: Using ColPaliProcessor instead of ColIdefics3Processor
    processor = ColPaliProcessor.from_pretrained(
        "vidore/colSmol-256M",
        cache_dir=cache_dir
    )
    return model, processor

def get_image_query_embedding(query_text, model, processor):
    """
    Computes a query embedding for the given query text using the Colpali model and processor.
    
    Parameters:
      - query_text (str): The input query text.
      - model: Loaded ColPali model.
      - processor: Loaded ColPaliProcessor.
    
    Returns:
      - List[float]: The query embedding as a list of floats.
    """
    batch_query = processor.process_queries([query_text]).to("cpu")
    with torch.no_grad():
        query_embedding = model(**batch_query)
    # Average the embeddings over the sequence dimension
    query_vector = query_embedding.mean(dim=1).squeeze(0)
    return query_vector.tolist()

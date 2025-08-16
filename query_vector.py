# query_vector.py

import requests
from sentence_transformers import SentenceTransformer
from PIL import Image

AZURE_OPENAI_ENDPOINT = "https://engassist-eus-dev-aais.openai.azure.com/openai/deployments/hackathon-emb-emb3l-team-21-cgcwn/embeddings?api-version=2023-05-15"
AZURE_OPENAI_API_KEY = "de7a14848db7462c9783adbcfbb0b430"

def get_query_embedding(query_text: str) -> list:
    """
    Generates a text embedding using the Azure OpenAI service.
    """
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

def load_image_embedding_model(cache_dir="./"):
    """
    Loads a lightweight CLIP model for creating embeddings from both images and text.
    """
    # Using a standard, effective CLIP model from sentence-transformers.
    # This is much more memory-efficient than the large ColPali model.
    model_name = "clip-ViT-B-32"
    model = SentenceTransformer(model_name, cache_folder=cache_dir)
    # The processor is not needed for this implementation.
    return model, None

def get_image_query_embedding(query_text, model, processor=None):
    """
    Computes a query embedding for the given text query using the loaded CLIP model.
    The processor argument is kept for API consistency but is not used.
    """
    # The sentence-transformer CLIP model can directly encode text into a compatible vector space.
    query_embedding = model.encode([query_text], normalize_embeddings=True)
    return query_embedding[0].tolist()

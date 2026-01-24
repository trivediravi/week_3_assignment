#!/usr/bin/env python3
"""
One-time script to embed all chunks and upload to Qdrant Cloud.
Run this ONCE before using the RAG pipeline.

Usage:
    python upload_to_qdrant.py
"""

import json
import os
import requests
import numpy as np
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models

from dotenv import load_dotenv
load_dotenv()

# Configuration
COLLECTION_NAME = "siggraph2025_papers"
CHUNKS_PATH = "./chunks.json"
BATCH_SIZE = 50  # How many chunks to embed per API call (OpenRouter limit)
EMBEDDING_MODEL = "openai/text-embedding-3-large"  # Via OpenRouter
VECTOR_SIZE = 3072  # Dimension for text-embedding-3-large


def load_chunks(path: str) -> list[dict]:
    """
    Load chunks from the JSON file.
    
    TODO:
    1. Open the file at `path` using open() with encoding='utf-8'
    2. Parse JSON using json.load()
    3. Return the list of chunks (hint: data["chunks"])
    
    Returns:
        List of chunk dictionaries
    """
    pass


def get_embeddings_batch(texts: list[str], api_key: str) -> list[list[float]]:
    """
    Get embeddings for multiple texts in one API call.
    
    TODO:
    1. Build headers dict with:
       - "Authorization": f"Bearer {api_key}"
       - "Content-Type": "application/json"
    
    2. Build payload dict with:
       - "model": EMBEDDING_MODEL
       - "input": texts  (this is the list of texts)
    
    3. Make POST request to "https://openrouter.ai/api/v1/embeddings"
       using requests.post(url, headers=headers, json=payload)
    
    4. Check response.status_code == 200, raise error if not
    
    5. Parse response: data = response.json()
    
    6. Extract embeddings: [item["embedding"] for item in data["data"]]
    
    7. Return the list of embeddings
    
    Args:
        texts: List of text strings to embed
        api_key: OpenRouter API key
        
    Returns:
        List of embedding vectors (each is a list of floats)
    """
    pass


def create_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int):
    """
    Create a Qdrant collection if it doesn't exist.
    
    TODO:
    1. Check if collection exists:
    
    2. Create the collection:
    
    3. Print success message
    
    Args:
        client: QdrantClient instance
        collection_name: Name for the collection
        vector_size: Dimension of vectors (3072 for text-embedding-3-large)
    """
    pass


def upload_chunks_to_qdrant(
    client: QdrantClient,
    collection_name: str,
    chunks: list[dict],
    api_key: str,
    batch_size: int = 50
):
    """
    Embed all chunks and upload to Qdrant in batches.
    
    TODO:
    1. Calculate total batches: total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    2. Loop through chunks in batches using range(0, len(chunks), batch_size)
       Use tqdm for progress bar: for i in tqdm(range(...), desc="Uploading"):
    
    3. For each batch:
       a. Get the batch slice: batch = chunks[i:i + batch_size]
       
       b. Extract texts for each chunk in the batch
       
       c. Get embeddings: embeddings = get_embeddings_batch(texts, api_key)
       
       d. Create Qdrant points - for each chunk and embedding pair:
          points = [
              models.PointStruct(
                  id=chunk["chunk_id"],  # Use chunk_id as the unique point ID
                  vector=embedding,
                  payload={
                      "chunk_id": chunk["chunk_id"],
                      "paper_id": chunk["paper_id"],
                      "title": chunk["title"],
                      "authors": chunk["authors"],
                      "text": chunk["text"],
                      "chunk_type": chunk["chunk_type"],
                      "chunk_section": chunk.get("chunk_section", ""),
                      "pdf_url": chunk.get("pdf_url"),
                      "github_link": chunk.get("github_link"),
                      "video_link": chunk.get("video_link"),
                      "acm_url": chunk.get("acm_url"),
                      "abstract_url": chunk.get("abstract_url"),
                  }
              )
              for chunk, embedding in zip(batch, embeddings)
          ]
       
       e. Upload to Qdrant: client.upsert(collection_name=collection_name, points=points)
    
    4. Print completion message with total chunks uploaded
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection
        chunks: List of chunk dictionaries
        api_key: OpenRouter API key
        batch_size: Number of chunks per batch
    """
    pass


def main():
    """Main function to run the upload process."""
    
    # Step 1: Load environment variables
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    # TODO: Step 2 - Validate environment variables
    # Check that all three variables are set (not None or empty)
    # If any are missing, print an error message and exit
    # Example: if not openrouter_api_key: raise ValueError("OPENROUTER_API_KEY not set")
    
    # TODO: Step 3 - Initialize Qdrant client
    
    # TODO: Step 4 - Load chunks from JSON file
    # chunks = load_chunks(CHUNKS_PATH)
    # print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")
    
    # TODO: Step 5 - Create collection
    # create_qdrant_collection(client, COLLECTION_NAME, VECTOR_SIZE)
    
    # TODO: Step 6 - Upload all chunks with embeddings
    # upload_chunks_to_qdrant(client, COLLECTION_NAME, chunks, openrouter_api_key, BATCH_SIZE)
    
    print("Done! Your vectors are now in Qdrant Cloud.")
    print(f"Collection: {COLLECTION_NAME}")
    print("You can verify in your Qdrant Cloud dashboard.")


if __name__ == "__main__":
    main()

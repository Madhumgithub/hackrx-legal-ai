# search.py

import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

# Load the model once
model = SentenceTransformer('all-MiniLM-L6-v2')

# Global variables to store index and associated chunks
faiss_index = None
stored_chunks = []

def split_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks for better retrieval.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def embed_text_chunks(chunks: List[str]) -> np.ndarray:
    """
    Convert text chunks into vector embeddings.
    """
    return model.encode(chunks)

def create_faiss_index(chunks: List[str]):
    """
    Create FAISS index from text chunks and store them.
    """
    global faiss_index, stored_chunks
    embeddings = embed_text_chunks(chunks)
    dimension = embeddings.shape[1]

    # Create FAISS index
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    faiss_index = index
    stored_chunks = chunks

def search_similar_chunks(query: str, top_k: int = 1) -> List[str]:
    """
    Perform semantic search and return top-k similar chunks.
    """
    global faiss_index, stored_chunks

    if faiss_index is None or not stored_chunks:
        raise ValueError("Index is not initialized. Please upload and index a document first.")

    query_embedding = model.encode([query])
    distances, indices = faiss_index.search(np.array(query_embedding), top_k)

    results = []
    for i in indices[0]:
        if i < len(stored_chunks):
            results.append(stored_chunks[i])

    return results

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def create_faiss_index(chunks):
    embeddings = MODEL.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    os.makedirs("faiss_index", exist_ok=True)
    faiss.write_index(index, "faiss_index/index.faiss")

    with open("faiss_index/chunks.txt", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c + "\n")

def search(query, top_k=3):
    index = faiss.read_index("faiss_index/index.faiss")
    with open("faiss_index/chunks.txt", "r", encoding="utf-8") as f:
        chunks = f.readlines()

    q_embedding = MODEL.encode([query])
    D, I = index.search(np.array(q_embedding), top_k)

    results = []
    for i in I[0]:
        results.append(chunks[i].strip())

    return results

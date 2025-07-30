from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load your embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to embed text chunks
def embed_chunks(chunks):
    return model.encode(chunks)

# Function to perform a similarity search
def query_documents(query, texts, index, chunk_embeddings):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=3)  # top 3 results
    results = [texts[i] for i in I[0]]
    return results

# Function to build FAISS index
def build_faiss_index(chunk_embeddings):
    dim = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(chunk_embeddings)
    return index

"""
Query script — test retrieval against the index built by build_index.py.
Run this after build_index.py has finished.

Install: pip install sentence-transformers faiss-cpu
"""

import pickle
import faiss
from sentence_transformers import SentenceTransformer

INDEX_DIR = "vector_index"
TOP_K = 5

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index(f"{INDEX_DIR}/filings.index")

with open(f"{INDEX_DIR}/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)


def search(query: str, k: int = TOP_K):
    query_vector = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_vector, k)

    print(f'\nQuery: "{query}"\n')
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        chunk = chunks[idx]
        print(f"#{rank} [{chunk['source']}] distance={dist:.3f}")
        print(f"    {chunk['text'][:300]}...\n")


if __name__ == "__main__":
    # try a few different queries to see retrieval quality across companies
    search("What are the risks related to supply chain and suppliers?")
    search("What competition risks does the company face?")
    search("What cybersecurity risks are disclosed?")

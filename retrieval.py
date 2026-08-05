"""
retrieval.py — shared retrieval + generation logic, used by the FastAPI app.

Keeping this separate from the API layer means the retrieval code isn't
tied to FastAPI at all — it could be reused in a CLI tool, a batch job,
or swapped into a different web framework without changes.
"""

import json
import os
import pickle
import re
import anthropic
import faiss
from sentence_transformers import SentenceTransformer

INDEX_DIR = "vector_index"
MODEL_NAME = "all-MiniLM-L6-v2"

# Generation via the direct Anthropic API (not Bedrock). Bedrock InvokeModel
# access for this account was stuck in a 24+ hour pending quota-approval
# queue with AWS Support with no ETA, so this uses the Anthropic API
# directly instead — same model family, same prompt/response shape, just a
# different client. Swapping back to Bedrock later (once/if that quota
# clears) would only require changing this client and generate_answer,
# not anything else in the retrieval pipeline.
GENERATION_MODEL = "claude-haiku-4-5-20251001"


class FilingRetriever:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.read_index(f"{INDEX_DIR}/filings.index")
        with open(f"{INDEX_DIR}/chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        # Reads ANTHROPIC_API_KEY from the environment — never hardcode
        # the key in source. Set it locally with `export ANTHROPIC_API_KEY=...`
        # before running, or pass it into the container at deploy time.
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Export it before starting the app, e.g.: export ANTHROPIC_API_KEY=sk-ant-..."
            )
        self.client = anthropic.Anthropic(api_key=api_key)

    def search(self, query: str, k: int = 5, deduplicate: bool = True) -> list[dict]:
        """
        Returns the top-k most relevant chunks for a query.

        deduplicate=True (default): SEC filings often carry the exact same
        boilerplate risk-factor sentence forward unchanged across several
        quarterly filings. Without deduplication, a single distinctive
        sentence can occupy 4+ of the top 5 results, crowding out other
        relevant content. When True, we search for more candidates than
        requested (k*4) and drop chunks whose normalized text we've already
        seen, keeping the highest-ranked (most similar) occurrence of each.
        Set to False if you specifically want to see repetition across
        filings, or to compare how language changed quarter to quarter.
        """
        search_k = k * 4 if deduplicate else k
        query_vector = self.model.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = self.index.search(query_vector, search_k)

        results = []
        seen_texts = set()
        for idx, dist in zip(indices[0], distances[0]):
            chunk = self.chunks[idx]
            if deduplicate:
                normalized = re.sub(r"\s+", " ", chunk["text"]).strip().lower()
                if normalized in seen_texts:
                    continue
                seen_texts.add(normalized)

            results.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "distance": float(dist),
            })
            if len(results) >= k:
                break

        return results

    def build_prompt(self, query: str, retrieved_chunks: list[dict]) -> str:
        """Constructs the prompt that would be sent to an LLM for generation."""
        context = "\n\n".join(
            f"[{c['source']}]\n{c['text']}" for c in retrieved_chunks
        )
        return f"""You are a financial analyst assistant. Answer the question using ONLY the
context below. If the context doesn't contain the answer, say so. Cite the source
for each claim.

Context:
{context}

Question: {query}

Answer:"""

    def generate_answer(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Sends the constructed RAG prompt to Claude Haiku 4.5 via the direct
        Anthropic API and returns the generated answer text.
        """
        message = self.client.messages.create(
            model=GENERATION_MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

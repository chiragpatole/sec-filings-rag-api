"""
main.py — FastAPI backend for the SEC filings Q&A service.

Run locally with: uvicorn main:app --reload
Then open http://127.0.0.1:8000/docs for the interactive Swagger UI.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from retrieval import FilingRetriever

app = FastAPI(
    title="SEC Filings Q&A API",
    description="Retrieval-augmented Q&A over real SEC 10-K/10-Q filings.",
    version="0.1.0",
)

# Allows the browser-based UI (index.html, opened directly as a local file
# or served from anywhere) to call this API. Without this, browsers block
# the response with a CORS error even though the request itself succeeds
# server-side. allow_origins=["*"] is fine for a portfolio demo; a real
# production service would restrict this to a specific known frontend origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once at startup, not per-request — loading the embedding model and
# FAISS index takes a couple of seconds, which you do NOT want to repeat on
# every single API call.
retriever: FilingRetriever | None = None


@app.on_event("startup")
def load_retriever():
    global retriever
    retriever = FilingRetriever()


# --- request/response schemas ---
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, example="What supply chain risks does Apple disclose?")
    top_k: int = Field(5, ge=1, le=20)
    deduplicate: bool = Field(True, description="Collapse near-identical chunks repeated across filings")


class RetrievedChunk(BaseModel):
    text: str
    source: str
    distance: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    prompt: str  # exposed for transparency/debugging — shows exactly what the LLM saw


# --- endpoints ---
@app.get("/health")
def health():
    """Basic liveness check — used by AWS/Kubernetes to know the service is up."""
    return {"status": "ok", "index_loaded": retriever is not None}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Retrieve the most relevant filing chunks for a question, construct the
    RAG prompt, and generate a grounded answer via Bedrock (Claude Haiku 4.5).
    """
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    chunks = retriever.search(request.question, k=request.top_k, deduplicate=request.deduplicate)
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant chunks found")

    prompt = retriever.build_prompt(request.question, chunks)

    try:
        answer = retriever.generate_answer(prompt)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Bedrock generation failed: {e}")

    return QueryResponse(
        question=request.question,
        answer=answer,
        retrieved_chunks=chunks,
        prompt=prompt,
    )


@app.post("/ingest")
def ingest():
    """
    Placeholder for now. In Phase 1/2 we ran ingestion and indexing as
    standalone scripts (edgar_ingest.py, build_index.py) — that's normal
    for a portfolio project and keeps the heavy one-off work out of the
    API's request path. This endpoint is a stub so the API surface matches
    the job description's expectations (an ingest endpoint exists), and
    we can wire it to trigger the real pipeline as a background job later
    if you want that.
    """
    raise HTTPException(status_code=501, detail="Not implemented yet — see build_index.py for now")

# Base image: slim Python, keeps the final image smaller than the full python image
FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies first (before copying code) so Docker can
# cache this layer — as long as requirements.txt doesn't change, rebuilds
# after a code change won't re-run this slow step.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model at BUILD time, not at container startup.
# Without this, every container start would need internet access to pull
# ~90MB from Hugging Face — slow, and a hard failure in an environment
# with restricted network access (e.g. certain AWS setups). Baking the
# model into the image means the container works offline once built.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Now copy the application code and the prebuilt vector index
COPY main.py retrieval.py ./
COPY vector_index/ ./vector_index/

EXPOSE 8000

# --host 0.0.0.0 is required (not 127.0.0.1) so the API is reachable from
# outside the container, not just from within it
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

"""
Phase 2 — Clean, chunk, embed, and index the filings downloaded in Phase 1.

Reads sec_filings_raw/manifest.json (produced by edgar_ingest.py), processes
each filing, and builds a local FAISS vector index you can query.

Install: pip install beautifulsoup4 sentence-transformers faiss-cpu
"""

import json
import os
import re
import pickle
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

MANIFEST_PATH = "sec_filings_raw/manifest.json"
INDEX_DIR = "vector_index"
CHUNK_MIN_LENGTH = 100  # skip tiny fragments (headers, table cells, etc.)
CHUNK_MAX_WORDS = 200   # split long paragraphs further


# --- Step 1: strip HTML down to clean text, block by block ---
def get_blocks(filepath: str):
    """
    Returns BeautifulSoup's block-level elements (p, div, td, li, headings)
    with scripts/styles removed. We work block-by-block rather than calling
    get_text() on the whole document, because SEC filings often split a
    single word across multiple sibling <span> tags for styling reasons
    (e.g. MSFT's "RISK FACTORS" was literally two spans: "...RIS" + "K
    FACTORS..."). Calling get_text(separator="\\n") on the whole soup
    inserts a break between EVERY tag, including those two spans — silently
    splitting "RISK" into "RIS" + "K" and breaking any regex looking for
    the word. Extracting text per block (joining spans with no separator)
    avoids that, while still giving us paragraph boundaries between blocks.
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.find_all(["p", "div", "td", "li", "h1", "h2", "h3", "h4"])


def blocks_to_text(blocks) -> str:
    """
    Joins block-level elements into paragraph lines, but with one important
    fix: filings frequently break a single sentence across multiple
    sibling HTML blocks (e.g. a <div> ending mid-sentence, continued in
    the next <div>) — purely for layout/formatting reasons, with no
    relation to actual paragraph structure. Treating every block as its
    own line meant chunk_text() was later splitting real sentences in
    half (confirmed: one chunk ended "...preclude the Company from
    selling certain products or services" and the very next chunk began
    "and expose the Company to significant licensing costs...").

    Fix: only start a new line when the previous block actually ended
    with sentence-ending punctuation (. ! ? or a colon/semicolon). If it
    didn't, glue it onto the next block's text instead of breaking there.
    """
    merged_lines = []
    buffer = ""
    for b in blocks:
        txt = b.get_text(separator="").replace("\xa0", " ")
        txt = re.sub(r"[ \t]+", " ", txt).strip()
        if not txt:
            continue

        buffer = f"{buffer} {txt}".strip() if buffer else txt
        # fix spans glued together with no space, e.g. "customers.Our products"
        buffer = re.sub(r"([.!?:;])(?=[A-Z])", r"\1 ", buffer)

        if re.search(r"[.!?:;]\s*$", buffer):
            merged_lines.append(buffer)
            buffer = ""
        # otherwise: keep accumulating — this block's text didn't end a
        # sentence, so the next block is very likely its continuation

    if buffer:  # flush anything left over at the end
        merged_lines.append(buffer)

    return "\n".join(merged_lines)


def is_numeric_table_chunk(text: str) -> bool:
    """
    Heuristic filter: financial statement tables, once flattened to text,
    are dense with digits/currency symbols and short "words" (numbers),
    unlike real prose. Flag anything where digits dominate the content.
    """
    digit_chars = sum(c.isdigit() for c in text)
    if len(text) == 0:
        return True
    digit_ratio = digit_chars / len(text)
    return digit_ratio > 0.15


# --- Step 2: isolate the Risk Factors section ---
def extract_risk_factors(blocks) -> str:
    """
    Finds the real 'Item 1A. Risk Factors' section heading by looking for a
    block whose ENTIRE content is just that phrase and nothing else.

    This is more robust than matching on capitalization or exact spacing,
    because each filer styles this heading differently — Tesla and
    Microsoft use ALL CAPS, Apple uses mixed case with multiple non-
    breaking spaces between "Item 1A." and "Risk Factors". What's
    consistent across all of them: the real heading is the ONLY content
    in its block, whereas every other mention of "Risk Factors" in the
    document (cross-references like "as discussed under Item 1A. Risk
    Factors", or the Table of Contents, which usually splits the item
    number and title into separate table cells) has other text alongside
    it in the same block.
    """
    heading_pattern = re.compile(r"^item\s+1a\.?\s+risk\s+factors$", re.IGNORECASE)
    end_pattern = re.compile(r"^item\s+1b\.?\s+unresolved\s+staff\s+comments", re.IGNORECASE)

    start_idx = None
    for i, b in enumerate(blocks):
        txt = re.sub(r"\s+", " ", b.get_text(separator="").replace("\xa0", " ")).strip()
        if heading_pattern.match(txt):
            start_idx = i
            break

    if start_idx is None:
        print("  WARNING: could not find standalone Risk Factors heading block — using full text")
        return blocks_to_text(blocks)

    end_idx = len(blocks)
    for i in range(start_idx + 1, len(blocks)):
        txt = re.sub(r"\s+", " ", blocks[i].get_text(separator="").replace("\xa0", " ")).strip()
        if end_pattern.match(txt):
            end_idx = i
            break

    return blocks_to_text(blocks[start_idx:end_idx])


def split_long_paragraph(para: str, max_words: int) -> list[str]:
    """
    Splits a long paragraph into pieces at SENTENCE boundaries, grouping
    consecutive sentences until adding the next one would exceed
    max_words. This replaces raw word-count slicing, which was cutting
    chunks mid-sentence (confirmed: a 200-word slice ended "...preclude
    the Company from selling certain products or services" with the next
    chunk starting "and expose the Company to significant licensing
    costs..." — the same sentence, torn in half).

    The sentence split itself is a simple heuristic (split after ./!/?
    followed by a space and a capital letter) — not a full NLP sentence
    tokenizer, but sufficient for this filing text and far better than
    splitting on word count alone.
    """
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", para)
    pieces = []
    current = []
    current_word_count = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())
        if current and current_word_count + sentence_words > max_words:
            pieces.append(" ".join(current))
            current = [sentence]
            current_word_count = sentence_words
        else:
            current.append(sentence)
            current_word_count += sentence_words

    if current:
        pieces.append(" ".join(current))

    return pieces


# --- Step 3: chunk into paragraph-sized pieces ---
def chunk_text(text: str, source_label: str) -> list[dict]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    for para in paragraphs:
        if len(para) < CHUNK_MIN_LENGTH:
            continue  # skip short fragments (likely headers or noise)
        if is_numeric_table_chunk(para):
            continue  # skip flattened financial tables (digit-heavy)
        words = para.split()
        if len(words) <= CHUNK_MAX_WORDS:
            chunks.append({"text": para, "source": source_label})
        else:
            for piece in split_long_paragraph(para, CHUNK_MAX_WORDS):
                chunks.append({"text": piece, "source": source_label})
    return chunks


def main():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    all_chunks = []

    for entry in manifest:
        print(f"Processing {entry['ticker']} {entry['form']} ({entry['filingDate']})...")
        blocks = get_blocks(entry["localPath"])
        risk_text = extract_risk_factors(blocks)
        source_label = f"{entry['ticker']} {entry['form']} {entry['filingDate']}"
        chunks = chunk_text(risk_text, source_label)
        print(f"  -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks across all filings: {len(all_chunks)}")

    # --- Step 4: embed every chunk ---
    print("Embedding chunks...")
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # --- Step 5: build and save a FAISS index ---
    os.makedirs(INDEX_DIR, exist_ok=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype("float32"))

    faiss.write_index(index, os.path.join(INDEX_DIR, "filings.index"))
    with open(os.path.join(INDEX_DIR, "chunks.pkl"), "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"\nDone. Index saved to {INDEX_DIR}/filings.index")
    print(f"Chunk metadata saved to {INDEX_DIR}/chunks.pkl")


if __name__ == "__main__":
    main()  

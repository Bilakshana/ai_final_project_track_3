"""
build_index.py
--------------
Step 2 of the pipeline: turn text chunks into vector embeddings
and store them in a FAISS index for fast similarity search.

Run once before using the chatbot:
    python build_index.py

This saves two files to ./vectorstore/:
    - faiss_index.bin   (the vector index)
    - chunks.json       (the original text chunks, for retrieval display)
"""

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from preprocess import load_documents, chunk_documents


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

DATA_DIR    = "./data"
INDEX_PATH  = "./vectorstore/faiss_index.bin"
CHUNKS_PATH = "./vectorstore/chunks.json"

# all-MiniLM-L6-v2: fast, lightweight, 384-dim embeddings, great for Q&A tasks
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


# ─────────────────────────────────────────────
# 1. Generate Embeddings
# ─────────────────────────────────────────────

def generate_embeddings(chunks: list, model: SentenceTransformer) -> np.ndarray:
    """
    Encode each chunk's text into a dense vector embedding.

    normalize_embeddings=True ensures vectors have length 1,
    which allows us to use Inner Product as cosine similarity.

    Returns:
        np.ndarray of shape (num_chunks, 384)
    """
    texts = [chunk["text"] for chunk in chunks]
    print(f"  Embedding {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True  # Required for cosine similarity via FAISS IndexFlatIP
    )

    return np.array(embeddings).astype("float32")


# ─────────────────────────────────────────────
# 2. Build FAISS Index
# ─────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS flat index using Inner Product similarity.

    Since embeddings are L2-normalized, Inner Product == Cosine Similarity.
    IndexFlatIP does exact (brute-force) search — accurate but fast enough
    for datasets up to ~100k chunks.

    Args:
        embeddings: float32 array of shape (n_chunks, embedding_dim)

    Returns:
        A FAISS index with all vectors added
    """
    dim = embeddings.shape[1]  # Should be 384 for MiniLM
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"  FAISS index built: {index.ntotal} vectors, dimension={dim}")
    return index


# ─────────────────────────────────────────────
# 3. Save & Load Index
# ─────────────────────────────────────────────

def save_index(index: faiss.IndexFlatIP, chunks: list):
    """Persist the FAISS index and chunk metadata to disk."""
    os.makedirs("./vectorstore", exist_ok=True)

    faiss.write_index(index, INDEX_PATH)
    print(f"  Index saved → {INDEX_PATH}")

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"  Chunks saved → {CHUNKS_PATH}")


def load_index():
    """
    Load a previously saved FAISS index and its chunk metadata.

    Returns:
        (faiss.Index, list of chunk dicts)
    """
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"Index not found at {INDEX_PATH}. Run build_index.py first."
        )

    index = faiss.read_index(INDEX_PATH)

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"  Loaded index: {index.ntotal} vectors")
    print(f"  Loaded chunks: {len(chunks)}")
    return index, chunks


# ─────────────────────────────────────────────
# Main — run this to build the index
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  RAG Index Builder")
    print("=" * 55)

    # Step 1: Load and chunk documents
    print("\n[Step 1] Loading documents from ./data/ ...")
    documents = load_documents(DATA_DIR)
    chunks = chunk_documents(documents, chunk_size=500, chunk_overlap=50)

    # Step 2: Load embedding model and generate embeddings
    print(f"\n[Step 2] Loading embedding model: {EMBED_MODEL_NAME}")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = generate_embeddings(chunks, embed_model)

    # Step 3: Build FAISS index
    print("\n[Step 3] Building FAISS index ...")
    faiss_index = build_faiss_index(embeddings)

    # Step 4: Save to disk
    print("\n[Step 4] Saving index and chunks ...")
    save_index(faiss_index, chunks)

    print("\n✅  Done! Run rag_pipeline.py or app.py to start chatting.")
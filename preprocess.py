"""
preprocess.py
-------------
Handles loading documents (PDF or TXT), cleaning the text,
and splitting it into manageable chunks for embedding.

Run this directly to test: python preprocess.py
"""

import os
import re
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ─────────────────────────────────────────────
# 1. Document Loaders
# ─────────────────────────────────────────────

def load_pdf(file_path: str) -> str:
    """Extract all text from a PDF file, page by page."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


def load_text_file(file_path: str) -> str:
    """Read a plain .txt file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_documents(data_dir: str) -> list:
    """
    Scan a directory and load all .pdf and .txt files.

    Returns:
        List of dicts: [{"source": "filename.pdf", "text": "full text..."}, ...]
    """
    documents = []

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    files = os.listdir(data_dir)
    if not files:
        raise ValueError(f"No files found in: {data_dir}")

    for filename in files:
        filepath = os.path.join(data_dir, filename)

        if filename.endswith(".pdf"):
            print(f"  [PDF] Loading: {filename}")
            text = load_pdf(filepath)
            documents.append({"source": filename, "text": text})

        elif filename.endswith(".txt"):
            print(f"  [TXT] Loading: {filename}")
            text = load_text_file(filepath)
            documents.append({"source": filename, "text": text})

        else:
            print(f"  [SKIP] Unsupported file type: {filename}")

    print(f"\n  Total documents loaded: {len(documents)}")
    return documents


# ─────────────────────────────────────────────
# 2. Text Cleaning
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean raw extracted text:
    - Collapse multiple spaces and newlines into one space
    - Remove non-ASCII characters (common in PDF extraction artifacts)
    - Strip leading/trailing whitespace
    """
    text = re.sub(r'\s+', ' ', text)             # Normalize whitespace
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII chars
    text = text.strip()
    return text


# ─────────────────────────────────────────────
# 3. Text Chunking
# ─────────────────────────────────────────────

def chunk_documents(documents: list, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """
    Split each document's text into smaller overlapping chunks.

    Why chunking?
        Embedding models have token limits, and small focused chunks
        retrieve more relevant context than huge blocks of text.

    Why overlap?
        If a key sentence falls at the boundary between two chunks,
        overlap ensures it's captured in at least one chunk fully.

    Args:
        documents:    List of {"source": ..., "text": ...} dicts
        chunk_size:   Max characters per chunk (500 is a good default)
        chunk_overlap: Characters shared between adjacent chunks

    Returns:
        List of chunk dicts: [{"source", "chunk_id", "text"}, ...]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Try these separators in order — prefer natural boundaries
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    all_chunks = []

    for doc in documents:
        cleaned = clean_text(doc["text"])
        splits = splitter.split_text(cleaned)

        for i, split_text in enumerate(splits):
            all_chunks.append({
                "source":   doc["source"],
                "chunk_id": i,
                "text":     split_text
            })

    print(f"  Total chunks created: {len(all_chunks)}")
    return all_chunks


# ─────────────────────────────────────────────
# Quick test — run this file directly
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("Testing preprocessing pipeline")
    print("=" * 50)

    docs = load_documents("./data")
    chunks = chunk_documents(docs)

    print(f"\nSample chunk (first one):")
    print("-" * 40)
    print(chunks[0]["text"])
    print("-" * 40)
    print(f"Source: {chunks[0]['source']}")
    print(f"Chunk ID: {chunks[0]['chunk_id']}")
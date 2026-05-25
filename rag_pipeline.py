"""
rag_pipeline.py
---------------
The heart of the RAG system. This module:

  1. Encodes the user query into an embedding
  2. Searches the FAISS index for the most relevant document chunks
  3. Injects retrieved context into a prompt
  4. Sends the prompt to LLaMA 3 (via Groq API) and returns the answer

Run interactively:
    python rag_pipeline.py

Set your key first:
    export GROQ_API_KEY=your_key_here
Or get a free key at: https://console.groq.com
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

from build_index import load_index, EMBED_MODEL_NAME


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

LLM_MODEL   = "llama-3.1-8b-instant"  # Fast, free via Groq
TOP_K       = 5                        # Retrieve top 5 most relevant chunks
TEMPERATURE = 0.2                      # Low = more factual, less creative


# ─────────────────────────────────────────────
# RAG Chatbot Class
# ─────────────────────────────────────────────

class RAGChatbot:
    """
    A Retrieval-Augmented Generation chatbot.

    Architecture:
        Query → Embedder → FAISS Search → Context Builder → LLM → Answer

    Prompting strategy used: Zero-shot with explicit instructions.
    The model is told to answer ONLY from context — this prevents hallucination.
    """

    def __init__(self, groq_api_key: str):
        print("Loading embedding model ...")
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)

        print("Loading FAISS index ...")
        self.index, self.chunks = load_index()

        print("Connecting to Groq LLM ...")
        self.llm = Groq(api_key=groq_api_key)

        print("\nRAG Chatbot is ready!\n")


    # ──────────────────────────────────────────
    # Step 1: Retrieval
    # ──────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K) -> list:
        """
        Convert the query to an embedding and find the most
        semantically similar chunks in the FAISS index.

        Returns:
            List of chunk dicts with an added 'score' field (0 to 1)
        """
        # Encode query with same model used for indexing
        query_vector = self.embedder.encode(
            [query],
            normalize_embeddings=True
        ).astype("float32")

        # FAISS search: returns (scores, indices) arrays of shape (1, top_k)
        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue  # FAISS returns -1 when fewer results than top_k exist
            chunk = dict(self.chunks[idx])  # Make a copy
            chunk["score"] = round(float(score), 4)
            results.append(chunk)

        return results


    # ──────────────────────────────────────────
    # Step 2: Prompt Engineering
    # ──────────────────────────────────────────

    def build_prompt(self, query: str, context_chunks: list) -> str:
        """
        Construct a prompt using retrieved context + the user query.

        Strategy: Zero-shot prompting with grounded context.
        The system message enforces answer grounding to prevent hallucination.

        The context is formatted with source labels so the model (and user)
        knows which document each piece of information came from.
        """
        # Combine retrieved chunks into one context block
        context_parts = []
        for i, chunk in enumerate(context_chunks, start=1):
            context_parts.append(
                f"[Excerpt {i} — Source: {chunk['source']}]\n{chunk['text']}"
            )
        context_text = "\n\n".join(context_parts)

        # The final prompt sent to the LLM
        prompt = f"""You are a helpful and precise assistant that answers questions \
based strictly on the provided document excerpts.

Rules:
- Answer ONLY using information found in the excerpts below.
- If the answer is not in the excerpts, respond with:
  "I don't have enough information to answer that based on the provided documents."
- Be concise and direct. Do not repeat the question.
- Cite the source excerpt number when relevant (e.g., "According to Excerpt 2...").

---
DOCUMENT EXCERPTS:
{context_text}
---

QUESTION: {query}

ANSWER:"""
        return prompt


    # ──────────────────────────────────────────
    # Step 3: Generation
    # ──────────────────────────────────────────

    def answer(self, query: str) -> dict:
        """
        Full RAG pipeline: retrieve → prompt → generate → return.

        Returns:
            dict with keys: question, answer, sources
        """
        # Retrieve relevant context
        retrieved_chunks = self.retrieve(query)

        if not retrieved_chunks:
            return {
                "question": query,
                "answer": "No relevant documents found. Please check your index.",
                "sources": []
            }

        # Build the grounded prompt
        prompt = self.build_prompt(query, retrieved_chunks)

        # Call LLaMA 3 via Groq API
        response = self.llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a document-grounded QA assistant. "
                        "Never answer from general knowledge. "
                        "Only use the provided excerpts."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=TEMPERATURE,
            max_tokens=512
        )

        answer_text = response.choices[0].message.content.strip()

        # Return answer + source info for transparency
        return {
            "question": query,
            "answer":   answer_text,
            "sources": [
                {
                    "source":  c["source"],
                    "score":   c["score"],
                    "preview": c["text"][:200] + "..."
                }
                for c in retrieved_chunks
            ]
        }


# ─────────────────────────────────────────────
# Interactive CLI — run directly
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Load API key from environment or prompt user
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("No GROQ_API_KEY environment variable found.")
        api_key = input("Enter your Groq API key: ").strip()

    bot = RAGChatbot(groq_api_key=api_key)

    print("Ask questions about your documents.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        result = bot.answer(query)

        print(f"\nBot: {result['answer']}")
        print(f"\n  Sources used:")
        for s in result["sources"]:
            print(f"    [{s['score']:.3f}] {s['source']}")
        print("─" * 60 + "\n")
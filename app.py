"""
app.py
------
Streamlit web interface for the RAG Chatbot.

Features:
  - Upload your own PDF or TXT documents
  - Automatically chunks and indexes them on-the-fly
  - Multi-turn chat interface
  - Shows retrieved context chunks on the side
  - Configurable chunk size and top-k retrieval

Run with:
    streamlit run app.py
"""

import os
import tempfile
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq

from preprocess import load_pdf, load_text_file, clean_text, chunk_documents


# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Domain-Specific RAG Chatbot")
st.caption("Upload your documents. Ask questions. Get grounded answers.")


# ─────────────────────────────────────────────
# Sidebar: Configuration
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Setup")

    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get a free key at https://console.groq.com"
    )
    st.markdown("[🔑 Get a free Groq API key](https://console.groq.com)")

    st.divider()
    st.header("📁 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="You can upload multiple files. The chatbot will answer from all of them."
    )

    st.divider()
    st.header("🔧 Settings")

    top_k      = st.slider("Chunks to Retrieve (Top-K)", 1, 10, 5)
    chunk_size = st.slider("Chunk Size (characters)",  200, 1000, 500, step=100)
    chunk_overlap = st.slider("Chunk Overlap", 0, 100, 50, step=10)

    build_button = st.button("🔨 Build Index from Documents", use_container_width=True)


# ─────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────

for key, default in {
    "index_built":    False,
    "chat_history":   [],
    "last_retrieved": [],
    "embedder":       None,
    "faiss_index":    None,
    "chunks":         None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────
# Index Building Logic
# ─────────────────────────────────────────────

if build_button:
    if not groq_api_key:
        st.sidebar.error("❌ Please enter your Groq API key first.")

    elif not uploaded_files:
        st.sidebar.error("❌ Please upload at least one PDF or TXT file.")

    else:
        with st.spinner("📄 Processing documents and building vector index..."):
            try:
                # ── Load documents ──
                all_docs = []
                for file in uploaded_files:
                    suffix = ".pdf" if file.name.endswith(".pdf") else ".txt"

                    # Write to a temp file so our parsers can read it
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(file.read())
                        tmp_path = tmp.name

                    if suffix == ".pdf":
                        text = load_pdf(tmp_path)
                    else:
                        text = load_text_file(tmp_path)

                    os.unlink(tmp_path)  # Clean up temp file
                    all_docs.append({"source": file.name, "text": text})

                # ── Chunk documents ──
                chunks = chunk_documents(all_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

                # ── Generate embeddings ──
                embedder = SentenceTransformer("all-MiniLM-L6-v2")
                embeddings = embedder.encode(
                    [c["text"] for c in chunks],
                    batch_size=32,
                    normalize_embeddings=True,
                    show_progress_bar=False
                ).astype("float32")

                # ── Build FAISS index ──
                dim   = embeddings.shape[1]
                index = faiss.IndexFlatIP(dim)
                index.add(embeddings)

                # ── Store in session state ──
                st.session_state.embedder    = embedder
                st.session_state.faiss_index = index
                st.session_state.chunks      = chunks
                st.session_state.index_built = True

                st.sidebar.success(
                    f"✅ Index ready!\n"
                    f"{len(all_docs)} file(s) → {len(chunks)} chunks."
                )

            except Exception as e:
                st.sidebar.error(f"Build failed: {e}")


# ─────────────────────────────────────────────
# Main Chat Area
# ─────────────────────────────────────────────

left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("💬 Chat")

    # Show status
    if not st.session_state.index_built:
        st.info("👈 Upload documents and click **Build Index** to start.")
    else:
        st.success(f"Index ready — {len(st.session_state.chunks)} chunks loaded.")

    # Render existing chat messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input box
    user_query = st.chat_input("Ask a question about your documents...")

    if user_query:
        if not st.session_state.index_built:
            st.error("Please build the index first (sidebar → Build Index).")

        elif not groq_api_key:
            st.error("Please enter your Groq API key in the sidebar.")

        else:
            # Show user message immediately
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.write(user_query)

            # Generate and show assistant answer
            with st.chat_message("assistant"):
                with st.spinner("Searching documents and generating answer..."):
                    try:
                        # ── Retrieve relevant chunks ──
                        q_vec = st.session_state.embedder.encode(
                            [user_query],
                            normalize_embeddings=True
                        ).astype("float32")

                        scores, idxs = st.session_state.faiss_index.search(q_vec, top_k)

                        retrieved = []
                        for score, idx in zip(scores[0], idxs[0]):
                            if idx != -1:
                                chunk = dict(st.session_state.chunks[idx])
                                chunk["score"] = float(score)
                                retrieved.append(chunk)

                        # ── Build prompt ──
                        context_parts = [
                            f"[Excerpt {i+1} — {c['source']}]\n{c['text']}"
                            for i, c in enumerate(retrieved)
                        ]
                        context_str = "\n\n".join(context_parts)

                        prompt = f"""Answer the question below using ONLY the document excerpts provided.
If the answer is not in the excerpts, say: "I don't have enough information to answer that."

EXCERPTS:
{context_str}

QUESTION: {user_query}

ANSWER:"""

                        # ── Call LLaMA 3 via Groq ──
                        client = Groq(api_key=groq_api_key)
                        response = client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        "You are a precise assistant. "
                                        "Answer only from the provided document excerpts. "
                                        "Be concise and direct."
                                    )
                                },
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.2,
                            max_tokens=512
                        )

                        answer = response.choices[0].message.content.strip()
                        st.write(answer)

                        # Store for history and context panel
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": answer}
                        )
                        st.session_state.last_retrieved = retrieved

                    except Exception as e:
                        error_msg = f"Error: {e}"
                        st.error(error_msg)
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": error_msg}
                        )


# ─────────────────────────────────────────────
# Right Column: Retrieved Context Viewer
# ─────────────────────────────────────────────

with right_col:
    st.subheader("🔍 Retrieved Chunks")

    if st.session_state.last_retrieved:
        for i, chunk in enumerate(st.session_state.last_retrieved):
            score_color = "🟢" if chunk["score"] > 0.6 else "🟡" if chunk["score"] > 0.4 else "🔴"
            with st.expander(
                f"{score_color} Chunk {i+1} · {chunk['source']} · score: {chunk['score']:.3f}"
            ):
                st.caption(f"Source: {chunk['source']} | Chunk #{chunk.get('chunk_id', '?')}")
                st.write(chunk["text"])
    else:
        st.info("The most relevant document chunks will appear here after you ask a question.")


# ─────────────────────────────────────────────
# Footer Controls
# ─────────────────────────────────────────────

st.divider()
col_a, col_b = st.columns([1, 4])
with col_a:
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history   = []
        st.session_state.last_retrieved = []
        st.rerun()

with col_b:
    st.caption(
        "Built with sentence-transformers + FAISS + LLaMA 3 (Groq). "
        "Track 3 — RAG Chatbot Final Project."
    )
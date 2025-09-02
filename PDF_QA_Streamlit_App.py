import os
import io
import faiss
import numpy as np
import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from typing import List
from sentence_transformers import SentenceTransformer

# ---------------- PDF Utilities ----------------

def extract_text_from_pdf(pdf_file: io.BytesIO) -> str:
    """Extract text from an uploaded PDF."""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text.strip()


def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ---------------- Embedding Utilities ----------------

@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Return Hugging Face sentence transformer embedder (no API key needed)."""
    return SentenceTransformer(model_name)


def embed_chunks(embedder, chunks: List[str]) -> np.ndarray:
    """Embed document chunks locally."""
    embs = embedder.encode(chunks, convert_to_numpy=True)
    return np.asarray(embs, dtype=np.float32)


def embed_query(embedder, query: str) -> np.ndarray:
    """Embed query locally."""
    emb = embedder.encode([query], convert_to_numpy=True)
    return np.asarray(emb, dtype=np.float32)


# ---------------- FAISS Utilities ----------------

def build_faiss_index(embs: np.ndarray) -> faiss.IndexFlatL2:
    """Build a FAISS index from embeddings."""
    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embs)
    return index


def retrieve(query_emb: np.ndarray, index: faiss.IndexFlatL2, k: int = 3) -> List[int]:
    """Retrieve top-k similar chunks."""
    distances, indices = index.search(query_emb, k)
    return indices[0]


# ---------------- QA Utilities ----------------

def answer_query(query: str, chunks: List[str], retrieved_ids: List[int]) -> str:
    """Simple local QA: return relevant chunks as the answer."""
    context = "\n\n".join([chunks[i] for i in retrieved_ids if i < len(chunks)])
    return f"Based on the document, here’s what I found:\n\n{context}"


# ---------------- Streamlit App ----------------

def main():
    st.set_page_config(page_title="PDF Q&A App (Offline)", layout="wide")
    st.title("📄 PDF Question Answering App (No API Needed)")

    # Sidebar
    st.sidebar.header("Upload PDF")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF", type="pdf")

    # Session state
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
        st.session_state.chunk_rows = []
        st.session_state.index = None
        st.session_state.embedder_name = "sentence-transformers/all-MiniLM-L6-v2"
        st.session_state.ready = False

    # PDF Upload
    if uploaded_file is not None:
        with st.spinner("Extracting text..."):
            text = extract_text_from_pdf(uploaded_file)
            if text:
                chunks = split_into_chunks(text)
                st.session_state.chunks = chunks
                st.session_state.chunk_rows = [{"chunk_id": i, "text": c} for i, c in enumerate(chunks)]

                embedder = get_embedder(st.session_state.embedder_name)
                with st.spinner("Creating embeddings..."):
                    embs = embed_chunks(embedder, chunks)
                    index = build_faiss_index(embs)
                    st.session_state.index = index
                    st.session_state.ready = True
                st.success("✅ Document processed and ready for Q&A")
            else:
                st.error("No text found in PDF.")

    # QA Section
    if st.session_state.ready:
        st.subheader("Ask a Question about the PDF")
        query = st.text_input("Enter your question:")

        if query:
            embedder = get_embedder(st.session_state.embedder_name)
            query_emb = embed_query(embedder, query)
            retrieved_ids = retrieve(query_emb, st.session_state.index, k=3)

            with st.spinner("Finding answer..."):
                answer = answer_query(query, st.session_state.chunks, retrieved_ids)

            st.write("### Answer:")
            st.write(answer)

            with st.expander("View Retrieved Context"):
                for i in retrieved_ids:
                    if i < len(st.session_state.chunks):
                        st.write(f"**Chunk {i}:** {st.session_state.chunks[i]}")

    # Dataset Download
    if st.session_state.chunk_rows:
        df = pd.DataFrame(st.session_state.chunk_rows)
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.sidebar.download_button("Download Chunks Dataset", csv_data, "chunks.csv", "text/csv")


if __name__ == "__main__":
    main()

# app.py – Streamlit RAG-Chatbot mit Qdrant, BM25, BGE-M3, TEI/GML/RDF-Support, Auth-Gate

import os
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
import pickle, hashlib
from pathlib import Path

# ===== Auth-Gate =====
def auth_gate():
    allowed_tokens = [t.strip() for t in os.getenv("AUTH_TOKENS", "").split(",") if t.strip()]
    if not allowed_tokens:
        return
    if "auth_ok" not in st.session_state:
        st.session_state["auth_ok"] = False
    if not st.session_state["auth_ok"]:
        token = st.text_input("Access Token", type="password")
        if st.button("Login"):
            if token in allowed_tokens:
                st.session_state["auth_ok"] = True
                st.rerun()
            else:
                st.error("Ungültiger Token")
        st.stop()

# ===== Initialisierung =====
st.set_page_config(page_title="Privater Daten-Chatbot – Skalierbar", layout="wide")
auth_gate()

DATA_DIR = Path("./data")
INDEX_DIR = Path("./index")
INDEX_DIR.mkdir(exist_ok=True)
BM25_FILE = INDEX_DIR / "bm25.pkl"
DOCS_FILE = INDEX_DIR / "docs.pkl"

# Lade Modelle
embed_model = SentenceTransformer("BAAI/bge-m3")
reranker = CrossEncoder("BAAI/bge-reranker-base")

# Qdrant-Client
qdr = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"), api_key=os.getenv("QDRANT_API_KEY"))

# ===== Hilfsfunktionen =====
def load_bm25():
    if BM25_FILE.exists() and DOCS_FILE.exists():
        with open(BM25_FILE, "rb") as f:
            bm25 = pickle.load(f)
        with open(DOCS_FILE, "rb") as f:
            docs = pickle.load(f)
        return bm25, docs
    return None, []

bm25, docs = load_bm25()

def hybrid_search(query, top_k=5):
    results = []
    # BM25
    if bm25 and docs:
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        bm25_hits = sorted(zip(range(len(scores)), scores), key=lambda x: x[1], reverse=True)[:top_k]
        for idx, score in bm25_hits:
            results.append((docs[idx].text, score, docs[idx].meta))
    # Dense
    q_vec = embed_model.encode([query], normalize_embeddings=True)[0]
    hits = qdr.search(collection_name="docs_bge_m3", query_vector=q_vec, limit=top_k)
    for hit in hits:
        results.append((hit.payload["text"], hit.score, {"source": hit.payload.get("source")}))
    # Rerank
    rerank_in = [(query, r[0]) for r in results]
    rr_scores = reranker.predict(rerank_in)
    results = sorted(zip(results, rr_scores), key=lambda x: x[1], reverse=True)
    return results[:top_k]

# ===== UI =====
st.sidebar.title("Index verwalten")
uploaded_files = st.sidebar.file_uploader("Dateien hochladen", accept_multiple_files=True)
if uploaded_files:
    for f in uploaded_files:
        (DATA_DIR / f.name).write_bytes(f.read())
    st.sidebar.success(f"{len(uploaded_files)} Datei(en) gespeichert.")

if st.sidebar.button("🧱 Vollständiger Rebuild"):
    os.system("python ingest.py --full-rebuild")
    st.sidebar.success("Index neu aufgebaut.")
    st.experimental_rerun()

if st.sidebar.button("🔄 Inkrementelles Update"):
    os.system("python ingest.py")
    st.sidebar.success("Index aktualisiert.")
    st.experimental_rerun()

st.title("💬 Privater Daten-Chatbot")
user_query = st.text_input("Frage eingeben")
if st.button("Senden") and user_query:
    results = hybrid_search(user_query, top_k=5)
    if results:
        st.subheader("Antwort")
        context = "\n\n".join([r[0][0] for r in results])
        # LLM-Aufruf
        import requests
        oss_base = os.getenv("OSS_API_BASE")
        oss_key = os.getenv("OSS_API_KEY")
        oss_model = os.getenv("OSS_MODEL")
        headers = {"Authorization": f"Bearer {oss_key}"}
        payload = {
            "model": oss_model,
            "messages": [
                {"role": "system", "content": "Du bist ein hilfreicher Assistent und beantwortest Fragen nur auf Basis des bereitgestellten Kontexts."},
                {"role": "user", "content": f"Kontext:\n{context}\n\nFrage: {user_query}"}
            ]
        }
        resp = requests.post(f"{oss_base}/v1/chat/completions", headers=headers, json=payload)
        if resp.ok:
            st.write(resp.json()["choices"][0]["message"]["content"])
        else:
            st.error(f"LLM-Fehler: {resp.text}")
        with st.expander("🔎 Verwendete Ausschnitte"):
            for (chunk, score, meta), rr_score in results:
                st.write(f"Score: {rr_score:.3f} | Quelle: {meta.get('source')}")
                st.write(chunk)
                st.markdown("---")
    else:
        st.warning("Keine Treffer gefunden.")

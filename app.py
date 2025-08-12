# app.py – ErinnerungsBot Steiermark (RAG: Qdrant + BM25 + BGE-M3)
# Features: Rollen (Admin/Viewer), Repo-only Daten, Auto-Rebuild bei Datenänderung, CPU-only, OpenRouter-Header
from __future__ import annotations
import os
import time
import pickle
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
import numpy as np
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

# ---------- Seite ----------
st.set_page_config(page_title="🔒 ErinnerungsBot Steiermark", layout="wide")

# ---------- Robuste HF-Caches (Streamlit Cloud) ----------
HF_DIR = "/tmp/hf"
os.environ["HF_HOME"] = HF_DIR
os.environ["TRANSFORMERS_CACHE"] = str(Path(HF_DIR) / "transformers")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(Path(HF_DIR) / "sbert")
Path(HF_DIR).mkdir(parents=True, exist_ok=True)
Path(os.environ["TRANSFORMERS_CACHE"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["SENTENCE_TRANSFORMERS_HOME"]).mkdir(parents=True, exist_ok=True)

# ---------- CPU-only ----------
torch.set_num_threads(1)
DEVICE = "cpu"

# ---------- Pfade ----------
DATA_DIR = Path("./data")
INDEX_DIR = Path("./index"); INDEX_DIR.mkdir(exist_ok=True)
BM25_FILE = INDEX_DIR / "bm25.pkl"
DOCS_FILE = INDEX_DIR / "docs.pkl"

# ---------- Qdrant ----------
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = "docs_bge_m3"
qdr = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ---------- OpenAI-kompatible API (OpenRouter) ----------
OSS_API_BASE = os.getenv("OSS_API_BASE", "https://openrouter.ai/api")
OSS_API_KEY  = os.getenv("OSS_API_KEY", "")
OSS_MODEL    = os.getenv("OSS_MODEL", "openai/gpt-oss-20b:free")

# ---------- Rollen / Tokens ----------
def _split_tokens(value: str) -> list[str]:
    return [t.strip() for t in value.split(",") if t.strip()]

ADMIN_SET = set(_split_tokens(os.getenv("ADMIN_TOKENS", "")))
VIEW_SET  = set(_split_tokens(os.getenv("VIEW_TOKENS", "")))
ALL_SET   = set(_split_tokens(os.getenv("AUTH_TOKENS", "")))
if ALL_SET and not ADMIN_SET and not VIEW_SET:
    ADMIN_SET = ALL_SET  # Fallback: AUTH_TOKENS = Admin

# ---------- Systemprompt ----------
SYSTEM_PROMPT = (
    "Du bist ein präziser Assistent. Antworte ausschließlich mit Informationen, "
    "die im bereitgestellten KONTEXT enthalten sind. Wenn der Kontext keine Antwort zulässt, "
    "sage eindeutig: 'Dazu habe ich keine Information in meinen Daten.' Erfinde nichts."
)

# ---------- Auth ----------
def auth_gate() -> None:
    if not (ADMIN_SET or VIEW_SET):
        st.session_state["authed"] = True
        st.session_state["role"] = "admin"
        return
    if st.session_state.get("authed"):
        with st.sidebar:
            st.caption(f"Rolle: **{st.session_state.get('role', 'viewer')}**")
            if st.button("Logout"):
                st.session_state["authed"] = False
                st.session_state["role"] = None
                st.rerun()
        return
    st.title("🔐 Login")
    token = st.text_input("Access Token", type="password")
    if st.button("Anmelden"):
        if token in ADMIN_SET:
            st.session_state["authed"] = True
            st.session_state["role"] = "admin"
            st.rerun()
        elif token in VIEW_SET:
            st.session_state["authed"] = True
            st.session_state["role"] = "viewer"
            st.rerun()
        else:
            st.error("Ungültiger Token")
    st.stop()

auth_gate()

# ---------- Modelle (CPU, gecached) ----------
@st.cache_resource(show_spinner=True)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer("BAAI/bge-m3", device=DEVICE)

@st.cache_resource(show_spinner=True)
def get_reranker() -> CrossEncoder:
    try:
        return CrossEncoder("BAAI/bge-reranker-base", device=DEVICE)
    except Exception:
        return None

embed_model = get_embedder()
reranker = get_reranker()

# ---------- BM25 & Docs ----------
@dataclass
class DocChunk:
    text: str
    source: str
    chunk_id: int
    meta: Dict[str, str]

def load_bm25() -> Tuple[BM25Okapi | None, List[DocChunk]]:
    if BM25_FILE.exists() and DOCS_FILE.exists():
        with open(BM25_FILE, "rb") as f: bm25 = pickle.load(f)
        with open(DOCS_FILE, "rb") as f: docs = pickle.load(f)
        return bm25, docs
    return None, []

# ---------- Auto-Rebuild bei Datenänderung ----------
def data_dir_hash() -> str:
    m = hashlib.sha256()
    if DATA_DIR.exists():
        for path in sorted(DATA_DIR.rglob("*")):
            if path.is_file():
                m.update(path.name.encode())
                m.update(str(path.stat().st_mtime).encode())
                try:
                    m.update(path.read_bytes())
                except Exception:
                    pass
    return m.hexdigest()

hash_file = INDEX_DIR / ".data_hash"
current_hash = data_dir_hash()
previous_hash = hash_file.read_text() if hash_file.exists() else ""

if current_hash != previous_hash:
    with st.spinner("📚 Datenänderung erkannt – baue Index neu auf …"):
        os.system("python ingest.py --full-rebuild")
    hash_file.write_text(current_hash)
    st.success("Index wurde automatisch aktualisiert!")
    time.sleep(0.8)

bm25, docs = load_bm25()

# ---------- Suche ----------
def hybrid_search(query: str, top_k: int = 8) -> List[Tuple[str, float, Dict]]:
    results: List[Tuple[str, float, Dict]] = []
    if bm25 and docs:
        tq = query.lower().split()
        scores = bm25.get_scores(tq)
        top_idx = np.argsort(scores)[-top_k:][::-1]
        for i in top_idx:
            if scores[i] <= 0: continue
            d: DocChunk = docs[i]
            results.append((d.text, float(scores[i]), {"source": d.source, "chunk_id": d.chunk_id, "kind": "bm25"}))
    try:
        qv = embed_model.encode([query], normalize_embeddings=True)[0].tolist()
        hits = qdr.search(collection_name=QDRANT_COLLECTION, query_vector=qv, limit=top_k, with_payload=True)
        for h in hits:
            payload = h.payload or {}
            results.append((payload.get("text", ""), float(h.score),
                            {"source": payload.get("source"), "chunk_id": payload.get("chunk_id"), "kind": "vector"}))
    except Exception as e:
        st.warning(f"Qdrant-Suche nicht möglich: {e}")
    if not results:
        return []
    pairs = [(query, r[0]) for r in results if r[0]]
    if reranker is not None:
        try:
            rr = reranker.predict(pairs)
            reranked = sorted(zip(results, rr), key=lambda x: x[1], reverse=True)
            return [(r[0][0], float(score), r[0][2]) for r, score in reranked[:top_k]]
        except Exception as e:
            st.warning(f"Reranker-Fehler: {e}. Zeige ungeordnete Ergebnisse.")
    return results[:top_k]

# ---------- LLM ----------
def call_llm(context: str, question: str) -> str:
    if not OSS_API_BASE or not OSS_API_KEY or not OSS_MODEL:
        return "LLM nicht konfiguriert."
    headers = {
        "Authorization": f"Bearer {OSS_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("APP_URL", "https://streamlit.io"),
        "X-Title": "ErinnerungsBot Steiermark",
    }
    payload = {
        "model": OSS_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"KONTEXT:\n{context}\n\nFRAGE:\n{question}"},
        ],
        "temperature": 0.2,
    }
    try:
        r = requests.post(OSS_API_BASE.rstrip("/") + "/v1/chat/completions", headers=headers, json=payload, timeout=120)
        if r.status_code >= 400:
            return f"LLM-Fehler ({r.status_code}): {r.text[:500]}"
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"LLM-Fehler: {e}"

# ---------- UI ----------
st.title("💬 ErinnerungsBot Steiermark")
st.caption("Antwortet strikt nur aus den Dokumenten im Repository-Ordner `data/`.")

with st.sidebar:
    st.header("Index verwalten")
    st.caption("📁 Datenquelle: `data/` im GitHub-Repo")
    role = st.session_state.get("role", "viewer")
    if role == "admin":
        if st.button("🧱 Vollständiger Rebuild"):
            with st.spinner("Baue Index neu auf …"):
                os.system("python ingest.py --full-rebuild")
            st.success("Index neu aufgebaut.")
            st.rerun()
        if st.button("🔄 Inkrementelles Update"):
            with st.spinner("Aktualisiere Index …"):
                os.system("python ingest.py")
            st.success("Index aktualisiert.")
            st.rerun()
    else:
        st.info("Nur Ansicht: Re-Index ist Administratoren vorbehalten.")

# Chat
question = st.text_input("Frage eingeben")
if st.button("Senden") and question:
    with st.spinner("Suche relevante Textstellen …"):
        hits = hybrid_search(question, top_k=8)
    if not hits:
        st.warning("Kein Kontext gefunden.")
    else:
        context = "\n\n".join([h[0] for h in hits if h[0]])
        answer = call_llm(context, question)
        st.subheader("Antwort")
        st.write(answer)
        with st.expander("🔎 Verwendete Ausschnitte"):
            for text, score, meta in hits:
                src = Path(str(meta.get("source", "—"))).name
                cid = meta.get("chunk_id", "—")
                kind = meta.get("kind", "—")
                st.markdown(f"**Quelle:** {src} · Chunk {cid} · {kind} · Score={score:.3f}")
                st.write((text or "")[:1200])
                st.markdown("---")

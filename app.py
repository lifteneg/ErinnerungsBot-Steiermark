# app.py – ErinnerungsBot Steiermark (RAG: Qdrant + BM25 + Jina-Embeddings)
from __future__ import annotations
import os
import time
import pickle
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import requests
import streamlit as st
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient

# ---------- Seite ----------
st.set_page_config(page_title="💬 ErinnerungsBot Steiermark", layout="wide")

# ---------- Pfade ----------
DATA_DIR = Path("./data")
INDEX_DIR = Path("./index"); INDEX_DIR.mkdir(exist_ok=True)
BM25_FILE = INDEX_DIR / "bm25.pkl"
DOCS_FILE = INDEX_DIR / "docs.pkl"

# ---------- Qdrant ----------
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = "docs_bge_m3"
qdr = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ---------- Jina Embeddings ----------
JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_MODEL = os.getenv("JINA_MODEL", "jina-embeddings-v2-base-de")
JINA_URL = "https://api.jina.ai/v1/embeddings"

def jina_embed(texts: list[str]) -> list[list[float]]:
    if not JINA_API_KEY:
        raise RuntimeError("JINA_API_KEY fehlt – bitte in Streamlit Secrets setzen.")
    headers = {"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"}
    out = []
    for i in range(0, len(texts), 128):
        payload = {"model": JINA_MODEL, "input": texts[i:i+128], "encoding_format": "float"}
        r = requests.post(JINA_URL, headers=headers, json=payload, timeout=60)
        if r.status_code >= 400:
            raise RuntimeError(f"Jina-API Fehler {r.status_code}: {r.text[:200]}")
        out.extend([d["embedding"] for d in r.json()["data"]])
    return out

# ---------- Rollen / Tokens ----------
def _split_tokens(value: str) -> list[str]:
    return [t.strip() for t in value.split(",") if t.strip()]

ADMIN_SET = set(_split_tokens(os.getenv("ADMIN_TOKENS", "")))
VIEW_SET = set(_split_tokens(os.getenv("VIEW_TOKENS", "")))
ALL_SET = set(_split_tokens(os.getenv("AUTH_TOKENS", "")))
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

# ---------- Dataclasses ----------
@dataclass
class DocChunk:
    text: str
    source: str
    chunk_id: int
    meta: Dict[str, str]

# ---------- BM25 Laden ----------
def load_bm25() -> Tuple[BM25Okapi | None, List[DocChunk]]:
    if BM25_FILE.exists() and DOCS_FILE.exists():
        with open(BM25_FILE, "rb") as f: bm25 = pickle.load(f)
        with open(DOCS_FILE, "rb") as f: docs = pickle.load(f)
        return bm25, docs
    return None, []

bm25, docs = load_bm25()

# ---------- Suche ----------
def hybrid_search(query: str, top_k: int = 8) -> List[Tuple[str, float, Dict]]:
    results: List[Tuple[str, float, Dict]] = []
    # BM25
    if bm25 and docs:
        tq = query.lower().split()
        scores = bm25.get_scores(tq)
        top_idx = np.argsort(scores)[-top_k:][::-1]
        for i in top_idx:
            if scores[i] <= 0: continue
            d: DocChunk = docs[i]
            results.append((d.text, float(scores[i]),
                            {"source": d.source, "chunk_id": d.chunk_id, "kind": "bm25"}))
    # Qdrant
    try:
        qv = jina_embed([query])[0]
        hits = qdr.search(collection_name=QDRANT_COLLECTION, query_vector=qv, limit=top_k, with_payload=True)
        for h in hits:
            payload = h.payload or {}
            results.append((payload.get("text", ""), float(h.score),
                            {"source": payload.get("source"), "chunk_id": payload.get("chunk_id"), "kind": "vector"}))
    except Exception as e:
        st.warning(f"Vektor-Suche nicht möglich: {e}")
    return results[:top_k]

# ---------- LLM ----------
OSS_API_BASE = os.getenv("OSS_API_BASE")
OSS_API_KEY = os.getenv("OSS_API_KEY")
OSS_MODEL = os.getenv("OSS_MODEL")

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

    # --- Statusanzeige ---
    try:
        count = qdr.count(QDRANT_COLLECTION, exact=True).count
    except Exception:
        count = None
    last_ingest_time = None
    state_file = INDEX_DIR / "ingest_state.json"
    if state_file.exists():
        last_ingest_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(state_file.stat().st_mtime))
    num_data_files = len(list(DATA_DIR.rglob("*.*")))

    st.subheader("📊 Index-Status")
    st.write(f"**Qdrant Chunks:** {count if count is not None else '—'}")
    st.write(f"**Letzter Ingest:** {last_ingest_time or '—'}")
    st.write(f"**Dateien in data/:** {num_data_files}")
    st.write(f"**BM25 geladen:** {'✅' if bm25 else '❌'}")

    role = st.session_state.get("role", "viewer")
    if role == "admin":
        if st.button("🧱 Vollständiger Rebuild"):
            with st.spinner("Baue Index neu auf …"):
                os.system("python ingest.py --full-rebuild --clear")
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

# app.py – ErinnerungsBot Steiermark (Streamlit + BM25 + Qdrant/Jina + OpenRouter)
from __future__ import annotations

import os
import sys
import pickle
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Any

import requests
import numpy as np
import streamlit as st
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

# -----------------------------
# Konstante Feineinstellungen
# -----------------------------
TOP_K = 10          # Maximale Trefferzahl für Hybrid-Suche
LOG_TAIL = 1000     # Anzahl der Logzeilen, die im UI angezeigt werden

# -----------------------------
# UI-Setup
# -----------------------------
st.set_page_config(page_title="💬 ErinnerungsBot Steiermark", page_icon="💬", layout="wide")
st.title("💬 ErinnerungsBot Steiermark")

# -----------------------------
# Helpers
# -----------------------------
def _get_secret_env(name: str, default: str = "") -> str:
    val = os.getenv(name, st.secrets.get(name, default))
    return (val or "").strip()

def _mask(s: str) -> str:
    if not s:
        return "—"
    return s[:4] + "…" + s[-4:] if len(s) > 8 else "•••"

# -----------------------------
# Secrets / Konfig
# -----------------------------
ADMIN_TOKENS = _get_secret_env("ADMIN_TOKENS")   # z.B. "Marlene"
VIEW_TOKENS  = _get_secret_env("VIEW_TOKENS")    # z.B. "Schule"

OSS_API_BASE = _get_secret_env("OSS_API_BASE", "https://openrouter.ai/api").rstrip("/")
OSS_API_KEY  = _get_secret_env("OSS_API_KEY")
OSS_MODEL    = _get_secret_env("OSS_MODEL", "openai/gpt-oss-20b:free")

QDRANT_URL        = _get_secret_env("QDRANT_URL")
QDRANT_API_KEY    = _get_secret_env("QDRANT_API_KEY")
QDRANT_COLLECTION = _get_secret_env("QDRANT_COLLECTION", "docs_bge_m3")

JINA_API_KEY = _get_secret_env("JINA_API_KEY")
JINA_MODEL   = _get_secret_env("JINA_MODEL", "jina-embeddings-v2-base-de")
JINA_URL     = "https://api.jina.ai/v1/embeddings"

# -----------------------------
# Auth
# -----------------------------
role = st.sidebar.selectbox("Rolle wählen", ["viewer", "admin"])
token = st.sidebar.text_input("Access Token", type="password", placeholder="Token eingeben")
if (role == "admin" and ADMIN_TOKENS and token != ADMIN_TOKENS) or (role == "viewer" and VIEW_TOKENS and token != VIEW_TOKENS):
    st.error("Falsches oder fehlendes Token.")
    st.stop()

# -----------------------------
# Index laden (falls vorhanden)
# -----------------------------
INDEX_DIR = Path("index")
BM25_PATH = INDEX_DIR / "bm25.pkl"
DOCS_PATH = INDEX_DIR / "docs.pkl"

def load_index() -> Tuple[BM25Okapi | None, List[Dict[str, Any]]]:
    if not (BM25_PATH.exists() and DOCS_PATH.exists()):
        return None, []
    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)
    with open(DOCS_PATH, "rb") as f:
        docs = pickle.load(f)
    return bm25, docs

bm25, docs = load_index()

# -----------------------------
# Qdrant-Client
# -----------------------------
qdr = None
if QDRANT_URL and QDRANT_API_KEY:
    try:
        qdr = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    except Exception as e:
        st.warning(f"Qdrant-Client konnte nicht initialisiert werden: {e}")

# -----------------------------
# Jina Embeddings für Query
# -----------------------------
def jina_embed(texts: List[str]) -> List[List[float]]:
    if not JINA_API_KEY:
        raise RuntimeError("JINA_API_KEY fehlt (Secrets).")
    headers = {"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"}
    out: List[List[float]] = []
    B = 32
    for i in range(0, len(texts), B):
        payload = {"model": JINA_MODEL, "input": texts[i:i+B]}
        tries = 0
        while tries < 3:
            try:
                r = requests.post(JINA_URL, headers=headers, json=payload, timeout=120)
                if r.status_code >= 400:
                    raise RuntimeError(f"Jina-API Fehler {r.status_code}: {r.text[:200]}")
                out.extend([d["embedding"] for d in r.json()["data"]])
                break
            except requests.exceptions.ReadTimeout:
                tries += 1
                if tries >= 3:
                    raise
    return out

# -----------------------------
# Hybrid-Suche (BM25 + Qdrant) – ohne harte Schwelle
# -----------------------------
def hybrid_search(query: str, top_k: int = TOP_K) -> List[Tuple[str, float, Dict[str, Any]]]:
    results: List[Tuple[str, float, Dict[str, Any]]] = []

    # BM25
    if bm25 and docs:
        tq = query.lower().split()
        scores = bm25.get_scores(tq)
        top_idx = np.argsort(scores)[-max(top_k, 8):][::-1]
        for i in top_idx:
            d = docs[i]
            results.append((
                d.get("text", ""),
                float(scores[i]),
                {"source": d.get("source", ""), "kind": "bm25"}
            ))

    # Qdrant
    if qdr is not None:
        try:
            qv = jina_embed([query])[0]
            hits = qdr.search(collection_name=QDRANT_COLLECTION, query_vector=qv, limit=top_k, with_payload=True)
            for h in hits:
                p = h.payload or {}
                results.append((
                    p.get("text", ""),
                    float(h.score),
                    {"source": p.get("source", ""), "kind": "vector"}
                ))
        except Exception as e:
            st.warning(f"Vektor-Suche nicht möglich: {e}")

    # Deduplizieren nach Text & sortieren
    seen = set()
    unique: List[Tuple[str, float, Dict[str, Any]]] = []
    for text, score, meta in sorted(results, key=lambda x: x[1], reverse=True):
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        unique.append((text, score, meta))
        if len(unique) >= top_k:
            break

    return unique

# -----------------------------
# LLM (OpenRouter)
# -----------------------------
SYSTEM_PROMPT = (
    "Du bist ein präziser Assistent. Antworte ausschließlich mit Informationen, "
    "die im bereitgestellten KONTEXT enthalten sind. Wenn der Kontext keine Antwort zulässt, "
    "sage eindeutig: 'Dazu habe ich keine Information in meinen Daten.' Erfinde nichts."
)

def call_llm(question: str, context: str) -> str:
    if not OSS_API_KEY:
        return "LLM nicht konfiguriert: OSS_API_KEY fehlt."
    url = OSS_API_BASE + "/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OSS_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("APP_URL", "https://streamlit.io"),
        "X-Title": "ErinnerungsBot Steiermark",
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"KONTEXT:\n{context}\n\nFRAGE:\n{question}"},
    ]
    try:
        r = requests.post(url, headers=headers, json={"model": OSS_MODEL, "messages": messages, "temperature": 0.2}, timeout=60)
        if r.status_code == 401:
            return "LLM-Fehler (401 Unauthorized): Prüfe OSS_API_KEY in den Secrets."
        if r.status_code >= 400:
            return f"LLM-Fehler ({r.status_code}): {r.text[:400]}"
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"LLM-Fehler: {e}"

# -----------------------------
# In-App Rebuild (ingest.py aufrufen) – mit Live-Logs & Reload
# -----------------------------
def run_ingest(full_rebuild: bool = True, clear: bool = False):
    env = os.environ.copy()
    for k, v in st.secrets.items():
        env[str(k)] = str(v)

    st.caption(
        "Übergabewerte: "
        f"JINA_API_KEY={_mask(env.get('JINA_API_KEY',''))} · "
        f"Q

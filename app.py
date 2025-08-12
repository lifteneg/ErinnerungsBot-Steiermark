# app.py – ErinnerungsBot Steiermark (RAG: Qdrant + BM25 + Jina-Embeddings)
from __future__ import annotations
import os, time, pickle, hashlib, subprocess
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
def _normalize_qdrant_url(raw: str | None) -> str | None:
    if not raw:
        return None
    # Cloud-URL ohne Port → :6333 ergänzen
    if raw.startswith("http") and ":" not in raw.split("//", 1)[1]:
        return raw.rstrip("/") + ":6333"
    return raw.rstrip("/")

QDRANT_URL = _normalize_qdrant_url(os.getenv("QDRANT_URL"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docs_bge_m3")
qdr = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ---------- Jina Embeddings ----------
JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_MODEL = os.getenv("JINA_MODEL", "jina-embeddings-v2-base-de")
JINA_URL = os.getenv("JINA_EMBED_URL", "https://api.jina.ai/v1/embeddings")

def jina_embed(texts: list[str]) -> list[list[float]]:
    if not JINA_API_KEY:
        raise RuntimeError("JINA_API_KEY fehlt – bitte in Streamlit Secrets setzen.")
    headers = {"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"}
    out: list[list[float]] = []
    B = 128
    for i in range(0, len(texts), B):
        payload = {"model": JINA_MODEL, "input": texts[i:i+B]}
        r = requests.post(JINA_URL, headers=headers, json=payload, timeout=60)
        if r.status_code >= 400:
            raise RuntimeError(f"Jina-API Fehler {r.status_code}: {r.text[:200]}")
        out.extend([d["embedding"] for d in r.json()["data"]])
    return out

# ---------- OpenRouter (LLM) ----------
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
    ADMIN_SET = ALL_SET  # Fallback

SYSTEM_PROMPT = (
    "Du bist ein präziser Assistent. Antworte ausschließlich mit Informationen, "
    "die im bereitgestellten KONTEXT enthalten sind. Wenn der Kontext keine Antwort zulässt, "
    "sage eindeutig: 'Dazu habe ich keine Information in meinen Daten.' Erfinde nichts."
)

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

# ---------- Dataclasses & BM25 ----------
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
    # Qdrant (dicht)
    try:
        qv = jina_embed([query])[0]
        hits = qdr.search(collection_name=QDRANT_COLLECTION, query_vector=qv, limit=top_k, with_payload=True)
        for h in hits:
            p = h.payload or {}
            results.append((p.get("text",""), float(h.score),
                            {"source": p.get("source"), "chunk_id": p.get("chunk_id"), "kind": "vector"}))
    except Exception as e:
        st.warning(f"Vektor-Suche nicht möglich: {e}")
    return results[:top_k] if results else []

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
    payload = {"model": OSS_MODEL, "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"KONTEXT:\n{context}\n\nFRAGE:\n{question}"},
    ], "temperature": 0.2}
    try:
        r = requests.post(OSS_API_BASE.rstrip("/") + "/v1/chat/completions", headers=headers, json=payload, timeout=120)
        if r.status_code >= 400:
            return f"LLM-Fehler ({r.status_code}): {r.text[:500]}"
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"LLM-Fehler: {e}"

# ---------- Admin: Diagnose & Rebuild ----------
def _masked(s: str) -> str:
    if not s: return "—"
    return s[:4] + "…" + s[-4:] if len(s) > 8 else "•••"

def run_ingest(incremental: bool = False, clear: bool = False):
    if not QDRANT_URL or not QDRANT_API_KEY:
        st.error("QDRANT_URL / QDRANT_API_KEY nicht gesetzt.")
        return
    if not JINA_API_KEY:
        st.error("JINA_API_KEY nicht gesetzt.")
        return
    st.write("### 🧱 In-App Rebuild (Logs)")
    args = ["python", "ingest.py"]
    if not incremental: args += ["--full-rebuild"]
    if clear: args += ["--clear"]
    env = os.environ.copy()
    env["QDRANT_URL"] = QDRANT_URL or ""
    env["QDRANT_API_KEY"] = QDRANT_API_KEY or ""
    env["JINA_API_KEY"] = JINA_API_KEY or ""
    env["JINA_MODEL"] = JINA_MODEL or "jina-embeddings-v2-base-de"
    log_box = st.empty()
    lines = []
    try:
        import subprocess
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                bufsize=1, universal_newlines=True, env=env)
        for line in proc.stdout:
            lines.append(line.rstrip("\n"))
            log_box.code("\n".join(lines[-200:]), language="bash")
        ret = proc.wait()
        if ret == 0:
            st.success("Ingest abgeschlossen ✅")
            global bm25, docs
            bm25, docs = load_bm25()
        else:
            st.error(f"Ingest fehlgeschlagen (exit {ret})")
    except Exception as e:
        st.error(f"Ingest-Aufruf fehlgeschlagen: {e}")

# ---------- UI ----------
st.title("💬 ErinnerungsBot Steiermark")
st.caption("Antwortet strikt nur aus den Dokumenten im Repository-Ordner `data/`.")

with st.sidebar:
    st.header("Index verwalten")
    st.caption("📁 Datenquelle: `data/` im GitHub-Repo")
    # Status
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
    st.write(f"🔌 Qdrant: `{QDRANT_URL or '—'}` · Collection: `{QDRANT_COLLECTION}`")
    st.write(f"🧠 Embeddings: `Jina · {JINA_MODEL}`")
    st.write(f"🔑 Keys: Jina={_masked(JINA_API_KEY)}, Qdrant={_masked(QDRANT_API_KEY)}, OpenRouter={_masked(OSS_API_KEY)}")

    role = st.session_state.get("role", "viewer")
    if role == "admin":
        colA, colB = st.columns(2)
        with colA:
            if st.button("🧱 Vollständiger Rebuild (CLEAR)"):
                run_ingest(incremental=False, clear=True)
        with colB:
            if st.button("🔄 Inkrementelles Update"):
                run_ingest(incremental=True, clear=False)
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

# app.py – ErinnerungsBot Steiermark (RAG: Qdrant + BM25 + Jina-Embeddings + PDF-Support)

from __future__ import annotations
import os, sys, pickle, subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any

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

# ---------- Helper ----------
def _normalize_qdrant_url(raw: str | None) -> str | None:
    if not raw:
        return None
    raw = raw.strip()
    if raw.startswith("http") and ":" not in raw.split("//", 1)[1]:
        return raw.rstrip("/") + ":6333"
    return raw.rstrip("/")

def _get_secret_env(name: str, default: str = "") -> str:
    val = os.getenv(name, st.secrets.get(name, default))
    return (val or "").strip()

def _mask(s: str) -> str:
    if not s: return "—"
    return s[:4] + "…" + s[-4:] if len(s) > 8 else "•••"

# ---------- Qdrant ----------
QDRANT_URL = _normalize_qdrant_url(_get_secret_env("QDRANT_URL"))
QDRANT_API_KEY = _get_secret_env("QDRANT_API_KEY")
QDRANT_COLLECTION = _get_secret_env("QDRANT_COLLECTION", "docs_bge_m3")
qdr = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ---------- Jina Embeddings ----------
JINA_API_KEY = _get_secret_env("JINA_API_KEY")
JINA_MODEL   = _get_secret_env("JINA_MODEL", "jina-embeddings-v2-base-de")
JINA_URL     = _get_secret_env("JINA_EMBED_URL") or "https://api.jina.ai/v1/embeddings"

def jina_embed(texts: list[str]) -> list[list[float]]:
    if not JINA_API_KEY:
        raise RuntimeError("JINA_API_KEY fehlt – bitte in Streamlit Secrets setzen.")
    headers = {"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"}
    out: list[list[float]] = []
    B = 32  # kleiner für Stabilität
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

# ---------- OpenRouter (LLM) ----------
OSS_API_BASE = (_get_secret_env("OSS_API_BASE") or "https://openrouter.ai/api").rstrip("/")
OSS_API_KEY  = _get_secret_env("OSS_API_KEY")
OSS_MODEL    = _get_secret_env("OSS_MODEL", "openai/gpt-oss-20b:free")

SYSTEM_PROMPT = (
    "Du bist ein präziser Assistent. Antworte ausschließlich mit Informationen, "
    "die im bereitgestellten KONTEXT enthalten sind. Wenn der Kontext keine Antwort zulässt, "
    "sage eindeutig: 'Dazu habe ich keine Information in meinen Daten.' Erfinde nichts."
)

def call_llm(context: str, question: str) -> str:
    if not OSS_API_KEY:
        return "LLM nicht konfiguriert: OSS_API_KEY fehlt."
    headers = {
        "Authorization": f"Bearer {OSS_API_KEY}",
        "Content-Type": "application/json",
        # OpenRouter empfiehlt Referer & X-Title:
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
        r = requests.post(OSS_API_BASE + "/v1/chat/completions", headers=headers, json=payload, timeout=60)
        if r.status_code == 401:
            return "LLM-Fehler (401 Unauthorized): Prüfe OSS_API_KEY in den Secrets."
        if r.status_code >= 400:
            return f"LLM-Fehler ({r.status_code}): {r.text[:500]}"
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"LLM-Fehler: {e}"

# ---------- Rollen / Tokens ----------
def _split_tokens(value: str) -> list[str]:
    return [t.strip() for t in (value or "").split(",") if t.strip()]

ADMIN_SET = set(_split_tokens(_get_secret_env("ADMIN_TOKENS")))
VIEW_SET  = set(_split_tokens(_get_secret_env("VIEW_TOKENS")))
ALL_SET   = set(_split_tokens(_get_secret_env("AUTH_TOKENS")))
if ALL_SET and not ADMIN_SET and not VIEW_SET:
    ADMIN_SET = ALL_SET

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

# ---------- Datatypes & BM25 ----------
@dataclass
class DocChunk:
    text: str
    source: str
    chunk_id: int
    meta: Dict[str, Any]

def load_bm25() -> Tuple[BM25Okapi | None, List]:
    if not (BM25_FILE.exists() and DOCS_FILE.exists()):
        return None, []
    with open(BM25_FILE, "rb") as f:
        bm25 = pickle.load(f)
    with open(DOCS_FILE, "rb") as f:
        docs_raw = pickle.load(f)
    return bm25, docs_raw

bm25, docs = load_bm25()

# ---------- Suche ----------
def hybrid_search(query: str, top_k: int = 8) -> List[Tuple[str, float, Dict]]:
    results: List[Tuple[str, float, Dict]] = []

    def read_chunk(x):
        if isinstance(x, dict):
            return x.get("text", ""), x.get("source", ""), int(x.get("chunk_id", 0))
        return getattr(x, "text", ""), getattr(x, "source", ""), int(getattr(x, "chunk_id", 0))

    # BM25
    if bm25 and docs:
        tq = query.lower().split()
        scores = bm25.get_scores(tq)
        top_idx = np.argsort(scores)[-top_k:][::-1]
        for i in top_idx:
            if scores[i] <= 0:
                continue
            text, source, cid = read_chunk(docs[i])
            results.append((text, float(scores[i]),
                            {"source": source, "chunk_id": cid, "kind": "bm25"}))

    # Vektor-Suche (Qdrant + Jina-Query-Embed)
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

# ---------- Admin: In-App Rebuild ----------
def run_ingest(incremental: bool = False, clear: bool = False):
    env = os.environ.copy()
    for key, value in st.secrets.items():
        env[str(key)] = str(value)

    st.caption(
        f"Übergabewerte: JINA_API_KEY={_mask(env.get('JINA_API_KEY',''))} · "
        f"QDRANT_API_KEY={_mask(env.get('QDRANT_API_KEY',''))} · "
        f"QDRANT_URL={env.get('QDRANT_URL','—')} · "
        f"JINA_MODEL={env.get('JINA_MODEL','—')} · "
        f"OSS_API_KEY={_mask(env.get('OSS_API_KEY',''))}"
    )

    args = [sys.executable, "ingest.py"]
    if not incremental: args += ["--full-rebuild"]
    if clear: args += ["--clear"]
    args += [
        f"--jina_api_key={env.get('JINA_API_KEY','')}",
        f"--jina_model={env.get('JINA_MODEL','jina-embeddings-v2-base-de')}",
        f"--qdrant_url={env.get('QDRANT_URL','')}",
        f"--qdrant_api_key={env.get('QDRANT_API_KEY','')}",
        f"--qdrant_collection={env.get('QDRANT_COLLECTION','docs_bge_m3')}",
    ]

    log_box = st.empty(); lines = []
    try:
        proc = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, universal_newlines=True, env=env, cwd=str(Path(__file__).parent)
        )
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
    role = st.session_state.get("role", "viewer")
    if role == "admin":
        if st.button("🧱 Vollständiger Rebuild (CLEAR)"):
            run_ingest(incremental=False, clear=True)
        if st.button("🧱 Vollständiger Rebuild"):
            run_ingest(incremental=False, clear=False)
        if st.button("🔄 Inkrementelles Update"):
            run_ingest(incremental=True, clear=False)

        st.divider()
        # 🧪 LLM Direkt-Test (prüft OSS_API_KEY + Modell)
        if st.button("🧪 LLM-Test (OpenRouter)"):
            if not OSS_API_KEY:
                st.error("OSS_API_KEY fehlt in den Secrets.")
            else:
                try:
                    test = requests.post(
                        OSS_API_BASE + "/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {OSS_API_KEY}",
                            "Content-Type": "application/json",
                            "HTTP-Referer": os.getenv("APP_URL", "https://streamlit.io"),
                            "X-Title": "ErinnerungsBot Steiermark",
                        },
                        json={"model": OSS_MODEL, "messages":[{"role":"user","content":"Antworte mit 'OK'."}], "temperature":0},
                        timeout=30
                    )
                    if test.status_code == 200:
                        st.success("OpenRouter OK ✅")
                    else:
                        st.error(f"OpenRouter Fehler {test.status_code}: {test.text[:300]}")
                except Exception as e:
                    st.error(f"OpenRouter-Request fehlgeschlagen: {e}")
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

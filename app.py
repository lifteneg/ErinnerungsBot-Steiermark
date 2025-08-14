# app.py – ErinnerungsBot Steiermark (ausführlichere Antworten)
# - Höheres TOP_K und smarter Kontext-Builder (mehr Details, pro-Dokument-Limit)
# - Größeres MAX_CONTEXT_CHARS, dynamisch nach „Antwortstil“
# - Prompting für ausführliche, strukturierte Antworten (ohne Quellen im Text)
# - „Verwendete Ausschnitte“ zeigt weiterhin nur TEI-URIs

from __future__ import annotations

import os
import sys
import time
import re
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
# Feineinstellungen (Default)
# -----------------------------
DEFAULT_TOP_K = 24          # mehr Treffer einsammeln
BASE_CONTEXT_CHARS = 12000  # höherer Grundwert
PER_DOC_LIMIT = 5           # max. Snippets pro Dokument (mehr Tiefe, aber keine Einzeldok-Dominanz)
LOG_TAIL = 1000

PRIMARY_MODEL   = "deepseek/deepseek-r1-0528:free"
FALLBACK_MODEL  = "openai/gpt-oss-20b:free"

# -----------------------------
# UI
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
ADMIN_TOKENS = _get_secret_env("ADMIN_TOKENS")
VIEW_TOKENS  = _get_secret_env("VIEW_TOKENS")

OSS_API_BASE = _get_secret_env("OSS_API_BASE", "https://openrouter.ai/api").rstrip("/")
OSS_API_KEY  = _get_secret_env("OSS_API_KEY")
OSS_MODEL    = _get_secret_env("OSS_MODEL", PRIMARY_MODEL)

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

# Antwortstil / Steuerung
st.sidebar.markdown("### Antwortstil")
style = st.sidebar.radio("Detailgrad", ["Kurz", "Mittel", "Ausführlich"], index=2)
if style == "Kurz":
    TOP_K = 12
    MAX_CONTEXT_CHARS = BASE_CONTEXT_CHARS // 2  # ~6k
    STYLE_HINT = (
        "Fasse dich relativ kurz, antworte in 1–2 Absätzen, nur die wichtigsten Fakten."
    )
elif style == "Mittel":
    TOP_K = DEFAULT_TOP_K
    MAX_CONTEXT_CHARS = BASE_CONTEXT_CHARS  # ~12k
    STYLE_HINT = (
        "Gib eine ausgewogene, vollständige Antwort in mehreren Absätzen mit klarer Struktur."
    )
else:  # Ausführlich
    TOP_K = 32
    MAX_CONTEXT_CHARS = int(BASE_CONTEXT_CHARS * 1.6)  # ~19k (Achtung: Modellkontext)
    STYLE_HINT = (
        "Liefere eine ausführliche, strukturierte Darstellung mit Abschnitten "
        "(z. B. Biografie, Kontext/Verfolgung, Orte/Zeiten, Gedenken). "
        "Nutze alle relevanten Details aus dem Kontext."
    )

# -----------------------------
# Index laden
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
# Jina Query-Embeddings
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
                time.sleep(2 ** tries)
    return out

# -----------------------------
# Retrieval
# -----------------------------
def hybrid_search(query: str, top_k: int) -> List[Tuple[str, float, Dict[str, Any]]]:
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
                {
                    "kind": "bm25",
                    "tei_uris": d.get("tei_uris", []),  # nur diese zeigen wir später
                    "source": d.get("source", ""),
                }
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
                    {
                        "kind": "vector",
                        "tei_uris": p.get("tei_uris", []),
                        "source": p.get("source", ""),
                    }
                ))
        except Exception as e:
            st.warning(f"Vektor-Suche nicht möglich: {e}")

    # Deduplizieren nach Text (grobe Heuristik)
    seen_text = set()
    deduped: List[Tuple[str, float, Dict[str, Any]]] = []
    for text, score, meta in sorted(results, key=lambda x: x[1], reverse=True):
        if not text:
            continue
        key = text[:500]
        if key in seen_text:
            continue
        seen_text.add(key)
        deduped.append((text, score, meta))
    return deduped[:max(top_k, 8)]

def build_context(hits: List[Tuple[str, float, Dict[str, Any]]],
                  max_chars: int,
                  per_doc_limit: int = PER_DOC_LIMIT) -> Tuple[str, List[Tuple[str, float, Dict[str, Any]]]]:
    """Nimmt die besten Snippets, begrenzt pro Dokument, und baut einen großen Kontext."""
    by_doc_count: Dict[str, int] = {}
    chosen: List[Tuple[str, float, Dict[str, Any]]] = []
    total = 0

    for text, score, meta in hits:
        src = meta.get("source", "")
        if by_doc_count.get(src, 0) >= per_doc_limit:
            continue
        tclean = re.sub(r"\s+", " ", text).strip()
        if not tclean:
            continue
        if total + len(tclean) > max_chars:
            continue
        chosen.append((tclean, score, meta))
        by_doc_count[src] = by_doc_count.get(src, 0) + 1
        total += len(tclean)
        if total >= max_chars:
            break

    context = "\n\n".join([c[0] for c in chosen])
    return context, chosen

def collect_all_tei_urls(hits: List[Tuple[str, float, Dict[str, Any]]]) -> List[str]:
    """Sammelt nur TEI-URIs aus den Treffer-Metas, dedupliziert und behält Reihenfolge."""
    seen = set()
    out: List[str] = []
    for _, __, meta in hits:
        for u in meta.get("tei_uris", []) or []:
            if u and u not in seen:
                seen.add(u)
                out.append(u)
    return out

# -----------------------------
# LLM (OpenRouter) – Retry + Fallback
# -----------------------------
SYSTEM_PROMPT_BASE = (
    "Du bist ein präziser Assistent. Antworte ausschließlich mit Informationen, "
    "die im bereitgestellten KONTEXT enthalten sind. "
    "Füge KEINE Quellenangaben in deine Antwort ein; die Quellen werden separat angezeigt. "
    "Wenn der Kontext keine Antwort zulässt, sage eindeutig: 'Dazu habe ich keine Information in meinen Daten.' "
    "Erfinde nichts."
)

def call_llm_with_models(question: str, context: str, models: List[str], style_hint: str) -> Tuple[str, str]:
    if not OSS_API_KEY:
        return "LLM nicht konfiguriert: OSS_API_KEY fehlt.", ""

    url = OSS_API_BASE + "/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OSS_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("APP_URL", "https://streamlit.io"),
        "X-Title": "ErinnerungsBot Steiermark",
    }

    # Stilhinweis ergänzen
    system_prompt = SYSTEM_PROMPT_BASE + " " + style_hint

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Verarbeite den folgenden KONTEXT vollständig, fasse zusammen und strukturiere:\n\n"
                f"KONTEXT:\n{context}\n\nFRAGE:\n{question}"
            ),
        },
    ]

    RETRY_STATUSES = {429, 502, 503, 504}
    MAX_TRIES = 4

    for mdl in models:
        for attempt in range(1, MAX_TRIES + 1):
            try:
                r = requests.post(
                    url,
                    headers=headers,
                    json={"model": mdl, "messages": messages, "temperature": 0.2},
                    timeout=120,
                )
                if r.status_code == 401:
                    return "LLM-Fehler (401 Unauthorized): Prüfe OSS_API_KEY in den Secrets.", ""
                if r.status_code in RETRY_STATUSES:
                    if attempt == MAX_TRIES:
                        break
                    time.sleep(2 ** attempt)
                    continue
                if r.status_code >= 400:
                    break
                return r.json()["choices"][0]["message"]["content"], mdl
            except requests.RequestException:
                if attempt == MAX_TRIES:
                    break
                time.sleep(2 ** attempt)

    return "LLM derzeit nicht erreichbar. Bitte später erneut versuchen.", ""

# -----------------------------
# In-App Rebuild
# -----------------------------
def run_ingest(full_rebuild: bool = True, clear: bool = False):
    env = os.environ.copy()
    for k, v in st.secrets.items():
        env[str(k)] = str(v)

    st.caption(
        "Übergabewerte: "
        f"JINA_API_KEY={_mask(env.get('JINA_API_KEY',''))} · "
        f"QDRANT_API_KEY={_mask(env.get('QDRANT_API_KEY',''))} · "
        f"QDRANT_URL={env.get('QDRANT_URL','—')} · "
        f"JINA_MODEL={env.get('JINA_MODEL','—')} · "
        f"OSS_API_KEY={_mask(env.get('OSS_API_KEY',''))}"
    )

    args = [sys.executable, "ingest.py"]
    if full_rebuild:
        args.append("--full-rebuild")
    if clear:
        args.append("--clear")

    log_box = st.empty()
    lines: List[str] = []
    try:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        for line in proc.stdout:
            lines.append(line.rstrip("\n"))
            log_box.code("\n".join(lines[-LOG_TAIL:]), language="bash")
        ret = proc.wait()
        if ret == 0:
            st.success("Ingest abgeschlossen ✅")
            global bm25, docs
            bm25, docs = load_index()
        else:
            st.error(f"Ingest fehlgeschlagen (exit {ret})")
    except Exception as e:
        st.error(f"Ingest-Aufruf fehlgeschlagen: {e}")

# -----------------------------
# Auto-Bootstrap
# -----------------------------
if not (bm25 and docs):
    if role == "admin":
        st.info("📦 Kein lokaler Index gefunden – starte automatischen Rebuild …")
        run_ingest(full_rebuild=True, clear=False)
        bm25, docs = load_index()
        if not (bm25 and docs):
            st.error("Index konnte nicht aufgebaut werden. Bitte Logs oben prüfen.")
            st.stop()
    else:
        st.warning("ℹ️ Kein lokaler Index vorhanden. Bitte einen Admin um einen Rebuild.")
        st.stop()

# -----------------------------
# Sidebar: Admin
# -----------------------------
with st.sidebar:
    st.header("Index verwalten")
    if role == "admin":
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🧱 Rebuild"):
                run_ingest(full_rebuild=True, clear=False)
        with c2:
            if st.button("🧱 Rebuild (CLEAR)"):
                run_ingest(full_rebuild=True, clear=True)

# -----------------------------
# Chat
# -----------------------------
if "last_question" not in st.session_state:
    st.session_state.last_question = ""
if "last_context" not in st.session_state:
    st.session_state.last_context = ""

question = st.text_input("Frage eingeben")
colA, colB = st.columns([1, 1])
send_clicked = colA.button("Senden")
retry_clicked = colB.button("Erneut senden")

if send_clicked and question:
    with st.spinner("Suche relevante Textstellen …"):
        hits_all = hybrid_search(question, top_k=TOP_K)

    if not hits_all:
        st.warning("Dazu habe ich keine Information in meinen Daten.")
    else:
        # Kontext groß & sinnvoll bauen (mehr Details, pro Dokument begrenzen)
        context, hits_used = build_context(hits_all, max_chars=MAX_CONTEXT_CHARS, per_doc_limit=PER_DOC_LIMIT)
        st.session_state.last_question = question
        st.session_state.last_context = context

        answer, used_model = call_llm_with_models(
            question,
            context,
            models=[OSS_MODEL or PRIMARY_MODEL, FALLBACK_MODEL],
            style_hint=STYLE_HINT,
        )

        st.subheader("Antwort")
        st.write(answer)
        if used_model:
            st.caption(f"LLM: {used_model}")

        # NUR URL-Liste anzeigen (kein Score, kein Text, nur TEI-URIs)
        urls = collect_all_tei_urls(hits_used)
        with st.expander("🔎 Verwendete Ausschnitte"):
            if urls:
                for u in urls:
                    st.markdown(f"- {u}")
            else:
                st.markdown("_Keine TEI-URIs im Kontext gefunden._")

elif retry_clicked:
    if not st.session_state.last_question or not st.session_state.last_context:
        st.warning("Es gibt noch keine vorherige Anfrage zum erneuten Senden.")
    else:
        q = st.session_state.last_question
        ctx = st.session_state.last_context
        answer, used_model = call_llm_with_models(
            q,
            ctx,
            models=[OSS_MODEL or PRIMARY_MODEL, FALLBACK_MODEL],
            style_hint=STYLE_HINT,
        )
        st.subheader("Antwort (erneut gesendet)")
        st.write(answer)
        if used_model:
            st.caption(f"LLM: {used_model}")

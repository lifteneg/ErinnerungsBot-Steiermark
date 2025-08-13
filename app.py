import os
import pickle
from pathlib import Path
import streamlit as st
import requests
from qdrant_client import QdrantClient

# -----------------------------
# Modell-Einstellungen
# -----------------------------
PRIMARY_MODEL = "deepseek/deepseek-r1-0528:free"
FALLBACK_MODEL = "openai/gpt-oss-20b:free"

# -----------------------------
# Index-Dateien prüfen
# -----------------------------
if not Path("index/bm25.pkl").exists() or not Path("index/docs.pkl").exists():
    st.warning("📦 Kein lokaler Index gefunden – starte automatischen Rebuild …")
    os.system("python ingest.py --full-rebuild --clear")

with open("index/bm25.pkl", "rb") as f:
    bm25 = pickle.load(f)
with open("index/docs.pkl", "rb") as f:
    docs = pickle.load(f)

# -----------------------------
# Streamlit-Secrets / Env-Variablen laden
# -----------------------------
ADMIN_TOKENS = st.secrets.get("ADMIN_TOKENS", os.getenv("ADMIN_TOKENS", ""))
VIEW_TOKENS = st.secrets.get("VIEW_TOKENS", os.getenv("VIEW_TOKENS", ""))

OSS_API_BASE = st.secrets.get("OSS_API_BASE", os.getenv("OSS_API_BASE", ""))
OSS_API_KEY = st.secrets.get("OSS_API_KEY", os.getenv("OSS_API_KEY", ""))

QDRANT_URL = st.secrets.get("QDRANT_URL", os.getenv("QDRANT_URL", ""))
QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY", os.getenv("QDRANT_API_KEY", ""))
QDRANT_COLLECTION = st.secrets.get("QDRANT_COLLECTION", os.getenv("QDRANT_COLLECTION", "docs_bge_m3"))

JINA_API_KEY = st.secrets.get("JINA_API_KEY", os.getenv("JINA_API_KEY", ""))
JINA_MODEL = st.secrets.get("JINA_MODEL", os.getenv("JINA_MODEL", "jina-embeddings-v2-base-de"))

# -----------------------------
# Qdrant-Setup
# -----------------------------
qdr = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# -----------------------------
# Jina Embedding Funktion
# -----------------------------
def jina_embed(texts):
    if not JINA_API_KEY:
        raise RuntimeError("JINA_API_KEY nicht gesetzt.")
    url = "https://api.jina.ai/v1/embeddings"
    headers = {"Authorization": f"Bearer {JINA_API_KEY}"}
    payload = {"model": JINA_MODEL, "input": texts}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return [d["embedding"] for d in r.json()["data"]]

# -----------------------------
# Hybrid-Suche ohne harte Score-Grenze
# -----------------------------
def hybrid_search(query: str, top_k: int = 8):
    results = []
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_hits = sorted(
        [(docs[i], float(score)) for i, score in enumerate(bm25_scores)],
        key=lambda x: x[1],
        reverse=True
    )[:top_k]
    for doc, score in bm25_hits:
        results.append((doc["text"], score, doc["source"]))

    try:
        vec = jina_embed([query])[0]
        q_hits = qdr.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=vec,
            limit=top_k
        )
        for hit in q_hits:
            payload = hit.payload
            results.append((payload["text"], float(hit.score), payload.get("source", "")))
    except Exception as e:
        print(f"Qdrant-Suche fehlgeschlagen: {e}")

    seen = set()
    unique_results = []
    for text, score, source in sorted(results, key=lambda x: x[1], reverse=True):
        if text not in seen:
            seen.add(text)
            unique_results.append((text, score, source))
    return unique_results[:top_k]

# -----------------------------
# LLM-Aufruf mit Fallback
# -----------------------------
def call_llm(model, question, context):
    headers = {
        "Authorization": f"Bearer {OSS_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = f"Beantworte die folgende Frage streng basierend auf den gegebenen Kontext:\n\nKontext:\n{context}\n\nFrage: {question}\nAntwort:"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    r = requests.post(f"{OSS_API_BASE}/chat/completions", headers=headers, json=payload)
    if r.status_code != 200:
        raise RuntimeError(f"LLM-Fehler ({r.status_code}): {r.text}")
    return r.json()["choices"][0]["message"]["content"]

def llm_answer_with_fallback(question, context):
    try:
        return call_llm(PRIMARY_MODEL, question, context), PRIMARY_MODEL
    except Exception as e1:
        print(f"Primäres Modell fehlgeschlagen ({PRIMARY_MODEL}): {e1}")
        try:
            return call_llm(FALLBACK_MODEL, question, context), FALLBACK_MODEL
        except Exception as e2:
            print(f"Fallback-Modell fehlgeschlagen ({FALLBACK_MODEL}): {e2}")
            return f"Beide Modelle fehlgeschlagen.\nPrimär: {e1}\nFallback: {e2}", None

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="💬 ErinnerungsBot Steiermark", page_icon="💬")
st.title("💬 ErinnerungsBot Steiermark")

role = st.sidebar.selectbox("Rolle wählen", ["viewer", "admin"])
token = st.sidebar.text_input("Access Token", type="password")
if (role == "admin" and token != ADMIN_TOKENS) or (role == "viewer" and token != VIEW_TOKENS):
    st.error("Falsches Token.")
    st.stop()

if role == "admin":
    if st.button("🧱 Vollständiger Rebuild"):
        os.system("python ingest.py --full-rebuild --clear")

question = st.text_input("Frage eingeben")
if question:
    hits = hybrid_search(question, top_k=8)
    context = "\n\n".join([h[0] for h in hits])
    answer, used_model = llm_answer_with_fallback(question, context)
    st.write("**Antwort**")
    st.write(answer)
    if used_model:
        st.caption(f"Antwort erstellt mit Modell: {used_model}")
    st.markdown("**🔎 Verwendete Ausschnitte:**")
    for text, score, source in hits:
        st.markdown(f"- ({score:.3f}) *{source}*: {text[:200]}…")

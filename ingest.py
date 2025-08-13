import os
import argparse
import pickle
import glob
import time
import requests
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from PyPDF2 import PdfReader

# -----------------------------
# Secrets / Env laden
# -----------------------------
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docs_bge_m3")

JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_MODEL = os.getenv("JINA_MODEL", "jina-embeddings-v2-base-de")

# -----------------------------
# Embedding-Funktion mit Retry
# -----------------------------
def jina_embed(texts, retries=3, delay=5):
    if not JINA_API_KEY:
        raise RuntimeError("JINA_API_KEY nicht gesetzt.")
    url = "https://api.jina.ai/v1/embeddings"
    headers = {"Authorization": f"Bearer {JINA_API_KEY}"}
    payload = {"model": JINA_MODEL, "input": texts}

    for attempt in range(1, retries + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            return [d["embedding"] for d in r.json()["data"]]
        except requests.exceptions.Timeout:
            print(f"⏳ Timeout bei Jina (Versuch {attempt}/{retries}), warte {delay}s …")
            time.sleep(delay)
        except Exception as e:
            raise RuntimeError(f"Jina-API Fehler: {e}")
    raise RuntimeError("Jina-API wiederholt fehlgeschlagen.")

# -----------------------------
# PDF-Text extrahieren
# -----------------------------
def load_pdf(path):
    text = ""
    with open(path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                text += txt + "\n"
    return text.strip()

# -----------------------------
# Dokumente laden
# -----------------------------
def load_documents():
    docs = []
    for file in glob.glob("data/**/*", recursive=True):
        if file.lower().endswith(".pdf"):
            content = load_pdf(file)
        elif file.lower().endswith(".txt") or file.lower().endswith(".md"):
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        else:
            continue
        if content.strip():
            docs.append({"text": content.strip(), "source": os.path.basename(file)})
    return docs

# -----------------------------
# Chunking
# -----------------------------
def chunk_text(text, chunk_size=900, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# -----------------------------
# Qdrant Collection sicherstellen
# -----------------------------
def ensure_collection(qdr, dim, clear=False):
    try:
        if clear:
            qdr.delete_collection(QDRANT_COLLECTION)
        if not qdr.collection_exists(QDRANT_COLLECTION):
            qdr.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE)
            )
    except Exception as e:
        raise RuntimeError(f"⚠️ Qdrant-Collection konnte nicht erstellt werden: {e}")

# -----------------------------
# Main Indexing
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-rebuild", action="store_true")
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    print("📂 Sammle & chunke Dokumente …")
    raw_docs = load_documents()
    chunks = []
    for d in raw_docs:
        for c in chunk_text(d["text"]):
            chunks.append({"text": c, "source": d["source"]})

    print(f"Dokument-Chunks: {len(chunks)}")

    print("🔄 Erstelle BM25 …")
    bm25 = BM25Okapi([c["text"].lower().split() for c in chunks])
    os.makedirs("index", exist_ok=True)
    with open("index/bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open("index/docs.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("🧠 Erstelle/aktualisiere Qdrant …")
    qdr = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    vectors = jina_embed([c["text"] for c in chunks])
    ensure_collection(qdr, dim=len(vectors[0]), clear=args.clear)

    points = []
    for idx, (vec, doc) in enumerate(zip(vectors, chunks)):
        points.append(qmodels.PointStruct(id=idx, vector=vec, payload=doc))

    qdr.upsert(collection_name=QDRANT_COLLECTION, points=points)
    print("✅ Ingest abgeschlossen.")

if __name__ == "__main__":
    main()

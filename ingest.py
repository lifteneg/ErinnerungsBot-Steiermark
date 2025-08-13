import os
import json
import glob
import pickle
import argparse
import requests
import time
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi
from PyPDF2 import PdfReader

# -----------------------------
# Lade Umgebungsvariablen / Secrets
# -----------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docs_bge_m3")

JINA_API_KEY = os.getenv("JINA_API_KEY", "")
JINA_MODEL = os.getenv("JINA_MODEL", "jina-embeddings-v2-base-de")

# -----------------------------
# Qdrant Client
# -----------------------------
qdr = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# -----------------------------
# Dokumente laden
# -----------------------------
def load_documents():
    docs = []
    for file_path in glob.glob("data/**/*", recursive=True):
        if file_path.lower().endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            docs.append({"text": text, "source": file_path})
        elif file_path.lower().endswith(".md"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            docs.append({"text": text, "source": file_path})
        elif file_path.lower().endswith(".pdf"):
            text = ""
            try:
                pdf = PdfReader(file_path)
                for page in pdf.pages:
                    text += page.extract_text() or ""
            except Exception as e:
                print(f"⚠️ PDF konnte nicht gelesen werden: {file_path} ({e})")
            docs.append({"text": text, "source": file_path})
    return docs

# -----------------------------
# Chunking
# -----------------------------
def chunk_text(text, chunk_size=900, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# -----------------------------
# Jina Embedding mit Retry
# -----------------------------
def jina_embed(texts, retries=3, delay=5):
    if not JINA_API_KEY:
        raise RuntimeError("JINA_API_KEY nicht gesetzt.")
    url = "https://api.jina.ai/v1/embeddings"
    headers = {"Authorization": f"Bearer {JINA_API_KEY}"}
    payload = {"model": JINA_MODEL, "input": texts}

    for attempt in range(retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=120)
            r.raise_for_status()
            return [d["embedding"] for d in r.json()["data"]]
        except requests.exceptions.Timeout:
            print(f"⏳ Timeout bei Jina, neuer Versuch {attempt+1}/{retries} in {delay}s …")
            time.sleep(delay)
        except Exception as e:
            raise RuntimeError(f"Jina-API Fehler: {e}")
    raise RuntimeError("Jina-API hat nach mehreren Versuchen nicht geantwortet.")

# -----------------------------
# Qdrant Collection sicherstellen
# -----------------------------
def ensure_collection(vector_size):
    try:
        qdr.get_collection(QDRANT_COLLECTION)
        print(f"✅ Collection {QDRANT_COLLECTION} existiert bereits.")
    except Exception:
        try:
            qdr.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"✅ Collection {QDRANT_COLLECTION} erstellt.")
        except Exception as e:
            raise RuntimeError(
                f"Qdrant: Collection konnte nicht erstellt werden ({e}). "
                f"Bitte manuell im Dashboard mit Vector size={vector_size} & Distance=Cosine anlegen."
            )

# -----------------------------
# BM25 Index erstellen
# -----------------------------
def build_bm25(docs):
    tokenized_corpus = [doc["text"].lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

# -----------------------------
# Hauptfunktion
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-rebuild", action="store_true")
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    print("📂 Sammle & chunke Dokumente …")
    raw_docs = load_documents()
    chunks = []
    for doc in raw_docs:
        for chunk in chunk_text(doc["text"]):
            chunks.append({"text": chunk, "source": doc["source"]})
    print(f"Dokument-Chunks: {len(chunks)}")

    print("🔄 Erstelle BM25 …")
    bm25 = build_bm25(chunks)

    os.makedirs("index", exist_ok=True)
    with open("index/bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open("index/docs.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("🧠 Erstelle/aktualisiere Qdrant …")
    vecs = jina_embed([c["text"] for c in chunks])
    ensure_collection(len(vecs[0]))

    points = [
        PointStruct(
            id=i,
            vector=vecs[i],
            payload={"text": chunks[i]["text"], "source": chunks[i]["source"]}
        )
        for i in range(len(chunks))
    ]
    qdr.upsert(collection_name=QDRANT_COLLECTION, points=points)
    print("✅ Fertig! Index erstellt.")

if __name__ == "__main__":
    main()

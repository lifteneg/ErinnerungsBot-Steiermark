# ingest.py – Index-Aufbau für ErinnerungsBot Steiermark
import os
import argparse
import pickle
import json
from pathlib import Path

import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models
from rank_bm25 import BM25Okapi

# Pfade
DATA_DIR = Path("./data")
INDEX_DIR = Path("./index"); INDEX_DIR.mkdir(exist_ok=True)
BM25_FILE = INDEX_DIR / "bm25.pkl"
DOCS_FILE = INDEX_DIR / "docs.pkl"
STATE_FILE = INDEX_DIR / "ingest_state.json"

# Qdrant-Konfiguration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = "docs_bge_m3"

# Jina-Konfiguration
JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_MODEL = os.getenv("JINA_MODEL", "jina-embeddings-v2-base-de")
JINA_URL = "https://api.jina.ai/v1/embeddings"

# Chunk-Größe
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def jina_embed(texts):
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

def chunk_text(text):
    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk = " ".join(words[i:i+CHUNK_SIZE])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def load_documents():
    docs = []
    for file in DATA_DIR.rglob("*.*"):
        if file.suffix.lower() not in [".txt", ".md"]:
            continue
        text = file.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            docs.append({
                "text": chunk,
                "source": str(file),
                "chunk_id": idx
            })
    return docs

def build_bm25(docs):
    corpus = [d["text"] for d in docs]
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(BM25_FILE, "wb") as f:
        pickle.dump(bm25, f)
    with open(DOCS_FILE, "wb") as f:
        pickle.dump(docs, f)

def build_qdrant(docs, clear=False):
    qdr = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    if clear:
        try:
            qdr.delete_collection(QDRANT_COLLECTION)
        except:
            pass
    if QDRANT_COLLECTION not in [c.name for c in qdr.get_collections().collections]:
        qdr.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=len(jina_embed(["probe"])[0]),
                distance=models.Distance.COSINE
            )
        )
    vectors = jina_embed([d["text"] for d in docs])
    points = []
    for i, (doc, vec) in enumerate(zip(docs, vectors)):
        points.append(models.PointStruct(
            id=i,
            vector=vec,
            payload=doc
        ))
    qdr.upsert(collection_name=QDRANT_COLLECTION, points=points)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-rebuild", action="store_true", help="Neuaufbau aus allen Daten")
    parser.add_argument("--clear", action="store_true", help="Qdrant-Collection vorab leeren")
    args = parser.parse_args()

    if not QDRANT_URL or not QDRANT_API_KEY:
        raise RuntimeError("QDRANT_URL / QDRANT_API_KEY nicht gesetzt")
    if not JINA_API_KEY:
        raise RuntimeError("JINA_API_KEY nicht gesetzt")

    print("📂 Lade Dokumente …")
    docs = load_documents()
    print(f"Gefundene Dokument-Chunks: {len(docs)}")

    if args.full_rebuild:
        print("🔄 Erstelle BM25-Index …")
        build_bm25(docs)
        print("🧠 Erstelle Qdrant-Vektorindex …")
        build_qdrant(docs, clear=args.clear)
    else:
        print("➕ Füge neue Dokumente hinzu …")
        # TODO: Inkrementelles Update implementieren
        build_qdrant(docs, clear=False)

    # Status speichern
    STATE_FILE.write_text(json.dumps({
        "timestamp": time.time(),
        "docs": len(docs)
    }))
    print("✅ Fertig.")

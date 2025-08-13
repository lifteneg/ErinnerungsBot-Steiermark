import os
import sys
import argparse
import pickle
import requests
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models
from rank_bm25 import BM25Okapi
from tqdm import tqdm

# ------------------------------------------------
# ENV / Secrets
# ------------------------------------------------
JINA_API_KEY = os.getenv("JINA_API_KEY") or ""
JINA_MODEL = os.getenv("JINA_MODEL", "jina-embeddings-v2-base-de")
QDRANT_URL = os.getenv("QDRANT_URL") or ""
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or ""
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docs_bge_m3")

if not JINA_API_KEY:
    raise RuntimeError("JINA_API_KEY nicht gesetzt.")

# ------------------------------------------------
# Chunking Funktion
# ------------------------------------------------
def chunk_text(text, size=700, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

# ------------------------------------------------
# Jina Embedding Funktion mit Fehlerhandling
# ------------------------------------------------
def jina_embed(texts, batch_size=32):
    url = "https://api.jina.ai/v1/embeddings"
    headers = {"Authorization": f"Bearer {JINA_API_KEY}"}
    vectors = []

    for i in range(0, len(texts), batch_size):
        batch = [t for t in texts[i:i + batch_size] if t.strip()]
        if not batch:
            continue
        payload = {"model": JINA_MODEL, "input": batch}

        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            if r.status_code == 422:
                print(f"⚠️ Jina 422-Fehler – Batch übersprungen ({len(batch)} Texte)")
                continue
            r.raise_for_status()
            vectors.extend([d["embedding"] for d in r.json()["data"]])
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Jina-API Fehler: {e}")
            continue

    return vectors

# ------------------------------------------------
# BM25 Index bauen
# ------------------------------------------------
def build_bm25(docs):
    tokenized_corpus = [doc["text"].lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    with open("index/bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open("index/docs.pkl", "wb") as f:
        pickle.dump(docs, f)
    print(f"✅ BM25-Index mit {len(docs)} Dokumenten gespeichert.")

# ------------------------------------------------
# Qdrant Index bauen
# ------------------------------------------------
def build_qdrant(docs, clear=False):
    qdr = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    dim = 768  # Jina Embeddings Dimension

    if clear:
        try:
            qdr.delete_collection(QDRANT_COLLECTION)
        except Exception as e:
            print(f"⚠️ Collection konnte nicht gelöscht werden: {e}")

    # Existenz prüfen und ggf. anlegen
    if not qdr.collection_exists(QDRANT_COLLECTION):
        qdr.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
        )

    # Embeddings erzeugen und hochladen
    vectors = jina_embed([d["text"] for d in docs])
    points = [
        models.PointStruct(id=i, vector=vectors[i], payload=docs[i])
        for i in range(len(vectors))
    ]
    qdr.upsert(collection_name=QDRANT_COLLECTION, points=points)
    print(f"✅ Qdrant: {len(points)} Vektoren hochgeladen.")

# ------------------------------------------------
# Main
# ------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-rebuild", action="store_true")
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    os.makedirs("index", exist_ok=True)

    # Daten sammeln
    docs = []
    for path in Path("data").rglob("*"):
        if path.suffix.lower() in [".txt", ".md", ".pdf"]:
            try:
                if path.suffix.lower() == ".pdf":
                    from PyPDF2 import PdfReader
                    reader = PdfReader(path)
                    text = "\n".join(page.extract_text() or "" for page in reader.pages)
                else:
                    text = path.read_text(encoding="utf-8")
            except Exception as e:
                print(f"⚠️ Fehler beim Lesen von {path}: {e}")
                continue

            for chunk in chunk_text(text):
                docs.append({"text": chunk, "source": str(path)})

    print(f"📂 Gesammelte Dokument-Chunks: {len(docs)}")

    # BM25
    build_bm25(docs)

    # Qdrant
    build_qdrant(docs, clear=args.clear)


if __name__ == "__main__":
    main()

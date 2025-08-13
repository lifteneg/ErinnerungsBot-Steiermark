# ingest.py – Index-Aufbau für ErinnerungsBot Steiermark
from __future__ import annotations
import os, sys, re, pickle, argparse, time
from pathlib import Path
from typing import List, Dict
import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from rank_bm25 import BM25Okapi
from PyPDF2 import PdfReader

# ---------- CLI-Parameter ----------
parser = argparse.ArgumentParser()
parser.add_argument("--full-rebuild", action="store_true")
parser.add_argument("--clear", action="store_true")
parser.add_argument("--jina_api_key", type=str, default=os.getenv("JINA_API_KEY"))
parser.add_argument("--jina_model", type=str, default=os.getenv("JINA_MODEL", "jina-embeddings-v2-base-de"))
parser.add_argument("--qdrant_url", type=str, default=os.getenv("QDRANT_URL"))
parser.add_argument("--qdrant_api_key", type=str, default=os.getenv("QDRANT_API_KEY"))
parser.add_argument("--qdrant_collection", type=str, default=os.getenv("QDRANT_COLLECTION", "docs_bge_m3"))
args = parser.parse_args()

JINA_API_KEY = (args.jina_api_key or "").strip()
JINA_MODEL   = (args.jina_model or "").strip()
QDRANT_URL   = (args.qdrant_url or "").strip()
QDRANT_API_KEY = (args.qdrant_api_key or "").strip()
QDRANT_COLLECTION = (args.qdrant_collection or "docs_bge_m3").strip()

if not JINA_API_KEY:
    raise RuntimeError("JINA_API_KEY nicht gesetzt.")
if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("Qdrant-URL oder API-Key fehlen.")

DATA_DIR = Path("./data")
INDEX_DIR = Path("./index"); INDEX_DIR.mkdir(exist_ok=True)
BM25_FILE = INDEX_DIR / "bm25.pkl"
DOCS_FILE = INDEX_DIR / "docs.pkl"

# ---------- Jina Embeddings ----------
JINA_URL = "https://api.jina.ai/v1/embeddings"

def jina_embed(texts: list[str]) -> list[list[float]]:
    headers = {"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"}
    out: list[list[float]] = []
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

# ---------- Dateien lesen ----------
def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="ignore")

def read_pdf_file(path: Path) -> str:
    text = ""
    with open(path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def load_documents() -> List[Dict]:
    docs = []
    for file in DATA_DIR.glob("**/*"):
        if file.suffix.lower() in [".txt", ".md"]:
            text = read_text_file(file)
        elif file.suffix.lower() == ".pdf":
            text = read_pdf_file(file)
        else:
            continue
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        chunk_size = 900
        overlap = 150
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i+chunk_size]
            docs.append({"text": chunk, "source": str(file), "chunk_id": len(docs)})
    return docs

# ---------- BM25 ----------
def build_bm25(docs: List[Dict]):
    bm25 = BM25Okapi([d["text"].lower().split() for d in docs])
    with open(BM25_FILE, "wb") as f:
        pickle.dump(bm25, f)
    with open(DOCS_FILE, "wb") as f:
        pickle.dump(docs, f)

# ---------- Qdrant ----------
def ensure_collection(qdr: QdrantClient, dim: int, clear: bool = False):
    try:
        exists = qdr.collection_exists(QDRANT_COLLECTION)
    except Exception:
        exists = False
    if exists and clear:
        try:
            qdr.delete_collection(QDRANT_COLLECTION)
        except Exception:
            pass
        exists = False
    if not exists:
        try:
            qdr.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE)
            )
        except Exception as e:
            raise RuntimeError(
                f"Qdrant: Collection existiert nicht und dein API-Key darf sie nicht anlegen: {e}\n"
                f"→ Bitte im Qdrant-Dashboard Collection '{QDRANT_COLLECTION}' anlegen "
                f"(Vector size={dim}, Distance=Cosine) oder Key mit Create-Rechten nutzen."
            )

def build_qdrant(docs: List[Dict], clear: bool = False):
    qdr = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    dim = 768
    ensure_collection(qdr, dim, clear=clear)
    vectors = jina_embed([d["text"] for d in docs])
    points = []
    for i, (doc, vec) in enumerate(zip(docs, vectors)):
        points.append(qmodels.PointStruct(id=i, vector=vec, payload=doc))
    qdr.upsert(collection_name=QDRANT_COLLECTION, points=points)

# ---------- Main ----------
def main():
    print("📂 Sammle & chunke Dokumente …")
    docs = load_documents()
    print(f"Dokument-Chunks: {len(docs)}")
    print("🔄 Erstelle BM25 …")
    build_bm25(docs)
    print("🧠 Erstelle/aktualisiere Qdrant …")
    build_qdrant(docs, clear=args.clear)
    print("✅ Fertig.")

if __name__ == "__main__":
    main()

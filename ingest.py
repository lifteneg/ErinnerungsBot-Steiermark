# ingest.py – Index-Aufbau für ErinnerungsBot Steiermark (Jina + Qdrant + BM25)
import os, argparse, pickle, json, time, hashlib
from pathlib import Path
import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models
from rank_bm25 import BM25Okapi

DATA_DIR = Path("./data")
INDEX_DIR = Path("./index"); INDEX_DIR.mkdir(exist_ok=True)
BM25_FILE = INDEX_DIR / "bm25.pkl"
DOCS_FILE = INDEX_DIR / "docs.pkl"
STATE_FILE = INDEX_DIR / "ingest_state.json"

def _normalize_qdrant_url(raw: str | None) -> str | None:
    if not raw: return None
    if raw.startswith("http") and ":" not in raw.split("//", 1)[1]:
        return raw.rstrip("/") + ":6333"
    return raw.rstrip("/")

QDRANT_URL = _normalize_qdrant_url(os.getenv("QDRANT_URL"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docs_bge_m3")

JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_MODEL = os.getenv("JINA_MODEL", "jina-embeddings-v2-base-de")
JINA_URL = os.getenv("JINA_EMBED_URL", "https://api.jina.ai/v1/embeddings")

CHUNK_CHARS = 900
CHUNK_OVERLAP = 150

def jina_embed(texts):
    if not JINA_API_KEY:
        raise RuntimeError("JINA_API_KEY fehlt.")
    headers = {"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"}
    out = []
    B = 128
    for i in range(0, len(texts), B):
        payload = {"model": JINA_MODEL, "input": texts[i:i+B]}
        r = requests.post(JINA_URL, headers=headers, json=payload, timeout=60)
        if r.status_code >= 400:
            raise RuntimeError(f"Jina-API Fehler {r.status_code}: {r.text[:200]}")
        out.extend([d["embedding"] for d in r.json()["data"]])
    return out

def chunk_text(text: str) -> list[str]:
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    chunks, i = [], 0
    while i < len(text):
        end = min(i + CHUNK_CHARS, len(text))
        chunks.append(text[i:end])
        if end == len(text): break
        i = max(0, end - CHUNK_OVERLAP)
    return chunks

def load_documents():
    docs = []
    for p in DATA_DIR.rglob("*"):
        if not p.is_file(): continue
        if p.suffix.lower() not in {".txt", ".md"}:
            continue
        raw = p.read_text(encoding="utf-8", errors="ignore")
        for j, ch in enumerate(chunk_text(raw)):
            docs.append({"text": ch, "source": str(p), "chunk_id": j})
    return docs

def build_bm25(docs):
    tokenized = [d["text"].lower().split() for d in docs] or [[]]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_FILE, "wb") as f: pickle.dump(bm25, f)
    with open(DOCS_FILE, "wb") as f: pickle.dump(docs, f)

def build_qdrant(docs, clear=False):
    qdr = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    if clear:
        try: qdr.delete_collection(QDRANT_COLLECTION)
        except: pass
    # Dimension via Probe
    dim = len(jina_embed(["probe"])[0])
    try:
        info = qdr.get_collection(QDRANT_COLLECTION)
        current = info.config.params.vectors.size
        if current != dim:
            qdr.recreate_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
            )
    except Exception:
        qdr.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
        )
    vectors = jina_embed([d["text"] for d in docs])
    points = [models.PointStruct(id=int(hashlib.blake2b(f"{d['source']}-{d['chunk_id']}".encode(), digest_size=8).hexdigest(),16),
                                 vector=v, payload=d)
              for d, v in zip(docs, vectors)]
    qdr.upsert(collection_name=QDRANT_COLLECTION, points=points)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-rebuild", action="store_true")
    ap.add_argument("--clear", action="store_true")
    args = ap.parse_args()

    if not QDRANT_URL or not QDRANT_API_KEY:
        raise RuntimeError("QDRANT_URL / QDRANT_API_KEY nicht gesetzt.")
    if not JINA_API_KEY:
        raise RuntimeError("JINA_API_KEY nicht gesetzt.")

    print("📂 Lade Dokumente …")
    docs = load_documents()
    print(f"Chunks: {len(docs)}")

    if args.full_rebuild:
        print("🔄 Erstelle BM25 …")
        build_bm25(docs)
        print("🧠 Erstelle/aktualisiere Qdrant …")
        build_qdrant(docs, clear=args.clear)
    else:
        print("➕ Inkrementelles Update (einfaches Upsert) …")
        build_qdrant(docs, clear=False)

    STATE_FILE.write_text(json.dumps({"timestamp": time.time(), "docs": len(docs)}))
    print("✅ Fertig.")

if __name__ == "__main__":
    main()

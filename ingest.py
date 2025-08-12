# ingest.py – Index-Aufbau für ErinnerungsBot Steiermark
# Formate: PDF, TXT, MD  (leicht erweiterbar)

import os, argparse, pickle, json, time, hashlib
from pathlib import Path
from typing import List, Dict

import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from rank_bm25 import BM25Okapi
from PyPDF2 import PdfReader  # PDF-Support

# ---------- Pfade ----------
DATA_DIR = Path("./data"); DATA_DIR.mkdir(exist_ok=True)
INDEX_DIR = Path("./index"); INDEX_DIR.mkdir(exist_ok=True)
BM25_FILE = INDEX_DIR / "bm25.pkl"
DOCS_FILE = INDEX_DIR / "docs.pkl"
STATE_FILE = INDEX_DIR / "ingest_state.json"

# ---------- Qdrant ----------
def _normalize_qdrant_url(raw: str | None) -> str | None:
    if not raw: return None
    if raw.startswith("http") and ":" not in raw.split("//", 1)[1]:
        return raw.rstrip("/") + ":6333"
    return raw.rstrip("/")

QDRANT_URL = _normalize_qdrant_url(os.getenv("QDRANT_URL"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docs_bge_m3")

# ---------- Jina ----------
JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_MODEL   = os.getenv("JINA_MODEL", "jina-embeddings-v2-base-de")
JINA_URL     = os.getenv("JINA_EMBED_URL", "https://api.jina.ai/v1/embeddings")

# ---------- Chunking ----------
CHUNK_CHARS   = 900
CHUNK_OVERLAP = 150

def jina_embed(texts: List[str]) -> List[List[float]]:
    """Embeddings von Jina holen (ohne extra Felder → 422 vermeiden)."""
    if not JINA_API_KEY:
        raise RuntimeError("JINA_API_KEY fehlt.")
    headers = {"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"}
    out: List[List[float]] = []
    B = 128
    for i in range(0, len(texts), B):
        payload = {"model": JINA_MODEL, "input": texts[i:i+B]}
        r = requests.post(JINA_URL, headers=headers, json=payload, timeout=60)
        if r.status_code >= 400:
            raise RuntimeError(f"Jina-API Fehler {r.status_code}: {r.text[:200]}")
        out.extend([d["embedding"] for d in r.json()["data"]])
    return out

def chunk_text(text: str) -> List[str]:
    text = "\n".join([ln.strip() for ln in text.splitlines() if ln.strip()])
    chunks, i, n = [], 0, len(text)
    while i < n:
        end = min(i + CHUNK_CHARS, n)
        chunks.append(text[i:end])
        if end == n: break
        i = max(0, end - CHUNK_OVERLAP)
    return chunks

def read_pdf(path: Path) -> str:
    """Extrahiert Text aus PDF. Scans (ohne Text) liefern leeren String."""
    try:
        reader = PdfReader(str(path))
        pages = []
        for p in reader.pages:
            t = p.extract_text() or ""
            if t.strip():
                pages.append(t)
        return "\n\n".join(pages)
    except Exception as e:
        raise RuntimeError(f"PDF-Parsing fehlgeschlagen: {e}")

def load_text_from_file(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == ".pdf":
        return read_pdf(path)
    if suf in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    # Fallback (sollte hier nicht landen)
    return path.read_text(encoding="utf-8", errors="ignore")

def collect_documents() -> List[Dict]:
    allowed = {".pdf", ".txt", ".md"}
    docs: List[Dict] = []
    for p in sorted(DATA_DIR.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in allowed:
            continue
        try:
            raw = load_text_from_file(p)
        except Exception as e:
            print(f"[WARN] {p.name}: {e}"); continue
        if not raw.strip():
            print(f"[WARN] {p.name}: kein extrahierbarer Text (evtl. Scan-PDF).")
            continue
        for j, ch in enumerate(chunk_text(raw)):
            docs.append({"text": ch, "source": str(p), "chunk_id": j})
    return docs

def build_bm25(docs: List[Dict]):
    tokenized = [d["text"].lower().split() for d in docs] or [[]]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_FILE, "wb") as f: pickle.dump(bm25, f)
    with open(DOCS_FILE, "wb") as f: pickle.dump(docs, f)

def build_qdrant(docs: List[Dict], clear: bool = False):
    qdr = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    if clear:
        try: qdr.delete_collection(QDRANT_COLLECTION)
        except: pass

    # Dimension via Probe
    dim = len(jina_embed(["probe"])[0])

    # Collection passend (re)anlegen
    recreate = False
    try:
        info = qdr.get_collection(QDRANT_COLLECTION)
        current = info.config.params.vectors.size
        if current != dim:
            recreate = True
    except Exception:
        recreate = True

    if recreate:
        qdr.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE)
        )

    # Punkte upserten
    vectors = jina_embed([d["text"] for d in docs])
    points = [
        qmodels.PointStruct(
            id=int(hashlib.blake2b(f"{d['source']}-{d['chunk_id']}".encode(), digest_size=8).hexdigest(), 16),
            vector=v, payload=d
        )
        for d, v in zip(docs, vectors)
    ]
    if points:
        qdr.upsert(collection_name=QDRANT_COLLECTION, points=points)
        print(f"Upserted {len(points)} Punkte → {QDRANT_COLLECTION}")
    else:
        print("Keine Punkte zum Upsert (keine passenden Dateien/kein Text gefunden).")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-rebuild", action="store_true", help="BM25 neu + Vektoren neu")
    ap.add_argument("--clear", action="store_true", help="Collection vorher leeren")
    args = ap.parse_args()

    if not QDRANT_URL or not QDRANT_API_KEY:
        raise RuntimeError("QDRANT_URL / QDRANT_API_KEY nicht gesetzt.")
    if not JINA_API_KEY:
        raise RuntimeError("JINA_API_KEY nicht gesetzt.")

    print("📂 Sammle & chunke Dokumente …")
    docs = collect_documents()
    print(f"Dokument-Chunks: {len(docs)}")

    if args.full_rebuild:
        print("🔄 Erstelle BM25 …")
        build_bm25(docs)
        print("🧠 Erstelle/aktualisiere Qdrant …")
        build_qdrant(docs, clear=args.clear)
    else:
        print("➕ Inkrementelles Update (Upsert) …")
        build_qdrant(docs, clear=False)

    STATE_FILE.write_text(json.dumps({"timestamp": time.time(), "docs": len(docs)}))
    print("✅ Fertig.")

if __name__ == "__main__":
    main()

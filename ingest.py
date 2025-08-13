# ingest.py – stabiler Ingest mit PDF/TXT/MD, Jina-Embeddings (Batch+Retry+Sanitizing), Qdrant-Upsert, BM25-Build
from __future__ import annotations

import os
import re
import glob
import time
import json
import pickle
import argparse
from pathlib import Path
from typing import List, Dict

import requests
from rank_bm25 import BM25Okapi
from PyPDF2 import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

QDRANT_URL        = (os.getenv("QDRANT_URL") or "").strip().rstrip("/")
QDRANT_API_KEY    = (os.getenv("QDRANT_API_KEY") or "").strip()
QDRANT_COLLECTION = (os.getenv("QDRANT_COLLECTION") or "docs_bge_m3").strip()

JINA_API_KEY = (os.getenv("JINA_API_KEY") or "").strip()
JINA_MODEL   = (os.getenv("JINA_MODEL") or "jina-embeddings-v2-base-de").strip()
JINA_URL     = "https://api.jina.ai/v1/embeddings"

INDEX_DIR = Path("index"); INDEX_DIR.mkdir(exist_ok=True)
BM25_FILE = INDEX_DIR / "bm25.pkl"
DOCS_FILE = INDEX_DIR / "docs.pkl"

def sanitize_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s

def read_pdf(path: str) -> str:
    txt_parts: List[str] = []
    try:
        pdf = PdfReader(path)
        for page in pdf.pages:
            t = page.extract_text() or ""
            t = sanitize_text(t)
            if t:
                txt_parts.append(t)
    except Exception as e:
        print(f"[WARN] PDF konnte nicht gelesen werden: {path} ({e})")
    return "\n".join(txt_parts).strip()

def load_documents() -> List[Dict]:
    docs: List[Dict] = []
    for file_path in glob.glob("data/**/*", recursive=True):
        p = Path(file_path)
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        text = ""
        if suf == ".pdf":
            text = read_pdf(str(p))
        elif suf in (".txt", ".md"):
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                text = p.read_text(encoding="latin-1", errors="ignore")
        else:
            continue
        text = sanitize_text(text)
        if not text:
            continue
        docs.append({"text": text, "source": str(p)})
    return docs

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])
        if end == n:
            break
        start += chunk_size - overlap
    return chunks

def jina_embed(texts: List[str], batch_size: int = 32, max_chars: int = 6000) -> List[List[float]]:
    if not JINA_API_KEY:
        raise RuntimeError("JINA_API_KEY nicht gesetzt.")
    headers = {"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"}

    vectors: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch_raw = texts[i:i + batch_size]
        batch: List[str] = []
        for t in batch_raw:
            tt = sanitize_text(t)
            if not tt:
                continue
            if len(tt) > max_chars:
                tt = tt[:max_chars]
            batch.append(tt)

        if not batch:
            continue

        payload = {"model": JINA_MODEL, "input": batch}
        tries = 0
        while tries < 3:
            try:
                r = requests.post(JINA_URL, headers=headers, json=payload, timeout=120)
                if r.status_code >= 400:
                    print(f"[ERR] Jina HTTP {r.status_code}: {r.text[:500]}")
                    if r.status_code == 422:
                        raise RuntimeError("Jina-API 422: Ein Batch enthält unzulässige Eingaben (z. B. zu lang).")
                    r.raise_for_status()
                data = r.json().get("data", [])
                if len(data) != len(batch):
                    print(f"[WARN] Embeddings Anzahl abweichend: expected={len(batch)} got={len(data)}")
                vectors.extend([d["embedding"] for d in data])
                break
            except requests.exceptions.ReadTimeout:
                tries += 1
                print(f"[WARN] Timeout bei Jina-Batch (i={i}, try={tries}/3) – erneuter Versuch in 3s …")
                time.sleep(3)
            except Exception as e:
                raise RuntimeError(f"Jina-API Fehler: {e}")
    return vectors

def ensure_collection(qdr: QdrantClient, dim: int, clear: bool = False):
    exists = False
    try:
        exists = qdr.collection_exists(QDRANT_COLLECTION)
    except Exception:
        try:
            qdr.get_collection(QDRANT_COLLECTION)
            exists = True
        except Exception:
            exists = False

    if clear and exists:
        try:
            qdr.delete_collection(QDRANT_COLLECTION)
            exists = False
        except UnexpectedResponse as e:
            if "403" in str(e):
                print("[WARN] Delete/CLEAR nicht erlaubt für diesen Qdrant-Key (403). Fahre ohne Clear fort.")
            else:
                print(f"[WARN] Delete fehlgeschlagen: {e}. Fahre ohne Clear fort.")

    if not exists:
        try:
            qdr.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE)
            )
            print(f"[OK] Collection '{QDRANT_COLLECTION}' erstellt (dim={dim}, cosine).")
        except UnexpectedResponse as e:
            if "403" in str(e):
                raise RuntimeError(
                    f"Qdrant: Collection fehlt und API-Key darf keine Collections erstellen (403).\n"
                    f"→ Bitte '{QDRANT_COLLECTION}' manuell im Dashboard anlegen (Vector size={dim}, Distance=Cosine)."
                )
            raise
        except Exception as e:
            raise RuntimeError(f"Create-Collection fehlgeschlagen: {e}")
    else:
        try:
            info = qdr.get_collection(QDRANT_COLLECTION)
            current = info.config.params.vectors.size
            if current and current != dim:
                raise RuntimeError(
                    f"Collection '{QDRANT_COLLECTION}' hat dim={current}, benötigt dim={dim}. "
                    f"Bitte löschen und neu anlegen (oder anderen Sammlungsnamen verwenden)."
                )
        except Exception:
            pass

def save_bm25_and_docs(chunks: List[Dict]):
    tokenized = [c["text"].lower().split() for c in chunks] or [[]]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_FILE, "wb") as f:
        pickle.dump(bm25, f)
    with open(DOCS_FILE, "wb") as f:
        pickle.dump(chunks, f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-rebuild", action="store_true")
    ap.add_argument("--clear", action="store_true")
    args = ap.parse_args()

    if not QDRANT_URL or not QDRANT_API_KEY:
        raise RuntimeError("QDRANT_URL / QDRANT_API_KEY fehlen.")
    if not JINA_API_KEY:
        raise RuntimeError("JINA_API_KEY fehlt.")

    print("📂 Sammle & chunke Dokumente …")
    raw_docs = load_documents()

    chunks: List[Dict] = []
    for d in raw_docs:
        for ch in chunk_text(d["text"], 900, 150):
            ch = sanitize_text(ch)
            if not ch:
                continue
            chunks.append({"text": ch, "source": d["source"]})
    print(f"Dokument-Chunks: {len(chunks)}")

    print("🔄 Erstelle BM25 …")
    save_bm25_and_docs(chunks)

    print("🧠 Erstelle/aktualisiere Qdrant …")
    qdr = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60, prefer_grpc=False, check_compatibility=False)

    probe_vec = jina_embed(["probe"])
    if not probe_vec or not probe_vec[0]:
        raise RuntimeError("Konnte keine Probe-Embeddings von Jina erhalten.")
    dim = len(probe_vec[0])

    ensure_collection(qdr, dim, clear=args.clear)

    vectors = jina_embed([c["text"] for c in chunks])

    if len(vectors) != len(chunks):
        print(f"[WARN] Vektor-Anzahl ({len(vectors)}) != Chunk-Anzahl ({len(chunks)}). Kürze auf Minimum.")
    n = min(len(vectors), len(chunks))

    points = [
        qmodels.PointStruct(id=i, vector=vectors[i], payload=chunks[i])
        for i in range(n)
    ]

    if points:
        qdr.upsert(collection_name=QDRANT_COLLECTION, points=points)
        print(f"✅ Upserted {len(points)} Punkte → {QDRANT_COLLECTION}")
    else:
        print("ℹ️ Keine Punkte zum Upsert.")

    print("✅ Fertig.")

if __name__ == "__main__":
    main()

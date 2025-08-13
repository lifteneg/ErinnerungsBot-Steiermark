# ingest.py – stabiler Ingest mit PDF/TXT/MD, Jina-Embeddings (Batch+Retry+Sanitizing),
# Qdrant-Upsert, BM25-Build und automatischer Collection-Erstellung (falls erlaubt)

from __future__ import annotations

import os
import re
import time
import pickle
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import requests
from tqdm import tqdm
from rank_bm25 import BM25Okapi

# PDF: PyMuPDF (fitz) ist robust; fällt zurück auf PyPDF2, falls nötig
try:
    import fitz  # PyMuPDF
    _PDF_ENGINE = "pymupdf"
except Exception:
    fitz = None
    from PyPDF2 import PdfReader  # Fallback
    _PDF_ENGINE = "pypdf2"

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

# -----------------------------
# Konfiguration aus ENV/Secrets
# -----------------------------
QDRANT_URL        = (os.getenv("QDRANT_URL") or "").strip().rstrip("/")
QDRANT_API_KEY    = (os.getenv("QDRANT_API_KEY") or "").strip()
QDRANT_COLLECTION = (os.getenv("QDRANT_COLLECTION") or "docs_bge_m3").strip()

JINA_API_KEY = (os.getenv("JINA_API_KEY") or "").strip()
JINA_MODEL   = (os.getenv("JINA_MODEL") or "jina-embeddings-v2-base-de").strip()
JINA_URL     = "https://api.jina.ai/v1/embeddings"

INDEX_DIR = Path("index"); INDEX_DIR.mkdir(exist_ok=True)
BM25_FILE = INDEX_DIR / "bm25.pkl"
DOCS_FILE = INDEX_DIR / "docs.pkl"

DATA_DIR = Path("data")

# -----------------------------
# Utilities
# -----------------------------
def sanitize_text(s: str) -> str:
    """Whitespace normalisieren, Steuerzeichen entfernen."""
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s)
    # Steuerzeichen raus (außer \n lassen wir ohnehin nicht drin)
    s = "".join(ch for ch in s if ord(ch) >= 32)
    return s.strip()

def read_pdf(path: Path) -> str:
    """PDF-Text robust extrahieren (PyMuPDF bevorzugt, sonst PyPDF2)."""
    try:
        if fitz is not None:
            doc = fitz.open(path)
            texts = []
            for page in doc:
                txt = page.get_text("text") or ""
                txt = sanitize_text(txt)
                if txt:
                    texts.append(txt)
            return "\n".join(texts).strip()
        else:
            reader = PdfReader(str(path))
            texts = []
            for page in reader.pages:
                txt = page.extract_text() or ""
                txt = sanitize_text(txt)
                if txt:
                    texts.append(txt)
            return "\n".join(texts).strip()
    except Exception as e:
        print(f"[WARN] PDF konnte nicht gelesen werden: {path.name} ({e})")
        return ""

def load_documents() -> List[Dict]:
    """Sammelt alle .txt, .md, .pdf aus data/ rekursiv."""
    docs: List[Dict] = []
    if not DATA_DIR.exists():
        print(f"[WARN] Datenordner '{DATA_DIR}' existiert nicht.")
        return docs

    for p in DATA_DIR.rglob("*"):
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        text = ""
        if suf == ".pdf":
            text = read_pdf(p)
        elif suf in (".txt", ".md"):
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                text = p.read_text(encoding="latin-1", errors="ignore")
        else:
            continue

        text = sanitize_text(text)
        if text:
            docs.append({"text": text, "source": str(p)})
    return docs

def chunk_text(text: str, chunk_size: int = 700, overlap: int = 100) -> List[str]:
    """Kleinere Chunks, um Jina-422 zu vermeiden."""
    chunks: List[str] = []
    n = len(text)
    i = 0
    while i < n:
        end = min(n, i + chunk_size)
        chunks.append(text[i:end])
        if end == n:
            break
        i += (chunk_size - overlap)
    return chunks

# -----------------------------
# Jina Embeddings – robust (Batch, Retry, 422-Split)
# -----------------------------
def _jina_embed_batch(inputs: List[str], headers: Dict[str, str], timeout: int = 120) -> List[List[float]]:
    """Sendet einen Batch an Jina, behandelt Timeout/Fehler. Gibt Embedding-Liste zurück."""
    payload = {"model": JINA_MODEL, "input": inputs}
    r = requests.post(JINA_URL, headers=headers, json=payload, timeout=timeout)
    if r.status_code == 422:
        # Eingabe problematisch – wir splitten auf Einzelsätze und überspringen fehlerhafte
        vectors: List[List[float]] = []
        for t in inputs:
            try:
                r1 = requests.post(JINA_URL, headers=headers, json={"model": JINA_MODEL, "input": [t]}, timeout=timeout)
                if r1.status_code == 422:
                    # Diesen Text überspringen, ist z.B. zu lang/leer
                    print(f"[WARN] Jina 422 für Eintrag (gekürzt): {t[:120]}… – übersprungen.")
                    continue
                r1.raise_for_status()
                vectors.extend([d["embedding"] for d in r1.json()["data"]])
            except Exception as e:
                print(f"[WARN] Einzel-Embedding fehlgeschlagen: {e} – übersprungen.")
        return vectors
    r.raise_for_status()
    return [d["embedding"] for d in r.json()["data"]]

def jina_embed(texts: List[str], batch_size: int = 32, max_chars: int = 6000, max_retries: int = 3) -> List[List[float]]:
    """Bereitet Texte vor (sanitize, truncate), schickt sie in Batches an Jina mit Retries."""
    if not JINA_API_KEY:
        raise RuntimeError("JINA_API_KEY nicht gesetzt.")
    headers = {"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"}

    # Preprocess
    cleaned: List[str] = []
    for t in texts:
        tt = sanitize_text(t)
        if not tt:
            continue
        if len(tt) > max_chars:
            tt = tt[:max_chars]
        cleaned.append(tt)

    vectors: List[List[float]] = []
    for i in tqdm(range(0, len(cleaned), batch_size), desc="Jina Embeddings", unit="batch"):
        batch = cleaned[i:i + batch_size]
        tries = 0
        while True:
            try:
                vecs = _jina_embed_batch(batch, headers)
                vectors.extend(vecs)
                break
            except requests.exceptions.ReadTimeout:
                tries += 1
                if tries >= max_retries:
                    print(f"[ERR] Timeout (Batch {i//batch_size+1}), gebe auf.")
                    # wir geben auf, fahren aber mit nächsten Batches fort
                    break
                print(f"[WARN] Timeout (Batch {i//batch_size+1}), erneuter Versuch in 3s …")
                time.sleep(3)
            except requests.RequestException as e:
                tries += 1
                if tries >= max_retries:
                    print(f"[ERR] Jina-RequestException nach {max_retries} Versuchen: {e}. Batch übersprungen.")
                    break
                print(f"[WARN] Jina-Fehler: {e}. Neuer Versuch in 3s …")
                time.sleep(3)
    return vectors

# -----------------------------
# BM25 speichern
# -----------------------------
def save_bm25_and_docs(chunks: List[Dict]):
    tokenized = [c["text"].lower().split() for c in chunks] or [[]]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_FILE, "wb") as f:
        pickle.dump(bm25, f)
    with open(DOCS_FILE, "wb") as f:
        pickle.dump(chunks, f)
    print(f"[OK] BM25 gespeichert: {BM25_FILE.name}, Docs: {len(chunks)}")

# -----------------------------
# Qdrant – Collection sicherstellen
# -----------------------------
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
            print(f"[OK] Collection '{QDRANT_COLLECTION}' gelöscht (CLEAR).")
        except UnexpectedResponse as e:
            if "403" in str(e):
                print("[WARN] Delete/CLEAR nicht erlaubt (403). Fahre ohne Clear fort.")
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
                    f"Qdrant: Collection fehlt und API-Key darf keine Collections erstellen (403 Forbidden).\n"
                    f"→ Bitte im Dashboard **{QDRANT_COLLECTION}** manuell anlegen (Vector size={dim}, Distance=Cosine)."
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
                    f"Bitte löschen und neu anlegen (oder anderen Namen verwenden)."
                )
        except Exception:
            pass

# -----------------------------
# Qdrant – Upsert in Batches
# -----------------------------
def upsert_points(qdr: QdrantClient, vectors: List[List[float]], chunks: List[Dict], batch_size: int = 1000):
    n = min(len(vectors), len(chunks))
    if n == 0:
        print("[WARN] Keine Punkte zum Upsert (0).")
        return
    print(f"[OK] Upsert {n} Punkte → {QDRANT_COLLECTION}")

    for start in tqdm(range(0, n, batch_size), desc="Qdrant Upsert", unit="batch"):
        end = min(n, start + batch_size)
        points = [
            qmodels.PointStruct(
                id=start + idx,
                vector=vectors[start + idx],
                payload=chunks[start + idx],
            )
            for idx in range(end - start)
        ]
        qdr.upsert(collection_name=QDRANT_COLLECTION, points=points)

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-rebuild", action="store_true")
    ap.add_argument("--clear", action="store_true")
    args = ap.parse_args()

    if not QDRANT_URL or not QDRANT_API_KEY:
        raise RuntimeError("QDRANT_URL / QDRANT_API_KEY fehlen.")
    if not JINA_API_KEY:
        raise RuntimeError("JINA_API_KEY fehlt.")

    print("📂 Sammle Dokumente …")
    raw_docs = load_documents()

    print("🔪 Chunking …")
    chunks: List[Dict] = []
    for d in raw_docs:
        for ch in chunk_text(d["text"], 700, 100):
            ch = sanitize_text(ch)
            if ch:
                chunks.append({"text": ch, "source": d["source"]})
    print(f"[OK] Dokument-Chunks: {len(chunks)}")

    print("🧮 BM25 bauen & speichern …")
    save_bm25_and_docs(chunks)

    print("🔌 Qdrant verbinden …")
    qdr = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60,
        prefer_grpc=False,
        check_compatibility=False,
    )

    print("🧪 Probe-Embedding (Dimension ermitteln) …")
    probe = jina_embed(["probe"])
    if not probe or not probe[0]:
        raise RuntimeError("Konnte keine Probe-Embeddings von Jina erhalten.")
    dim = len(probe[0])
    print(f"[OK] Embedding-Dimension: {dim}")

    print("📦 Collection prüfen/erstellen …")
    ensure_collection(qdr, dim, clear=args.clear)

    print("🧠 Embeddings für alle Chunks berechnen …")
    vectors = jina_embed([c["text"] for c in chunks])

    print("⬆️ Upsert zu Qdrant …")
    upsert_points(qdr, vectors, chunks, batch_size=1000)

    print("✅ Ingest abgeschlossen.")

if __name__ == "__main__":
    main()

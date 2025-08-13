# ingest.py – Index-Aufbau für ErinnerungsBot Steiermark
# Unterstützte Formate: PDF, TXT, MD
# Erzeugt BM25 + pushed Vektoren nach Qdrant (Jina-Embeddings).
# Fehlt die Collection, wird sie automatisch (dim=768, Cosine) angelegt – ohne Delete.

import os, argparse, pickle, json, time, hashlib
from pathlib import Path
from typing import List, Dict

import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse
from rank_bm25 import BM25Okapi
from PyPDF2 import PdfReader

# ---------- CLI / ENV ----------
def cli_env_value(cli_val: str | None, env_key: str, default: str = "") -> str:
    """Nimmt CLI-Arg, sonst ENV, sonst Default."""
    return cli_val if cli_val is not None else os.getenv(env_key, default)

ap = argparse.ArgumentParser()
ap.add_argument("--full-rebuild", action="store_true", help="BM25 neu + Vektoren neu")
ap.add_argument("--clear", action="store_true", help="Collection vorher leeren (falls erlaubt)")
# Fallback-Parameter (falls ENV/Secrets nicht greifen)
ap.add_argument("--jina_api_key", type=str, default=None)
ap.add_argument("--jina_model", type=str, default=None)
ap.add_argument("--qdrant_url", type=str, default=None)
ap.add_argument("--qdrant_api_key", type=str, default=None)
ap.add_argument("--qdrant_collection", type=str, default=None)
args, _ = ap.parse_known_args()

# ---------- Pfade ----------
DATA_DIR = Path("./data"); DATA_DIR.mkdir(exist_ok=True)
INDEX_DIR = Path("./index"); INDEX_DIR.mkdir(exist_ok=True)
BM25_FILE = INDEX_DIR / "bm25.pkl"
DOCS_FILE = INDEX_DIR / "docs.pkl"
STATE_FILE = INDEX_DIR / "ingest_state.json"

# ---------- Konfiguration ----------
def _normalize_qdrant_url(raw: str | None) -> str | None:
    if not raw: return None
    if raw.startswith("http") and ":" not in raw.split("//", 1)[1]:
        return raw.rstrip("/") + ":6333"
    return raw.rstrip("/")

JINA_API_KEY = cli_env_value(args.jina_api_key, "JINA_API_KEY")
JINA_MODEL   = cli_env_value(args.jina_model, "JINA_MODEL", "jina-embeddings-v2-base-de")
JINA_URL     = os.getenv("JINA_EMBED_URL", "https://api.jina.ai/v1/embeddings")

QDRANT_URL        = _normalize_qdrant_url(cli_env_value(args.qdrant_url, "QDRANT_URL"))
QDRANT_API_KEY    = cli_env_value(args.qdrant_api_key, "QDRANT_API_KEY")
QDRANT_COLLECTION = cli_env_value(args.qdrant_collection, "QDRANT_COLLECTION", "docs_bge_m3")

# ---------- Chunking ----------
CHUNK_CHARS   = 900
CHUNK_OVERLAP = 150

def jina_embed(texts: List[str]) -> List[List[float]]:
    """Embeddings vom Jina-API holen (ohne Zusatzfelder → 422 vermeiden)."""
    if not JINA_API_KEY:
        raise RuntimeError("JINA_API_KEY nicht gesetzt.")
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
    """Extrahiert Text aus PDF. (Hinweis: Scan-PDFs haben evtl. keinen Text)"""
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
    # Fallback
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
            print(f"[WARN] {p.name}: kein extrahierbarer Text.")
            continue
        for j, ch in enumerate(chunk_text(raw)):
            docs.append({"text": ch, "source": str(p), "chunk_id": j})
    return docs

def build_bm25(docs: List[Dict]):
    tokenized = [d["text"].lower().split() for d in docs] or [[]]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_FILE, "wb") as f: pickle.dump(bm25, f)
    with open(DOCS_FILE, "wb") as f: pickle.dump(docs, f)

# ---------- Qdrant Helpers ----------
def ensure_collection(qdr: QdrantClient, dim: int, clear: bool):
    """Sichert, dass die Collection existiert; legt sie bei Bedarf an.
       Bei fehlenden Rechten zum Create/Delete gibt es klare Hinweise."""
    # Existenz prüfen
    exists = False
    try:
        exists = qdr.collection_exists(QDRANT_COLLECTION)
    except Exception:
        try:
            qdr.get_collection(QDRANT_COLLECTION)
            exists = True
        except Exception:
            exists = False

    if not exists:
        # versuchen, neu zu erstellen
        try:
            qdr.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE)
            )
            print(f"Collection '{QDRANT_COLLECTION}' erstellt (dim={dim}, cosine).")
            return
        except UnexpectedResponse as e:
            if "403" in str(e):
                raise RuntimeError(
                    "Qdrant: Collection existiert nicht und dein API-Key darf sie nicht anlegen (403 Forbidden).\n"
                    f"→ Bitte im Qdrant-Dashboard die Collection **{QDRANT_COLLECTION}** manuell anlegen "
                    "mit **Vector size=768** und **Distance=Cosine**, oder einen API-Key mit Create-Rechten verwenden."
                )
            raise
        except Exception as e:
            raise RuntimeError(f"Create-Collection fehlgeschlagen: {e}")

    # existiert: Dimension prüfen
    try:
        info = qdr.get_collection(QDRANT_COLLECTION)
        current = info.config.params.vectors.size
    except Exception:
        current = None

    if current and current != dim:
        raise RuntimeError(
            f"Collection '{QDRANT_COLLECTION}' hat dim={current}, benötigt dim={dim}. "
            f"Lösche sie manuell im Dashboard und lege sie neu an (oder ändere QDRANT_COLLECTION)."
        )

    # optionales Clear – nur wenn erlaubt
    if clear:
        try:
            qdr.delete_collection(QDRANT_COLLECTION)
            qdr.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE)
            )
            print(f"Collection '{QDRANT_COLLECTION}' geleert und neu erstellt.")
        except UnexpectedResponse as e:
            if "403" in str(e):
                print("[WARN] Clear/Delete ist mit diesem API-Key nicht erlaubt (403). Fahre ohne Clear fort.")
            else:
                print(f"[WARN] Clear fehlgeschlagen: {e}. Fahre ohne Clear fort.")
        except Exception as e:
            print(f"[WARN] Clear fehlgeschlagen: {e}. Fahre ohne Clear fort.")

def build_qdrant(docs: List[Dict], clear: bool = False):
    qdr = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60,
        prefer_grpc=False,
        check_compatibility=False,  # vermeidet Versionswarnung/Check
    )

    # Dimension via Probe (Jina v2 base de = 768)
    dim = len(jina_embed(["probe"])[0])

    # Collection anlegen/prüfen (ohne Delete)
    ensure_collection(qdr, dim, clear=clear)

    # Upsert der Punkte
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
        print("Keine Punkte zum Upsert.")

def main():
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

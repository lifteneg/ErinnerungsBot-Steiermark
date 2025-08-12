# ingest.py – Ingest für Qdrant + BM25 (ENV: EMBED_MODEL_NAME, QDRANT_URL/API_KEY)
from __future__ import annotations
import os, json, pickle, hashlib, argparse
from dataclasses import dataclass
from typing import List, Dict
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# ---------- Pfade ----------
DATA_DIR = Path("./data"); DATA_DIR.mkdir(exist_ok=True)
STATE_DIR = Path("./index"); STATE_DIR.mkdir(exist_ok=True)
BM25_FILE = STATE_DIR / "bm25.pkl"
DOCS_FILE = STATE_DIR / "docs.pkl"
INGEST_STATE = STATE_DIR / "ingest_state.json"

# ---------- Modelle ----------
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
HF_DIR = "/tmp/hf"  # passt zur App

# ---------- Dataklassen ----------
@dataclass
class DocChunk:
    text: str
    source: str
    chunk_id: int
    meta: Dict[str, str]

# ---------- Parser ----------
def load_text_from_file(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suf in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suf in {".xml", ".gml"}:
        from lxml import etree
        parser = etree.XMLParser(remove_blank_text=True, recover=True)
        tree = etree.parse(str(path), parser)
        root = tree.getroot()
        nsmap = {k if k is not None else 'ns': v for k, v in (root.nsmap or {}).items()}
        def tjoin(elems):
            out = []
            for x in elems:
                s = " ".join(" ".join(x.itertext()).split())
                if s: out.append(s)
            return "\n\n".join(out)
        tag_lower = etree.QName(root).localname.lower()
        # TEI
        if "tei" in tag_lower or any("tei" in (v or "") for v in nsmap.values()):
            teins = nsmap.get('tei', 'http://www.tei-c.org/ns/1.0')
            ns = {**nsmap, 'tei': teins}
            header = tree.xpath("//tei:teiHeader", namespaces=ns)
            body = tree.xpath("//tei:body//tei:p | //tei:text//tei:p | //tei:div//tei:p", namespaces=ns)
            return (f"[TEI] {path.name}\n\n" + tjoin(header) + ("\n\n" if header else "") + tjoin(body)).strip()
        # Generisches XML
        texts = []
        for elem in root.iter():
            t = (elem.text or "").strip()
            if t:
                try:
                    pth = tree.getpath(elem)
                except Exception:
                    pth = elem.tag
                texts.append(f"{pth}: {' '.join(t.split())}")
        return f"[XML] {path.name}\n" + "\n".join(texts)
    if suf in {".rdf", ".ttl", ".nt"} or (suf == ".xml" and "rdf" in path.name.lower()):
        import rdflib
        g = rdflib.Graph(); g.parse(str(path))
        def qn(x):
            try:
                return g.namespace_manager.normalizeUri(x)
            except Exception:
                return str(x)
        lines = []
        for i, (s,p,o) in enumerate(g):
            if i > 15000: break
            lines.append(f"{qn(s)} {qn(p)} {qn(o)}")
        return f"[RDF] {path.name}\n" + "\n".join(lines)
    return path.read_text(encoding="utf-8", errors="ignore")

# ---------- Chunking ----------
def chunk_text(text: str, max_chars: int = 900, overlap: int = 150) -> List[str]:
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    chunks, i = [], 0
    while i < len(text):
        end = min(i + max_chars, len(text))
        chunks.append(text[i:end])
        if end == len(text): break
        i = max(0, end - overlap)
    return chunks

def file_sig(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_mtime_ns}-{stat.st_size}"

def save_bm25(bm25, docs: List[DocChunk]):
    with open(BM25_FILE, 'wb') as f: pickle.dump(bm25, f)
    with open(DOCS_FILE, 'wb') as f: pickle.dump(docs, f)

def _normalize_qdrant_url(raw: str | None) -> str | None:
    if not raw:
        return None
    if raw.startswith("http") and ":" not in raw.split("//", 1)[1]:
        return raw.rstrip("/") + ":6333"
    return raw.rstrip("/")

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--collection', default=os.getenv("QDRANT_COLLECTION", "docs_bge_m3"))
    ap.add_argument('--full-rebuild', action='store_true')
    ap.add_argument('--clear', action='store_true', help='Collection vorher leeren')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--max-chars', type=int, default=900)
    ap.add_argument('--overlap', type=int, default=150)
    args = ap.parse_args()

    embed = SentenceTransformer(EMBED_MODEL_NAME, cache_folder=HF_DIR)
    dim = embed.get_sentence_embedding_dimension()

    qdr = QdrantClient(
        url=_normalize_qdrant_url(os.getenv('QDRANT_URL', 'http://localhost:6333')),
        api_key=os.getenv('QDRANT_API_KEY')
    )

    # Collection vorbereiten
    recreate = False
    try:
        col = qdr.get_collection(args.collection)
        # Falls Dimension nicht passt → neu anlegen
        current = col.vectors_count and col.vectors_config and col.vectors_config.size or None
        # qdrant-client liefert size je nach Version anders; sichere Variante:
        try:
            current = col.config.params.vectors.size  # type: ignore
        except Exception:
            pass
        if args.clear or (current and current != dim):
            recreate = True
    except Exception:
        recreate = True

    if recreate:
        qdr.recreate_collection(
            collection_name=args.collection,
            vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
            optimizers_config=qmodels.OptimizersConfigDiff(indexing_threshold=20000),
            hnsw_config=qmodels.HnswConfigDiff(m=16, ef_construct=256)
        )

    # Ingest-State
    state = {}
    if INGEST_STATE.exists() and not args.full_rebuild:
        try:
            state = json.loads(INGEST_STATE.read_text())
        except Exception:
            state = {}

    # Sammle Dateien
    paths = [p for p in DATA_DIR.rglob('*') if p.is_file() and p.suffix.lower() in {
        '.pdf','.txt','.md','.xml','.gml','.rdf','.ttl','.nt'
    }]

    new_docs: List[DocChunk] = []
    to_upsert = []

    for path in sorted(paths):
        sig = file_sig(path)
        if (not args.full_rebuild) and state.get(str(path)) == sig:
            continue
        try:
            raw = load_text_from_file(path)
        except Exception as e:
            print(f"[WARN] {path.name}: {e}")
            continue
        chunks = chunk_text(raw, max_chars=args.max_chars, overlap=args.overlap)
        if not chunks:
            continue
        embs = embed.encode(chunks, batch_size=args.batch, normalize_embeddings=True)
        for j, (txt, vec) in enumerate(zip(chunks, embs)):
            meta = {"source": str(path), "chunk_id": j}
            new_docs.append(DocChunk(txt, str(path), j, meta))
            pid = int(hashlib.blake2b(f"{path}-{j}".encode(), digest_size=8).hexdigest(), 16)
            to_upsert.append(qmodels.PointStruct(id=pid, vector=vec.tolist(),
                                                 payload={"text": txt, **meta}))
        state[str(path)] = sig

    if to_upsert:
        qdr.upsert(collection_name=args.collection, points=to_upsert)
        print(f"Upserted {len(to_upsert)} Punkte → {args.collection}")
    else:
        print("Keine neuen/aktualisierten Punkte für Qdrant.")

    # BM25 aktualisieren/neu bauen
    from rank_bm25 import BM25Okapi
    if args.full_rebuild:
        all_docs: List[DocChunk] = []
        for path in sorted(paths):
            try:
                raw = load_text_from_file(path)
            except Exception:
                continue
            for j, ch in enumerate(chunk_text(raw, max_chars=args.max_chars, overlap=args.overlap)):
                all_docs.append(DocChunk(ch, str(path), j, {"source": str(path), "chunk_id": j}))
        tokenized = [d.text.lower().split() for d in all_docs] or [[]]
        bm25 = BM25Okapi(tokenized)
        docs = all_docs
    else:
        if DOCS_FILE.exists():
            with open(DOCS_FILE, 'rb') as f: docs = pickle.load(f)
        else:
            docs = []
        docs.extend(new_docs)
        tokenized = [d.text.lower().split() for d in docs] or [[]]
        bm25 = BM25Okapi(tokenized)

    save_bm25(bm25, docs)
    INGEST_STATE.write_text(json.dumps(state))
    print(f"BM25 gespeichert: {BM25_FILE}, Docs: {len(docs)}")

if __name__ == '__main__':
    main()

"""
Eval-Skript (offline, ohne UI) – nutzt dieselben ENV/Modelle wie die App.
"""
from __future__ import annotations
import os, csv, argparse, json, pickle
from dataclasses import dataclass
from typing import List, Dict, Tuple
from pathlib import Path

import numpy as np
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

DATA_DIR = Path("./data"); DATA_DIR.mkdir(exist_ok=True)
STATE_DIR = Path("./index"); STATE_DIR.mkdir(exist_ok=True)
BM25_FILE = STATE_DIR / "bm25.pkl"
DOCS_FILE = STATE_DIR / "docs.pkl"
REPORTS_DIR = Path("./reports"); REPORTS_DIR.mkdir(exist_ok=True)
COLLECTION = os.getenv("QDRANT_COLLECTION", "docs_bge_m3")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "BAAI/bge-reranker-base")
DEFAULT_SYSTEM_PROMPT = (
    "Du bist ein sehr präziser Assistent. Antworte ausschließlich mit Informationen, "
    "die im bereitgestellten KONTEXT enthalten sind. Wenn der Kontext keine Antwort zulässt, "
    "sage eindeutig: 'Dazu habe ich keine Information in meinen Daten.' Erfinde nichts."
)

@dataclass
class DocChunk:
    text: str
    source: str
    chunk_id: int
    meta: Dict[str, str]

def _normalize_qdrant_url(raw: str | None) -> str | None:
    if not raw:
        return None
    if raw.startswith("http") and ":" not in raw.split("//", 1)[1]:
        return raw.rstrip("/") + ":6333"
    return raw.rstrip("/")

def get_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)

def get_reranker():
    return CrossEncoder(RERANK_MODEL_NAME)

def get_qdrant_client():
    return QdrantClient(url=_normalize_qdrant_url(os.getenv("QDRANT_URL", "http://localhost:6333")),
                        api_key=os.getenv("QDRANT_API_KEY"))

def load_bm25():
    if BM25_FILE.exists() and DOCS_FILE.exists():
        with open(BM25_FILE, 'rb') as f: bm25 = pickle.load(f)
        with open(DOCS_FILE, 'rb') as f: docs = pickle.load(f)
        return bm25, docs
    return BM25Okapi([[]]), []

def hybrid_search(query: str, topk_vec: int, topk_bm25: int, alpha: float):
    embed = get_embedder(); qdr = get_qdrant_client(); bm25, docs = load_bm25()
    # Dense
    qv = embed.encode([query], normalize_embeddings=True)[0].tolist()
    res = qdr.search(collection_name=COLLECTION, query_vector=qv, limit=topk_vec, with_payload=True)
    vec_hits = []
    for r in res:
        p = r.payload or {}
        vec_hits.append((DocChunk(p.get("text",""), p.get("source",""), int(p.get("chunk_id",0)), {k:v for k,v in p.items() if k!="text"}), float(r.score)))
    # BM25
    if docs:
        tq = query.lower().split()
        scores = bm25.get_scores(tq)
        top_idx = np.argsort(scores)[-topk_bm25:][::-1]
        bm25_hits = [(docs[i], float(scores[i])) for i in top_idx if scores[i] > 0]
    else:
        bm25_hits = []
    # Normalize & fuse
    def norm(xs):
        if not xs: return []
        a = np.array([s for _, s in xs], dtype=float)
        mn, mx = a.min(), a.max()
        if mx - mn < 1e-9:
            return [(xs[i][0], 1.0) for i in range(len(xs))]
        return [(xs[i][0], float((a[i]-mn)/(mx-mn))) for i in range(len(xs))]
    vec_n, bm25_n = norm(vec_hits), norm(bm25_hits)
    scores_map: Dict[Tuple[str,int], float] = {}; obj_map: Dict[Tuple[str,int], DocChunk] = {}
    for d, s in vec_n:
        key = (d.source, d.chunk_id); scores_map[key] = max(scores_map.get(key, 0.0), alpha * s); obj_map[key] = d
    for d, s in bm25_n:
        key = (d.source, d.chunk_id); scores_map[key] = max(scores_map.get(key, 0.0), (1-alpha) * s + scores_map.get(key, 0.0)); obj_map[key] = d
    fused = sorted([(obj_map[k], v) for k, v in scores_map.items()], key=lambda x: x[1], reverse=True)
    return fused

def rerank_snippets(query: str, candidates, k_final: int):
    rr = get_reranker()
    pairs = [[query, d.text] for d,_ in candidates]
    scores = rr.predict(pairs)
    ranked = sorted([(candidates[i][0], float(scores[i])) for i in range(len(candidates))],
                    key=lambda x: x[1], reverse=True)
    return ranked[:k_final]

def make_context(snips, max_chars: int, min_score: float, max_per_source: int = 2):
    picked = []; per_src = {}; total = 0
    for d, sc in snips:
        if sc < min_score: continue
        if per_src.get(d.source, 0) >= max_per_source: continue
        if total + len(d.text) > max_chars: break
        picked.append((d, sc)); per_src[d.source] = per_src.get(d.source, 0)+1; total += len(d.text)
    ctx = "\n\n".join([d.text for d,_ in picked])
    return ctx, picked

def call_oss_api(messages, model, api_base, api_key) -> str:
    url = api_base.rstrip("/") + "/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": 0.2}
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json(); return data["choices"][0]["message"]["content"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--questions', required=True)
    ap.add_argument('--llm', action='store_true')
    ap.add_argument('--topk_vec', type=int, default=50)
    ap.add_argument('--topk_bm25', type=int, default=50)
    ap.add_argument('--alpha', type=float, default=0.6)
    ap.add_argument('--k_final', type=int, default=8)
    ap.add_argument('--min_score', type=float, default=0.25)
    ap.add_argument('--max_chars', type=int, default=2400)
    args = ap.parse_args()

    _ = get_qdrant_client(); _ = get_embedder()

    qfile = Path(args.questions)
    questions = [line.strip() for line in qfile.read_text(encoding='utf-8').splitlines()
                 if line.strip() and not line.strip().startswith('#')]

    rows = []
    for q in questions:
        fused = hybrid_search(q, args.topk_vec, args.topk_bm25, args.alpha)
        cands = fused[:max(50, args.k_final)]
        try:
            ranked = rerank_snippets(q, cands, args.k_final)
        except Exception:
            ranked = cands[:args.k_final]
        context, used = make_context(ranked, args.max_chars, args.min_score, max_per_source=2)

        answer = "(LLM deaktiviert)"
        if args.llm and used:
            api_base = os.getenv('OSS_API_BASE', '')
            api_key = os.getenv('OSS_API_KEY', '')
            model = os.getenv('OSS_MODEL', 'openai/gpt-oss-20b:free')
            if not (api_base and api_key):
                answer = "[Fehler] OSS_API_BASE/OSS_API_KEY nicht gesetzt"
            else:
                sys = {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
                user = {"role": "user", "content": f"KONTEXT:\n{context}\n\nFRAGE:\n{q}"}
                try:
                    answer = call_oss_api([sys, user], model, api_base, api_key)
                except Exception as e:
                    answer = f"[LLM Fehler] {e}"
        elif args.llm and not used:
            answer = "Dazu habe ich keine Information in meinen Daten."

        srcs = "; ".join(sorted({f"{Path(d.source).name}#${d.chunk_id}" for d,_ in used}))
        rows.append({'question': q, 'num_used': len(used), 'sources': srcs, 'answer': answer})
        print(f"Q: {q}\n  used={len(used)} sources=[{srcs}]\n")

    csv_path = REPORTS_DIR / 'eval_results.csv'
    md_path = REPORTS_DIR / 'eval_results.md'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['question','num_used','sources','answer'])
        w.writeheader(); w.writerows(rows)
    with open(md_path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(f"### Frage\n{r['question']}\n\n")
            f.write(f"**Snippets genutzt:** {r['num_used']}  ")
            f.write(f"**Quellen:** {r['sources']}\n\n")
            f.write(f"**Antwort:**\n\n{r['answer']}\n\n---\n\n")
    print(f"\nErgebnisse gespeichert:\n- CSV: {csv_path}\n- MD : {md_path}")

if __name__ == '__main__':
    main()

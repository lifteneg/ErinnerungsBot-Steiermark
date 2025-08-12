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
JINA_URL     = os.getenv("JINA_E

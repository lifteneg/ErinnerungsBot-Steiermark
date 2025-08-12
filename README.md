# 💬 ErinnerungsBot Steiermark

Ein Streamlit-Chatbot, der **nur** auf deine eigenen Dokumente aus dem Ordner `data/` zugreift.  
Technologie-Stack:

- **Streamlit** für die Benutzeroberfläche (läuft im Browser, auch mobil nutzbar)
- **Jina AI Embeddings** (`jina-embeddings-v2-base-de`) für deutsche Vektor-Repräsentationen
- **Qdrant Cloud** als Vektorspeicher
- **BM25** für schnelles Keyword-Matching
- **OpenRouter LLM** (`openai/gpt-oss-20b:free`) für die Antwortgenerierung

## 🚀 Funktionen

- Rollen-Login (Admin / Viewer) über Access Tokens
- Admins können den Index direkt aus der App heraus aktualisieren
- Antworten **nur** aus den eigenen Daten
- Hybrid-Suche (BM25 + Vektor-Suche mit Reranking)
- Unterstützung für `.txt` und `.md` Dateien
- Vollständig cloudbasiert (keine lokale GPU notwendig)

---

## 📂 Projektstruktur

├── app.py # Streamlit-App (UI, Auth, Suche, LLM-Aufruf)
├── ingest.py # CLI-Tool zum Erstellen/Aktualisieren des Indexes
├── data/ # Deine eigenen Dokumente (.txt, .md)
├── index/ # Generierter BM25-Index und ingest_state.json
├── requirements.txt
└── README.md

---

## ⚙️ Installation & Start

1. **Repository klonen**  
   ```bash
   git clone <dein-repo>
   cd <dein-repo>

   python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

ADMIN_TOKENS = "Marlene"
VIEW_TOKENS  = "Schule"

OSS_API_BASE = "https://openrouter.ai/api"
OSS_API_KEY = "<dein_openrouter_key>"
OSS_MODEL = "openai/gpt-oss-20b:free"

QDRANT_URL = "https://<deine-instanz>.gcp.cloud.qdrant.io"
QDRANT_API_KEY = "<dein_qdrant_key>"

JINA_API_KEY = "<dein_jina_key>"
JINA_MODEL = "jina-embeddings-v2-base-de"

export QDRANT_URL="https://<deine-instanz>.gcp.cloud.qdrant.io"
export QDRANT_API_KEY="<dein_qdrant_key>"
export JINA_API_KEY="<dein_jina_key>"
export JINA_MODEL="jina-embeddings-v2-base-de"
python ingest.py --full-rebuild --clear

streamlit run app.py


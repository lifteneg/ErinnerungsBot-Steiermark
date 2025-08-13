# 💬 ErinnerungsBot Steiermark

Ein Streamlit-Chatbot, der **nur** auf deine eigenen Dokumente aus dem Ordner `data/` zugreift.

## 🛠 Technologie-Stack
- **Streamlit** für die Benutzeroberfläche (läuft im Browser, auch mobil nutzbar)
- **Jina AI Embeddings** (`jina-embeddings-v2-base-de`) für deutsche Vektor-Repräsentationen
- **Qdrant Cloud** als Vektorspeicher
- **BM25** für schnelles Keyword-Matching
- **OpenRouter LLM** (`openai/gpt-oss-20b:free`) für die Antwortgenerierung

---

## 🚀 Funktionen
- Rollen-Login (Admin / Viewer) über Access Tokens
- Admins können den Index direkt aus der App heraus aktualisieren
- Antworten **nur** aus den eigenen Daten
- Hybrid-Suche (BM25 + Vektor-Suche mit Reranking)
- Unterstützung für `.txt` und `.md` Dateien
- Vollständig cloudbasiert (keine lokale GPU notwendig)

---

## 📂 Projektstruktur
```
├── app.py            # Streamlit-App (UI, Auth, Suche, LLM-Aufruf)
├── ingest.py         # CLI-Tool zum Erstellen/Aktualisieren des Indexes
├── data/             # Deine eigenen Dokumente (.txt, .md, .pdf)
├── index/            # Generierter BM25-Index und ingest_state.json
├── requirements.txt
└── README.md         # Wichtige Hinweise für Administrator*innen
└── README_Schüler.md # Anleitung für Schüler*innen
```

---

## ⚙️ Installation & Start

### 1️⃣ Repository klonen
```bash
git clone <dein-repo>
cd <dein-repo>
```

### 2️⃣ Python-Umgebung erstellen und Abhängigkeiten installieren
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3️⃣ Streamlit-Secrets setzen  
Im Streamlit Cloud Dashboard oder in `.streamlit/secrets.toml`:
```toml
ADMIN_TOKENS = "Passwort Administrator*in"
VIEW_TOKENS  = "Passwort Schüler*innen"

OSS_API_BASE = "https://openrouter.ai/api"
OSS_API_KEY  = "<dein_openrouter_key>"
OSS_MODEL    = "openai/gpt-oss-20b:free"

QDRANT_URL   = "https://<deine-instanz>.gcp.cloud.qdrant.io"
QDRANT_API_KEY = "<dein_qdrant_key>"

JINA_API_KEY = "<dein_jina_key>"
JINA_MODEL   = "jina-embeddings-v2-base-de"
```

### 4️⃣ Index aufbauen (nur beim ersten Mal oder nach Datenänderung)
```bash
export QDRANT_URL="https://<deine-instanz>.gcp.cloud.qdrant.io"
export QDRANT_API_KEY="<dein_qdrant_key>"
export JINA_API_KEY="<dein_jina_key>"
export JINA_MODEL="jina-embeddings-v2-base-de"
python ingest.py --full-rebuild --clear
```

### 5️⃣ App starten
```bash
streamlit run app.py
```

---

## 🔐 Rollen & Login
- **Admin**: Kann Index neu aufbauen oder aktualisieren
- **Viewer**: Kann nur Fragen stellen
- Login erfolgt über ein Access Token, das in den Streamlit Secrets hinterlegt ist.

---

## 📱 Nutzung
1. App im Browser aufrufen (z. B. `https://dein-bot.streamlit.app`)
2. Access Token eingeben
3. Frage stellen → Bot antwortet nur basierend auf den eigenen Daten

---

## 🛠 Wartung
- **Daten aktualisieren**: Neue Dateien in `data/` legen → Admin in der App auf „🧱 Vollständiger Rebuild“ klicken
- **Qdrant leeren**:
```bash
python ingest.py --full-rebuild --clear
```

---

## 📌 Hinweise
- Große Dateien werden automatisch in Chunks aufgeteilt (900 Zeichen, 150 Zeichen Überlappung)
- Funktioniert auch mobil auf Smartphone/Tablet
- Bei Fehlern in Jina oder Qdrant: Keys & URL in den Secrets prüfen

---

## 📄 Lizenz
Dieses Projekt ist privat für schulische Zwecke gedacht.

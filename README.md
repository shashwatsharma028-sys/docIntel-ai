# DocIntel AI — Multi-Domain Document Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.0-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Gemini_2.0_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-FF6F00?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> An enterprise-grade RAG (Retrieval-Augmented Generation) platform that lets users upload PDFs and query them in natural language — across 4 specialized domains with source citations, confidence scoring, and audit logging.

---

## 🎯 Problem Statement

Organizations and individuals deal with large volumes of documents — legal contracts, medical reports, company SOPs, research papers — that are time-consuming to read and hard to query. DocIntel AI solves this by enabling instant, accurate, natural language Q&A over any uploaded document, with domain-specific AI behavior for each use case.

---

## ✨ Features

- 📄 **PDF Upload & Indexing** — Upload any PDF and index it in seconds using Google Gemini Embeddings + FAISS vector search
- 🤖 **Multi-Domain AI Modes** — 4 specialized modes with tailored system prompts:
  - 🏢 **Enterprise / SOP** — Query company policies, manuals, and procedures
  - ⚖️ **Legal / Contracts** — Simplify legal language, identify risky clauses
  - 🏥 **Medical Reports** — Explain medical terms in plain language
  - 🎓 **Study Assistant** — Summarize notes, generate Q&A pairs
- 🔍 **Source Citations** — Every answer references the exact source document
- 📊 **Confidence Scoring** — High / Medium / Low confidence on every response
- 📝 **Audit Logging** — Every query logged to CSV for enterprise traceability
- 🔄 **LLM Fallback** — Gemini 2.0 Flash primary, Groq Llama 3.3 70B as fallback
- ⚡ **Fast Responses** — Sub-second retrieval with FAISS IndexFlatL2

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER (Browser)                        │
│                  React + Tailwind Frontend                   │
│         PDF Upload │ Mode Selector │ Chat Interface          │
└────────────────────────┬────────────────────────────────────┘
                         │ REST API (HTTP)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                           │
│                                                              │
│   POST /upload          POST /chat          GET /health      │
│       │                     │                               │
│       ▼                     ▼                               │
│  ┌─────────┐         ┌─────────────┐                        │
│  │ ingest  │         │  retriever  │                        │
│  │  .py    │         │    .py      │                        │
│  └────┬────┘         └──────┬──────┘                        │
│       │                     │                               │
│       ▼                     ▼                               │
│  ┌─────────────────────────────────┐                        │
│  │         FAISS Vector Index       │                        │
│  │   (gemini-embedding-001, 768d)   │                        │
│  └─────────────────────────────────┘                        │
│                     │                                        │
│                     ▼                                        │
│              ┌─────────┐                                     │
│              │  llm.py │                                     │
│              └────┬────┘                                     │
│                   │                                          │
│         ┌─────────┴──────────┐                              │
│         ▼                    ▼                              │
│  ┌─────────────┐    ┌──────────────┐                        │
│  │Gemini 2.0   │    │ Groq Llama   │                        │
│  │Flash (Primary)   │3.3 (Fallback)│                        │
│  └─────────────┘    └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

### RAG Pipeline Flow

```
PDF Upload
    │
    ▼
PyMuPDF (Text Extraction)
    │
    ▼
Text Chunking (800 tokens, 100 overlap)
    │
    ▼
Gemini Embeddings (gemini-embedding-001)
    │
    ▼
FAISS Index (IndexFlatL2) ──── Persisted to disk
    │
User Query
    │
    ▼
Query Embedding → FAISS Search → Top-4 Chunks
    │
    ▼
Domain System Prompt + Context → Gemini 2.0 Flash
    │
    ▼
Answer + Sources + Confidence Score + Audit Log
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| LLM | Google Gemini 2.0 Flash | Primary answer generation |
| LLM Fallback | Groq — Llama 3.3 70B | Fallback when Gemini fails |
| Embeddings | Gemini Embedding 001 | Document & query vectorization |
| Vector DB | FAISS (IndexFlatL2) | Semantic similarity search |
| PDF Parsing | PyMuPDF (fitz) | Text extraction from PDFs |
| Backend | FastAPI + Uvicorn | REST API server |
| Frontend | React 18 + Vite | Chat UI |
| Styling | Tailwind CSS 3 | UI components |
| Icons | Lucide React | UI icons |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- Google Gemini API key → [aistudio.google.com](https://aistudio.google.com)
- Groq API key → [console.groq.com](https://console.groq.com)

### 1. Clone the repository

```bash
git clone https://github.com/your_username/docIntel-ai.git
cd docIntel-ai
```

### 2. Backend setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment variables

Create a `.env` file in the project root:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the backend

```bash
PYTHONPATH=. uvicorn backend.main:app --reload
```

Backend runs at `http://127.0.0.1:8000`
API docs at `http://127.0.0.1:8000/docs`

### 5. Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:5173`

---

## 📖 Usage

1. **Select a document mode** from the sidebar (Enterprise, Legal, Medical, Study)
2. **Upload a PDF** by dragging and dropping or clicking the upload area
3. **Wait for indexing** (usually 10–30 seconds depending on PDF size)
4. **Ask questions** in natural language in the chat interface
5. **View sources** by clicking the sources toggle under each answer
6. **Check confidence** — green = high, yellow = medium, red = low

### Example Queries by Mode

| Mode | Example Query |
|------|--------------|
| 🏢 Enterprise | "What is the leave policy for new employees?" |
| ⚖️ Legal | "What am I agreeing to? Are there any red flags?" |
| 🏥 Medical | "What does my blood test result mean?" |
| 🎓 Study | "Summarize the key concepts in chapter 3" |

---

## 📁 Project Structure

```
docIntel-ai/
├── backend/
│   ├── __init__.py
│   ├── ingest.py        # PDF parsing, chunking, embedding, FAISS indexing
│   ├── retriever.py     # Semantic search over FAISS index
│   ├── llm.py           # Gemini + Groq integration, confidence scoring, audit log
│   └── main.py          # FastAPI app, REST endpoints
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Sidebar.jsx   # Mode selector + PDF uploader
│   │   │   ├── Chat.jsx      # Chat interface + input
│   │   │   └── Message.jsx   # Message bubbles + sources + confidence
│   │   ├── App.jsx           # Root component + state management
│   │   └── index.css         # Tailwind directives
│   ├── tailwind.config.js
│   └── package.json
├── data/
│   ├── pdfs/            # Uploaded PDFs (gitignored)
│   ├── index/           # FAISS index files (gitignored)
│   └── audit_log.csv    # Query audit trail (gitignored)
├── requirements.txt
├── .env                 # API keys (gitignored)
├── .gitignore
└── README.md
```

---

## 🔑 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/upload` | Upload and index a PDF |
| `POST` | `/chat` | Query the indexed documents |

### Chat Request Body

```json
{
  "question": "What are the key terms in this contract?",
  "mode": "legal"
}
```

### Chat Response

```json
{
  "answer": "The key terms include...",
  "model": "gemini-2.0-flash",
  "confidence": "high",
  "sources": ["contract.pdf"]
}
```

---

## 🔒 Enterprise Features

- **Audit Logging** — Every query, answer, mode, and source logged to `data/audit_log.csv` with timestamp
- **Source Citations** — Every answer references exact source documents
- **Confidence Scoring** — Transparency about answer reliability
- **LLM Fallback** — Automatic failover from Gemini to Groq ensures 99%+ uptime
- **Domain Isolation** — Mode switching clears chat history to prevent cross-domain confusion

---

## 🛣️ Roadmap

- [ ] Multi-file indexing without re-indexing on every upload
- [ ] Hybrid search (BM25 + semantic) for better retrieval
- [ ] User authentication and per-user document isolation
- [ ] Docker deployment
- [ ] Cloud deployment on Render / GCP

---

## 👨‍💻 Author

**Shashwat Sharma**
- GitHub: [shashwatsharma028-sys] (https://github.com/shashwatsharma028-sys/docIntel-ai)
- LinkedIn: [Shashwat Sharma](www.linkedin.com/in/shashwat-sharma-138b21213)

---

## 📄 License

This project is licensed under the MIT License.

---

*Built with ❤️ using Google Gemini, FAISS, FastAPI, and React*
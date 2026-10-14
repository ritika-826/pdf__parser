# PDF_Parser — AI-Powered PDF Question Answering System (RAG)

An enterprise-grade, retrieval-augmented generation (RAG) web application that enables users to upload PDF documents, extract text per page with PDFPlumber, split text into semantically preserved chunks, generate vector embeddings using Hugging Face Sentence Transformers, store and index chunks in ChromaDB, and ask natural-language questions grounded strictly in document context using a local Ollama LLM.

---

## Table of Contents

1. [Project Overview & Key Features](#project-overview--key-features)
2. [Architecture & RAG Workflow](#architecture--rag-workflow)
3. [Technology Stack](#technology-stack)
4. [Prerequisites](#prerequisites)
5. [Ollama Installation & Model Setup](#ollama-installation--model-setup)
6. [Backend Setup & Execution](#backend-setup--execution)
7. [Frontend Setup & Execution](#frontend-setup--execution)
8. [Environment Configuration (.env)](#environment-configuration-env)
9. [REST API Documentation](#rest-api-documentation)
10. [Testing & Verification](#testing--verification)
11. [Known Limitations & Security Considerations](#known-limitations--security-considerations)

---

## Project Overview & Key Features

- **Accurate PDF Ingestion & Extraction:** Uses PDFPlumber to parse PDF files page-by-page, retaining page numbers and metadata. Detects scanned/image-only PDFs and alerts users that OCR is required rather than silently producing blank results.
- **Strictly Grounded RAG Pipeline:** Answers questions using only facts retrieved from the selected PDF document. When the document lacks sufficient information, the model strictly responds: *"The uploaded document does not provide enough information to answer this question."*
- **Source Citations & Excerpts:** Displays the exact PDF page numbers, chunk identifiers, similarity scores, and textual excerpts alongside each answer.
- **Document Scoping & Cross-Document Isolation:** ChromaDB queries are filtered by `document_id` to guarantee that answers never leak content across unrelated files.
- **Session-Aware Chat:** Maintains in-memory conversational history for follow-up questions within the active session and document scope. Allows clearing conversation history at any time.
- **Modern Responsive UI:** Built with React, Vite, and Lucide icons. Includes a drag-and-drop file upload zone, 6-stage visual pipeline status tracker, multi-document library, and a conversational chat interface.

---

## Architecture & RAG Workflow

```
[User PDF Upload]
       │
       ▼
[PDFPlumber] ──> Page Extraction (Page 1..N) & Scanned PDF Detection
       │
       ▼
[Recursive Character Text Splitter] ──> Overlapping Chunks (500 tokens / 100 overlap)
       │
       ▼
[Hugging Face Sentence Transformers] ──> Dense Vector Embeddings (all-MiniLM-L6-v2)
       │
       ▼
[ChromaDB Local Vector DB] ──> Stored with Metadata (doc_id, page_no, chunk_id)
       │
       ├───────────────────────────────────────┐
       │                                       │
[User Question]                                │
       │                                       │
       ▼                                       ▼
[Query Embedding] ──> [Semantic Similarity Search (Filtered by doc_id)]
                               │
                               ▼
                    [Top-K Retrieved Chunks]
                               │
                               ▼
               [Context Construction & Grounding Prompt]
                               │
                               ▼
                     [Ollama Local LLM] (e.g. llama3 / qwen2.5)
                               │
                               ▼
             [Grounded Answer + Page-Level Citations]
```

---

## Technology Stack

| Layer | Component | Description |
| :--- | :--- | :--- |
| **Frontend** | React 18, Vite, CSS3 | Modern, responsive conversational UI and upload manager |
| **Backend** | Python 3.11, FastAPI, Uvicorn | High-performance asynchronous REST API |
| **PDF Extraction** | PDFPlumber | Page-by-page PDF parsing and metadata preservation |
| **LLM Orchestration** | LangChain Core / LangChain Ollama | Prompt orchestration, context construction, chat history |
| **Embeddings** | Hugging Face (`sentence-transformers/all-MiniLM-L6-v2`)| 384-dimensional dense semantic representations |
| **Vector Database** | ChromaDB | Local persistent vector database with document filtering |
| **Local LLM Runtime** | Ollama (`llama3:latest` or `smollm`)| Local private inference without cloud API keys |

---

## Prerequisites

Before starting, ensure the following software is installed on your system:

- **Windows 10 / 11** (or macOS / Linux)
- **Python 3.11** (`py -3.11 --version` or `python --version`)
- **Node.js 18+** and **npm** (`node --version`, `npm --version`)
- **Git** (`git --version`)
- **Ollama** (`ollama --version`)

---

## Ollama Installation & Model Setup

1. **Install Ollama:**
   Download and install Ollama from [https://ollama.com/download](https://ollama.com/download).

2. **Start the Ollama Server:**
   In PowerShell:
   ```powershell
   ollama serve
   ```

3. **Pull the Desired LLM Model:**
   For standard systems (8GB+ RAM):
   ```powershell
   ollama pull llama3:latest
   ```
   Or for lightweight / resource-constrained systems:
   ```powershell
   ollama pull smollm:latest
   ```

4. **Verify Ollama is Running:**
   ```powershell
   curl http://localhost:11434/api/tags
   ```

---

## Backend Setup & Execution

### 1. Navigate to the backend directory:
```powershell
cd c:\Users\ACER\OneDrive\Documents\Desktop\pdf_parser\backend
```

### 2. Create and Activate Virtual Environment:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables:
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```

### 5. Start the FastAPI Server:
```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
The backend API will be available at:
- **API Base:** `http://127.0.0.1:8000`
- **Interactive Swagger Docs:** `http://127.0.0.1:8000/docs`

---

## Frontend Setup & Execution

### 1. Open a new PowerShell terminal and navigate to frontend:
```powershell
cd c:\Users\ACER\OneDrive\Documents\Desktop\pdf_parser\frontend
```

### 2. Install Node Dependencies:
```powershell
npm install
```

### 3. Start the Vite Development Server:
```powershell
npm run dev
```
The React application will be live at:
- `http://localhost:5173`

---

## Environment Configuration (.env)

The backend configuration is managed via `backend/.env`:

| Key | Default Value | Description |
| :--- | :--- | :--- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL of the local Ollama daemon |
| `OLLAMA_MODEL` | `llama3:latest` | Target Ollama model name |
| `EMBEDDING_MODEL`| `sentence-transformers/all-MiniLM-L6-v2` | Hugging Face embedding model |
| `CHROMA_PERSIST_DIRECTORY` | `./chroma_db` | Persistent vector store directory |
| `CHUNK_SIZE` | `500` | Target chunk size in characters/tokens |
| `CHUNK_OVERLAP` | `100` | Overlap between adjacent chunks |
| `TOP_K` | `4` | Number of chunks retrieved per question |
| `MAX_FILE_SIZE_MB` | `20` | Maximum upload size in megabytes |

---

## REST API Documentation

### 1. Health Check
- **Endpoint:** `GET /health`
- **Description:** Verifies service health, Ollama connectivity, and lists available local models.
- **Response:**
  ```json
  {
    "status": "ok",
    "ollama_connected": true,
    "available_models": ["llama3:latest", "smollm:latest"],
    "active_model": "llama3:latest",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
  }
  ```

### 2. Upload PDF
- **Endpoint:** `POST /upload-pdf`
- **Content-Type:** `multipart/form-data`
- **Description:** Validates PDF, extracts text per page, chunks content, creates embeddings, and stores in ChromaDB.
- **Response:**
  ```json
  {
    "success": true,
    "message": "Document 'report.pdf' successfully processed and indexed.",
    "document": {
      "document_id": "8f3b1a20",
      "filename": "report.pdf",
      "file_size_bytes": 1048576,
      "page_count": 12,
      "chunk_count": 48,
      "uploaded_at": "2026-10-14T11:45:00",
      "status": "ready"
    },
    "stages": [
      {"name": "PDF uploaded", "status": "completed", "detail": "..."},
      {"name": "Text extracted", "status": "completed", "detail": "..."},
      {"name": "Text chunked", "status": "completed", "detail": "..."},
      {"name": "Embeddings generated", "status": "completed", "detail": "..."},
      {"name": "Vectors stored in ChromaDB", "status": "completed", "detail": "..."},
      {"name": "Document ready for questions", "status": "completed", "detail": "..."}
    ]
  }
  ```

### 3. Ask Question
- **Endpoint:** `POST /ask`
- **Content-Type:** `application/json`
- **Request Body:**
  ```json
  {
    "document_id": "8f3b1a20",
    "question": "What were the total quarterly revenues?",
    "session_id": "user_session_1",
    "top_k": 4
  }
  ```
- **Response:**
  ```json
  {
    "answer": "According to the financial statement, total quarterly revenues were $42.5 million [Page 3].",
    "document_id": "8f3b1a20",
    "session_id": "user_session_1",
    "sources": [
      {
        "page": 3,
        "chunk_id": "8f3b1a20_p3_c8",
        "text": "Total revenues for Q3 reached $42.5 million, reflecting a 12% year-over-year increase...",
        "score": 0.89
      }
    ]
  }
  ```

### 4. List Documents
- **Endpoint:** `GET /documents`
- **Response:**
  ```json
  {
    "documents": [...]
  }
  ```

### 5. Delete Document
- **Endpoint:** `DELETE /documents/{document_id}`
- **Response:**
  ```json
  {
    "success": true,
    "message": "Document '8f3b1a20' and all associated vectors deleted successfully."
  }
  ```

### 6. Clear Chat Session
- **Endpoint:** `DELETE /chat/{session_id}`
- **Response:**
  ```json
  {
    "success": true,
    "message": "Cleared 1 conversation history scopes for session 'user_session_1'.",
    "session_id": "user_session_1"
  }
  ```

---

## Testing & Verification

The project includes an automated test suite covering unit, integration, and RAG pipeline behaviors with mock embeddings and models so testing does not require live internet access or external Ollama daemons.

### Run tests:
```powershell
cd c:\Users\ACER\OneDrive\Documents\Desktop\pdf_parser\backend
.\.venv\Scripts\Activate.ps1
pytest -v
```

### Test Coverage Highlights:
- `test_pdf_service.py`: Valid PDF extraction, invalid magic bytes header, oversized files, scanned PDF detection.
- `test_chunking.py`: Recursive token/character splitting and chunk metadata preservation.
- `test_vector_store.py`: ChromaDB indexing, semantic search, cross-document isolation, vector deletion.
- `test_rag_pipeline.py`: Grounded context answering, absent answer fallback, multi-turn history isolation.
- `test_api_endpoints.py`: FastAPI endpoints `/health`, `/upload-pdf`, `/ask`, `/documents`, `/chat/{session_id}`.

---

## Known Limitations & Security Considerations

1. **Scanned / Image-Only PDFs:** PDFPlumber extracts embedded vector and text fonts. Scanned image-only PDFs will return an explicit error explaining that OCR is required.
2. **In-Memory Chat History:** Multi-turn conversation history is scoped to `(session_id, document_id)` in memory. It resets when the FastAPI backend process restarts.
3. **Local LLM Performance:** Response generation speed depends on your local hardware (CPU/GPU) and the selected model size. For lower latency on machines without dedicated GPUs, configure `OLLAMA_MODEL=smollm:latest`.

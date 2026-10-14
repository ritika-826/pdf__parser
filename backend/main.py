import os
import io
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pdfplumber
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
import httpx

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pdf_parser")

# --- Configuration ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "20"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "4"))

# --- Pydantic Data Models ---
class ProcessingStage(BaseModel):
    name: str
    status: str
    detail: Optional[str] = None

class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    file_size_bytes: int
    page_count: int
    chunk_count: int
    uploaded_at: str
    status: str

class UploadResponse(BaseModel):
    success: bool
    message: str
    document: Optional[DocumentInfo] = None
    stages: List[ProcessingStage] = []

class QuestionRequest(BaseModel):
    document_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    session_id: str = "default"
    top_k: Optional[int] = None

class SourceReference(BaseModel):
    page: int
    chunk_id: str
    text: str
    score: Optional[float] = None

class QuestionResponse(BaseModel):
    answer: str
    document_id: str
    session_id: str
    sources: List[SourceReference] = []

class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]

class ActionResponse(BaseModel):
    success: bool
    message: str
    session_id: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    ollama_connected: bool
    available_models: List[str]
    active_model: str
    embedding_model: str

# --- 1. PDF Extraction (PDFPlumber) ---
def extract_pdf_pages(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """Extract text page by page with PDFPlumber and validate against scanned/empty PDFs."""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF (.pdf) files are supported.")
    
    if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_FILE_SIZE_MB}MB size limit.")
    
    if not file_bytes.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Invalid PDF header. File is not a valid PDF.")

    pages_data = []
    total_text = 0

    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            if len(pdf.pages) == 0:
                raise HTTPException(status_code=422, detail="The PDF contains no pages.")

            for idx, page in enumerate(pdf.pages):
                extracted = page.extract_text()
                text = (extracted or "").strip()
                if text:
                    total_text += len(text)
                    pages_data.append({"page_number": idx + 1, "text": text})
    except HTTPException:
        raise
    except Exception as e:
        err_str = str(e).lower()
        if "password" in err_str or "encrypt" in err_str:
            raise HTTPException(status_code=400, detail="PDF document is password protected or encrypted.")
        raise HTTPException(status_code=400, detail=f"Corrupted or invalid PDF: {str(e)}")

    if total_text == 0 or len(pages_data) == 0:
        raise HTTPException(
            status_code=422,
            detail="No extractable text found in this PDF. The document may be scanned or image-based, which requires OCR to extract text."
        )

    return pages_data

# --- 2. Hugging Face Embeddings & ChromaDB Storage ---
class HuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._get_model().encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        if not text:
            return []
        return self._get_model().encode([text], convert_to_numpy=True, show_progress_bar=False)[0].tolist()

embedding_service = HuggingFaceEmbeddings()

class ChromaStore:
    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir, settings=ChromaSettings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection("pdf_documents")
        self.registry: Dict[str, Dict[str, Any]] = {}

    def store_document(self, doc_id: str, filename: str, file_size: int, pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        
        chunks, metadatas, ids = [], [], []
        chunk_idx = 0

        for page in pages:
            for text in splitter.split_text(page["text"]):
                clean = text.strip()
                if not clean:
                    continue
                cid = f"{doc_id}_p{page['page_number']}_c{chunk_idx}"
                ids.append(cid)
                chunks.append(clean)
                metadatas.append({
                    "document_id": doc_id,
                    "filename": filename,
                    "page_number": page["page_number"],
                    "chunk_id": cid
                })
                chunk_idx += 1

        if not chunks:
            raise ValueError("No text chunks created.")

        embeddings = embedding_service.embed_documents(chunks)
        
        try:
            self.collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
        except Exception as e:
            if "dimension" in str(e).lower():
                self.client.delete_collection("pdf_documents")
                self.collection = self.client.get_or_create_collection("pdf_documents")
                self.collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
            else:
                raise

        info = {
            "document_id": doc_id,
            "filename": filename,
            "file_size_bytes": file_size,
            "page_count": len(pages),
            "chunk_count": len(chunks),
            "uploaded_at": datetime.utcnow().isoformat(),
            "status": "ready"
        }
        self.registry[doc_id] = info
        return info

    def search(self, query: str, doc_id: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        vec = embedding_service.embed_query(query)
        res = self.collection.query(query_embeddings=[vec], n_results=top_k, where={"document_id": doc_id})
        
        sources = []
        if res and res.get("documents") and res["documents"][0]:
            docs = res["documents"][0]
            metas = res["metadatas"][0] if res.get("metadatas") else []
            dists = res["distances"][0] if res.get("distances") else []
            for i, text in enumerate(docs):
                meta = metas[i] if i < len(metas) else {}
                dist = dists[i] if i < len(dists) else None
                score = round(1.0 - (dist / 2.0), 3) if dist is not None else None
                sources.append({
                    "page": meta.get("page_number", 1),
                    "chunk_id": meta.get("chunk_id", f"c_{i}"),
                    "text": text,
                    "score": score
                })
        return sources

    def delete(self, doc_id: str) -> bool:
        existed = doc_id in self.registry
        try:
            self.collection.delete(where={"document_id": doc_id})
        except Exception:
            pass
        if existed:
            del self.registry[doc_id]
            return True
        return False

vector_store = ChromaStore()

# --- 3. LangChain & Ollama RAG Service ---
chat_history: Dict[str, List[Dict[str, str]]] = {}

def check_ollama() -> Tuple[bool, List[str]]:
    try:
        with httpx.Client(timeout=3.0) as client:
            res = client.get(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags")
            if res.status_code == 200:
                models = [m["name"] for m in res.json().get("models", []) if m.get("name")]
                return True, models
    except Exception:
        pass
    return False, []

def answer_rag(doc_id: str, question: str, session_id: str = "default", top_k: int = TOP_K) -> Dict[str, Any]:
    doc = vector_store.registry.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")

    ollama_ok, models = check_ollama()
    if not ollama_ok:
        raise HTTPException(status_code=503, detail="Ollama service is unavailable. Please run 'ollama serve'.")

    sources = vector_store.search(query=question, doc_id=doc_id, top_k=top_k)
    if not sources:
        return {
            "answer": "The uploaded document does not provide enough information to answer this question.",
            "document_id": doc_id,
            "session_id": session_id,
            "sources": []
        }

    context_str = "\n\n".join([f"[Page {s['page']}]: {s['text']}" for s in sources])
    system_prompt = (
        "You are an AI assistant answering questions about a PDF document.\n"
        "RULES:\n"
        "1. Answer ONLY using the facts from the Provided Context below.\n"
        "2. If the context does not contain the answer, respond EXACTLY: "
        "\"The uploaded document does not provide enough information to answer this question.\"\n"
        "3. Cite the page number when stating facts (e.g. '[Page 1]').\n\n"
        f"--- CONTEXT ---\n{context_str}\n--- END CONTEXT ---"
    )

    messages = [SystemMessage(content=system_prompt)]
    hist_key = f"{session_id}:{doc_id}"
    for msg in chat_history.get(hist_key, [])[-6:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=question))

    active_model = os.getenv("OLLAMA_MODEL", OLLAMA_MODEL)
    if models and active_model not in models:
        for m in models:
            if m.split(":")[0] == active_model.split(":")[0]:
                active_model = m
                break
        else:
            active_model = models[0]

    try:
        llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=active_model, temperature=0.0)
        answer = llm.invoke(messages).content.strip()
    except Exception as e:
        err_msg = str(e)
        logger.error(f"Inference error with model {active_model}: {err_msg}")
        if "requires more system memory" in err_msg.lower() or "memory" in err_msg.lower() or "status code: 500" in err_msg.lower():
            fallback_model = "smollm:latest" if "smollm:latest" in models and active_model != "smollm:latest" else None
            if fallback_model:
                logger.info(f"Retrying with lightweight model: {fallback_model}")
                try:
                    llm_fb = ChatOllama(base_url=OLLAMA_BASE_URL, model=fallback_model, temperature=0.0)
                    answer = llm_fb.invoke(messages).content.strip()
                except Exception as fb_err:
                    answer = f"System Memory Warning: Both {active_model} and {fallback_model} exceeded available RAM. Please free up system memory."
            else:
                answer = f"System Memory Warning: Model '{active_model}' requires more memory than is currently available on your system. Please close background applications."
        else:
            raise

    if hist_key not in chat_history:
        chat_history[hist_key] = []
    chat_history[hist_key].append({"role": "user", "content": question})
    chat_history[hist_key].append({"role": "assistant", "content": answer})

    return {
        "answer": answer,
        "document_id": doc_id,
        "session_id": session_id,
        "sources": sources
    }

# --- 4. FastAPI Application ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("PDF_Parser backend running...")
    yield

app = FastAPI(title="PDF_Parser", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/health", response_model=HealthResponse)
async def health():
    ollama_ok, models = check_ollama()
    return HealthResponse(
        status="ok",
        ollama_connected=ollama_ok,
        available_models=models,
        active_model=OLLAMA_MODEL,
        embedding_model=EMBEDDING_MODEL
    )

@app.post("/upload-pdf", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    filename = file.filename or "uploaded.pdf"
    content = await file.read()
    
    stages = [
        ProcessingStage(name="PDF uploaded", status="completed", detail=f"Received {len(content) / 1024:.1f} KB"),
        ProcessingStage(name="Text extracted", status="processing"),
        ProcessingStage(name="Text chunked", status="pending"),
        ProcessingStage(name="Embeddings generated", status="pending"),
        ProcessingStage(name="Vectors stored in ChromaDB", status="pending"),
        ProcessingStage(name="Document ready for questions", status="pending")
    ]

    pages = extract_pdf_pages(content, filename)
    stages[1].status = "completed"
    stages[1].detail = f"Extracted {len(pages)} pages."

    doc_id = str(uuid.uuid4())[:8]
    stages[2].status = "completed"
    stages[2].detail = "Chunked using RecursiveCharacterTextSplitter."

    doc_info = vector_store.store_document(doc_id, filename, len(content), pages)
    stages[3].status = "completed"
    stages[3].detail = f"Embedded using {EMBEDDING_MODEL}."
    stages[4].status = "completed"
    stages[4].detail = f"Indexed {doc_info['chunk_count']} vectors in ChromaDB."
    stages[5].status = "completed"
    stages[5].detail = "Document ready for questions."

    return UploadResponse(
        success=True,
        message=f"Document '{filename}' successfully indexed.",
        document=DocumentInfo(**doc_info),
        stages=stages
    )

@app.post("/ask", response_model=QuestionResponse)
async def ask(req: QuestionRequest):
    res = answer_rag(req.document_id, req.question, req.session_id, req.top_k or TOP_K)
    return QuestionResponse(
        answer=res["answer"],
        document_id=res["document_id"],
        session_id=res["session_id"],
        sources=[SourceReference(**s) for s in res["sources"]]
    )

@app.get("/documents", response_model=DocumentListResponse)
async def get_documents():
    return DocumentListResponse(documents=[DocumentInfo(**d) for d in vector_store.registry.values()])

@app.delete("/documents/{document_id}", response_model=ActionResponse)
async def delete_doc(document_id: str):
    if not vector_store.delete(document_id):
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")
    return ActionResponse(success=True, message=f"Document '{document_id}' deleted.")

@app.delete("/chat/{session_id}", response_model=ActionResponse)
async def clear_session(session_id: str):
    prefix = f"{session_id}:"
    keys = [k for k in chat_history if k.startswith(prefix) or k == session_id]
    for k in keys:
        del chat_history[k]
    return ActionResponse(success=True, message=f"Cleared history for {session_id}.", session_id=session_id)

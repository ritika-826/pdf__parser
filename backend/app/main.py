import os
import uuid
import logging
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.pdf_service import (
    PDFService,
    PDFProcessingError,
    PDFSizeExceededError,
    PDFInvalidFormatError,
    PDFEncryptedError,
    PDFCorruptedError,
    PDFEmptyError,
)
from app.vector_store import vector_store_service
from app.rag_service import rag_service

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("pdf_parser")

# --- Configuration Defaults ---
APP_NAME = os.getenv("APP_NAME", "PDF_Parser")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "20"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "4"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

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

# --- Application Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"{APP_NAME} v{APP_VERSION} starting up...")
    yield
    logger.info(f"{APP_NAME} shutting down...")

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="Simple, AI-powered PDF Question Answering System based on RAG",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Verify backend health, Ollama availability, and models."""
    ollama_ok, models = rag_service.is_ollama_available()
    return HealthResponse(
        status="ok",
        ollama_connected=ollama_ok,
        available_models=models,
        active_model=OLLAMA_MODEL,
        embedding_model=EMBEDDING_MODEL
    )

@app.post("/upload-pdf", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process PDF through extraction, chunking, and ChromaDB vector indexing."""
    stages = [
        ProcessingStage(name="PDF uploaded", status="pending"),
        ProcessingStage(name="Text extracted", status="pending"),
        ProcessingStage(name="Text chunked", status="pending"),
        ProcessingStage(name="Embeddings generated", status="pending"),
        ProcessingStage(name="Vectors stored in ChromaDB", status="pending"),
        ProcessingStage(name="Document ready for questions", status="pending"),
    ]

    filename = file.filename or "uploaded_document.pdf"

    # Stage 1: Upload & Validate
    stages[0].status = "processing"
    try:
        content = await file.read()
        file_size = len(content)
        PDFService.validate_file(filename, file_size, content, max_mb=MAX_FILE_SIZE_MB)
        stages[0].status = "completed"
        stages[0].detail = f"File '{filename}' ({file_size / 1024:.1f} KB) validated."
    except PDFInvalidFormatError as e:
        stages[0].status = "failed"
        stages[0].detail = str(e)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except PDFSizeExceededError as e:
        stages[0].status = "failed"
        stages[0].detail = str(e)
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(e))
    except Exception as e:
        stages[0].status = "failed"
        stages[0].detail = str(e)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    # Stage 2: Extract Text
    stages[1].status = "processing"
    try:
        pages_data = PDFService.extract_text_per_page(content, filename, max_mb=MAX_FILE_SIZE_MB)
        stages[1].status = "completed"
        stages[1].detail = f"Extracted text across {len(pages_data)} pages."
    except PDFEmptyError as e:
        stages[1].status = "failed"
        stages[1].detail = str(e)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except (PDFEncryptedError, PDFCorruptedError) as e:
        stages[1].status = "failed"
        stages[1].detail = str(e)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        stages[1].status = "failed"
        stages[1].detail = str(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Text extraction failed: {str(e)}")

    # Stage 3, 4, 5, 6: Chunk, Embed, Store, Ready
    doc_id = str(uuid.uuid4())[:8]
    stages[2].status = "processing"

    try:
        stages[2].status = "completed"
        stages[2].detail = "Chunked text using RecursiveCharacterTextSplitter."

        stages[3].status = "processing"
        stages[4].status = "processing"

        doc_info = vector_store_service.chunk_and_store(
            document_id=doc_id,
            filename=filename,
            file_size=file_size,
            pages_data=pages_data,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        stages[3].status = "completed"
        stages[3].detail = f"Generated embeddings via {EMBEDDING_MODEL}."

        stages[4].status = "completed"
        stages[4].detail = f"Stored {doc_info['chunk_count']} vectors into ChromaDB."

        stages[5].status = "completed"
        stages[5].detail = "Document indexed and ready for queries."

        return UploadResponse(
            success=True,
            message=f"Document '{filename}' successfully indexed.",
            document=DocumentInfo(**doc_info),
            stages=stages
        )
    except Exception as e:
        logger.error(f"Indexing error: {e}", exc_info=True)
        for stage in stages:
            if stage.status in ("processing", "pending"):
                stage.status = "failed"
                stage.detail = str(e)
                break
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Indexing failed: {str(e)}")

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Retrieve relevant chunks and generate context-grounded answer via Ollama."""
    try:
        res = rag_service.answer_question(
            document_id=request.document_id,
            question=request.question,
            session_id=request.session_id,
            top_k=request.top_k or TOP_K
        )
        return QuestionResponse(
            answer=res["answer"],
            document_id=res["document_id"],
            session_id=res["session_id"],
            sources=[SourceReference(**s) for s in res["sources"]]
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ConnectionError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.error(f"Ask question error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """List all indexed documents."""
    docs = vector_store_service.list_documents()
    return DocumentListResponse(documents=[DocumentInfo(**d) for d in docs])

@app.delete("/documents/{document_id}", response_model=ActionResponse)
async def delete_document(document_id: str):
    """Delete document and its associated vectors from ChromaDB."""
    deleted = vector_store_service.delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{document_id}' not found.")
    return ActionResponse(success=True, message=f"Document '{document_id}' and all vectors deleted.")

@app.delete("/chat/{session_id}", response_model=ActionResponse)
async def clear_chat(session_id: str):
    """Clear conversation history for a given session."""
    cleared = rag_service.clear_session_history(session_id)
    return ActionResponse(
        success=True,
        message=f"Cleared {cleared} history scope(s) for session '{session_id}'.",
        session_id=session_id
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred."}
    )

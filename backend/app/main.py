import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.routes import upload_router, questions_router, documents_router
from app.services.rag_service import rag_service
from app.models.schemas import HealthResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("pdf_parser")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"{settings.APP_NAME} v{settings.APP_VERSION} starting up...")
    logger.info(f"Ollama Base URL: {settings.OLLAMA_BASE_URL} | Model: {settings.OLLAMA_MODEL}")
    logger.info(f"Embeddings: {settings.EMBEDDING_MODEL} | ChromaDB: {settings.CHROMA_PERSIST_DIRECTORY}")
    yield
    logger.info(f"{settings.APP_NAME} shutting down...")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered PDF Question Answering System based on RAG (LangChain + ChromaDB + HuggingFace + Ollama)",
    lifespan=lifespan
)

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(upload_router)
app.include_router(questions_router)
app.include_router(documents_router)

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Check backend health, Ollama availability, and model status.
    """
    ollama_ok, models = rag_service.is_ollama_available()
    return HealthResponse(
        status="ok",
        ollama_connected=ollama_ok,
        available_models=models,
        active_model=settings.OLLAMA_MODEL,
        embedding_model=settings.EMBEDDING_MODEL
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Avoid exposing internal stack traces to users."""
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Please try again or check server logs."}
    )

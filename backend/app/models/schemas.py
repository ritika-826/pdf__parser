from typing import List, Optional
from pydantic import BaseModel, Field

class ProcessingStage(BaseModel):
    name: str
    status: str = Field(description="pending, processing, completed, or failed")
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
    document_id: str = Field(..., min_length=1, description="Target document ID")
    question: str = Field(..., min_length=1, description="Question text")
    session_id: str = Field(default="default", description="Conversation session ID")
    top_k: Optional[int] = Field(default=None, ge=1, le=20, description="Optional override for chunk retrieval count")

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

class DeleteResponse(BaseModel):
    success: bool
    message: str

class ClearChatResponse(BaseModel):
    success: bool
    message: str
    session_id: str

class HealthResponse(BaseModel):
    status: str
    ollama_connected: bool
    available_models: List[str]
    active_model: str
    embedding_model: str

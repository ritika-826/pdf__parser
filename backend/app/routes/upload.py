import uuid
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.config import settings
from app.services.pdf_service import (
    PDFService,
    PDFProcessingError,
    PDFSizeExceededError,
    PDFInvalidFormatError,
    PDFEncryptedError,
    PDFCorruptedError,
    PDFEmptyError,
)
from app.services.vector_store import vector_store_service
from app.models.schemas import UploadResponse, ProcessingStage

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Upload"])

@router.post("/upload-pdf", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and process a PDF document through the complete ingestion pipeline:
    1. PDF uploaded & validated
    2. Text extracted
    3. Text chunked
    4. Embeddings generated
    5. Vectors stored in ChromaDB
    6. Document ready for questions
    """
    stages = [
        ProcessingStage(name="PDF uploaded", status="pending"),
        ProcessingStage(name="Text extracted", status="pending"),
        ProcessingStage(name="Text chunked", status="pending"),
        ProcessingStage(name="Embeddings generated", status="pending"),
        ProcessingStage(name="Vectors stored in ChromaDB", status="pending"),
        ProcessingStage(name="Document ready for questions", status="pending"),
    ]

    filename = file.filename or "uploaded_document.pdf"
    
    # Stage 1: PDF uploaded
    stages[0].status = "processing"
    try:
        content = await file.read()
        file_size = len(content)

        PDFService.validate_file(filename, file_size, content)
        stages[0].status = "completed"
        stages[0].detail = f"File '{filename}' ({file_size / 1024:.1f} KB) uploaded and validated."
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
        stages[0].detail = f"Upload error: {str(e)}"
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    # Stage 2: Text extracted
    stages[1].status = "processing"
    try:
        pages_data = PDFService.extract_text_per_page(content, filename)
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
        stages[1].detail = f"Text extraction failed: {str(e)}"
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to extract PDF text: {str(e)}")

    # Stage 3, 4, 5, 6: Chunking, Embedding, Vector Storage, Ready
    stages[2].status = "processing"
    doc_id = str(uuid.uuid4())[:8]

    try:
        # Vector store handles chunking, embedding generation and storage
        stages[2].status = "completed"
        stages[2].detail = "Chunked pages using recursive character splitter."

        stages[3].status = "processing"
        stages[4].status = "processing"
        
        doc_info = vector_store_service.chunk_and_store(
            document_id=doc_id,
            filename=filename,
            file_size=file_size,
            pages_data=pages_data
        )

        stages[3].status = "completed"
        stages[3].detail = f"Generated embeddings using {settings.EMBEDDING_MODEL}."
        stages[4].status = "completed"
        stages[4].detail = f"Stored {doc_info.chunk_count} chunks into ChromaDB."
        
        stages[5].status = "completed"
        stages[5].detail = "Document indexed and ready for semantic queries."

        return UploadResponse(
            success=True,
            message=f"Document '{filename}' successfully processed and indexed.",
            document=doc_info,
            stages=stages
        )

    except Exception as e:
        logger.error(f"Error indexing document {filename}: {e}", exc_info=True)
        # Mark remaining active stages as failed
        for stage in stages:
            if stage.status in ("processing", "pending"):
                stage.status = "failed"
                stage.detail = str(e)
                break
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process and index document: {str(e)}"
        )

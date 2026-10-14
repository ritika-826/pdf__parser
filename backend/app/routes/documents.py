from fastapi import APIRouter, HTTPException, status
from app.services.vector_store import vector_store_service
from app.models.schemas import DocumentListResponse, DeleteResponse

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.get("", response_model=DocumentListResponse)
async def list_documents():
    """List all documents indexed in the vector store."""
    docs = vector_store_service.list_documents()
    return DocumentListResponse(documents=docs)

@router.delete("/{document_id}", response_model=DeleteResponse)
async def delete_document(document_id: str):
    """Delete a document and its associated vectors from ChromaDB."""
    deleted = vector_store_service.delete_document(document_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found."
        )
    return DeleteResponse(
        success=True,
        message=f"Document '{document_id}' and all associated vectors deleted successfully."
    )

import logging
from fastapi import APIRouter, HTTPException, status
from app.models.schemas import QuestionRequest, QuestionResponse, ClearChatResponse
from app.services.rag_service import rag_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Questions"])

@router.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Accept a natural-language question, retrieve relevant chunks from ChromaDB,
    and generate a grounded answer using Ollama LLM.
    """
    try:
        response = rag_service.answer_question(
            document_id=request.document_id,
            question=request.question,
            session_id=request.session_id,
            top_k=request.top_k
        )
        return response
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except ConnectionError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error answering question: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while generating the answer: {str(e)}"
        )

@router.delete("/chat/{session_id}", response_model=ClearChatResponse)
async def clear_chat_history(session_id: str):
    """
    Clear conversation history for a given session.
    """
    cleared = rag_service.clear_session_history(session_id)
    return ClearChatResponse(
        success=True,
        message=f"Cleared {cleared} conversation history scopes for session '{session_id}'.",
        session_id=session_id
    )

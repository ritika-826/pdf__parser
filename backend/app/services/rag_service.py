import logging
from typing import List, Dict, Tuple, Optional
import httpx
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama

from app.config import settings
from app.services.vector_store import vector_store_service
from app.models.schemas import SourceReference, QuestionResponse

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        # In-memory conversation history scoped by f"{session_id}:{document_id}"
        self._history: Dict[str, List[Dict[str, str]]] = {}

    def _get_history_key(self, session_id: str, document_id: str) -> str:
        return f"{session_id}:{document_id}"

    def get_history(self, session_id: str, document_id: str) -> List[Dict[str, str]]:
        key = self._get_history_key(session_id, document_id)
        return self._history.get(key, [])

    def clear_session_history(self, session_id: str) -> int:
        """Clear all chat histories associated with a session_id."""
        prefix = f"{session_id}:"
        keys_to_delete = [k for k in self._history if k.startswith(prefix) or k == session_id]
        for k in keys_to_delete:
            del self._history[k]
        return len(keys_to_delete)

    def is_ollama_available(self) -> Tuple[bool, List[str]]:
        """Check if Ollama service is reachable and retrieve available models."""
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/tags")
                if res.status_code == 200:
                    data = res.json()
                    models = [m.get("name") for m in data.get("models", []) if m.get("name")]
                    return True, models
        except Exception as e:
            logger.warning(f"Ollama availability check failed: {e}")
        return False, []

    def answer_question(
        self,
        document_id: str,
        question: str,
        session_id: str = "default",
        top_k: Optional[int] = None
    ) -> QuestionResponse:
        """
        Execute the complete RAG pipeline:
        1. Retrieve semantically relevant chunks from ChromaDB for document_id.
        2. Format context with page citations.
        3. Construct grounded prompt with conversation history.
        4. Query Ollama LLM.
        5. Return grounded answer with source references.
        """
        # Validate that document exists in vector store
        doc = vector_store_service.get_document(document_id)
        if not doc:
            raise ValueError(f"Document with ID '{document_id}' was not found. Please upload or select a valid document.")

        # Check Ollama status
        available, models = self.is_ollama_available()
        if not available:
            raise ConnectionError(
                f"Ollama service is not reachable at '{settings.OLLAMA_BASE_URL}'. "
                "Please verify that Ollama is installed and running locally ('ollama serve')."
            )

        # Retrieve relevant chunks from ChromaDB
        sources = vector_store_service.similarity_search(
            query=question,
            document_id=document_id,
            top_k=top_k
        )

        if not sources:
            return QuestionResponse(
                answer="The uploaded document does not provide enough information to answer this question.",
                document_id=document_id,
                session_id=session_id,
                sources=[]
            )

        # Build context string
        context_parts = []
        for idx, src in enumerate(sources, 1):
            context_parts.append(f"[Excerpt {idx} - Page {src.page}]:\n{src.text}")
        context_str = "\n\n".join(context_parts)

        # Build system prompt with strict grounding instructions
        system_prompt = (
            "You are an intelligent, precise AI assistant analyzing a PDF document.\n"
            "Your task is to answer user questions truthfully and strictly based on the document excerpts provided below.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Answer ONLY using the facts directly stated in the Provided Context below.\n"
            "2. If the provided context does not contain sufficient information to answer the question, respond EXACTLY:\n"
            "   \"The uploaded document does not provide enough information to answer this question.\"\n"
            "3. Do NOT extrapolate, hallucinate, assume, or bring in outside information that is not in the text.\n"
            "4. Cite the page number(s) (e.g. '[Page 1]') when stating facts from the context.\n"
            "5. Maintain a professional, clear, and direct tone.\n\n"
            f"--- PROVIDED CONTEXT FROM DOCUMENT '{doc.filename}' ---\n"
            f"{context_str}\n"
            "--- END OF CONTEXT ---"
        )

        messages = [SystemMessage(content=system_prompt)]

        # Append recent history for this session & document (up to 6 previous messages)
        history_key = self._get_history_key(session_id, document_id)
        current_history = self._history.get(history_key, [])
        for entry in current_history[-6:]:
            if entry.get("role") == "user":
                messages.append(HumanMessage(content=entry["content"]))
            elif entry.get("role") == "assistant":
                messages.append(AIMessage(content=entry["content"]))

        # Append current user question
        messages.append(HumanMessage(content=question))

        # Check model name fallback if configured model isn't in Ollama
        active_model = settings.OLLAMA_MODEL
        if models and active_model not in models:
            # Check without tag e.g. "llama3" vs "llama3:latest"
            matched = False
            for m in models:
                if m.split(":")[0] == active_model.split(":")[0]:
                    active_model = m
                    matched = True
                    break
            if not matched and models:
                active_model = models[0]
                logger.info(f"Model '{settings.OLLAMA_MODEL}' not found. Falling back to available model '{active_model}'.")

        # Initialize Ollama LLM
        llm = ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=active_model,
            temperature=0.0
        )

        try:
            response = llm.invoke(messages)
            answer_text = response.content.strip()
        except Exception as e:
            logger.error(f"Error during Ollama inference: {e}")
            raise RuntimeError(f"Ollama generation failed: {str(e)}") from e

        # Update in-memory history
        if history_key not in self._history:
            self._history[history_key] = []
        self._history[history_key].append({"role": "user", "content": question})
        self._history[history_key].append({"role": "assistant", "content": answer_text})

        return QuestionResponse(
            answer=answer_text,
            document_id=document_id,
            session_id=session_id,
            sources=sources
        )

# Global singleton instance
rag_service = RAGService()

import os
import logging
from typing import List, Dict, Any, Tuple, Optional
import httpx
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama

from app.vector_store import vector_store_service

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3:latest")
        self._history: Dict[str, List[Dict[str, str]]] = {}

    def _get_history_key(self, session_id: str, document_id: str) -> str:
        return f"{session_id}:{document_id}"

    def get_history(self, session_id: str, document_id: str) -> List[Dict[str, str]]:
        return self._history.get(self._get_history_key(session_id, document_id), [])

    def clear_session_history(self, session_id: str) -> int:
        prefix = f"{session_id}:"
        keys_to_delete = [k for k in self._history if k.startswith(prefix) or k == session_id]
        for k in keys_to_delete:
            del self._history[k]
        return len(keys_to_delete)

    def is_ollama_available(self) -> Tuple[bool, List[str]]:
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"{self.ollama_base_url.rstrip('/')}/api/tags")
                if res.status_code == 200:
                    data = res.json()
                    models = [m.get("name") for m in data.get("models", []) if m.get("name")]
                    return True, models
        except Exception as e:
            logger.warning(f"Ollama check failed: {e}")
        return False, []

    def answer_question(
        self,
        document_id: str,
        question: str,
        session_id: str = "default",
        top_k: int = 4
    ) -> Dict[str, Any]:
        doc = vector_store_service.get_document(document_id)
        if not doc:
            raise ValueError(f"Document with ID '{document_id}' was not found. Please upload or select a valid document.")

        available, models = self.is_ollama_available()
        if not available:
            raise ConnectionError(
                f"Ollama service is not reachable at '{self.ollama_base_url}'. "
                "Please verify that Ollama is running ('ollama serve')."
            )

        sources = vector_store_service.similarity_search(
            query=question,
            document_id=document_id,
            top_k=top_k
        )

        if not sources:
            return {
                "answer": "The uploaded document does not provide enough information to answer this question.",
                "document_id": document_id,
                "session_id": session_id,
                "sources": []
            }

        context_parts = []
        for idx, src in enumerate(sources, 1):
            context_parts.append(f"[Excerpt {idx} - Page {src['page']}]:\n{src['text']}")
        context_str = "\n\n".join(context_parts)

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
            f"--- PROVIDED CONTEXT FROM DOCUMENT '{doc.get('filename', 'document.pdf')}' ---\n"
            f"{context_str}\n"
            "--- END OF CONTEXT ---"
        )

        messages = [SystemMessage(content=system_prompt)]

        history_key = self._get_history_key(session_id, document_id)
        current_history = self._history.get(history_key, [])
        for entry in current_history[-6:]:
            if entry.get("role") == "user":
                messages.append(HumanMessage(content=entry["content"]))
            elif entry.get("role") == "assistant":
                messages.append(AIMessage(content=entry["content"]))

        messages.append(HumanMessage(content=question))

        active_model = self.ollama_model
        if models and active_model not in models:
            for m in models:
                if m.split(":")[0] == active_model.split(":")[0]:
                    active_model = m
                    break
            else:
                active_model = models[0]

        llm = ChatOllama(
            base_url=self.ollama_base_url,
            model=active_model,
            temperature=0.0
        )

        response = llm.invoke(messages)
        answer_text = response.content.strip()

        if history_key not in self._history:
            self._history[history_key] = []
        self._history[history_key].append({"role": "user", "content": question})
        self._history[history_key].append({"role": "assistant", "content": answer_text})

        return {
            "answer": answer_text,
            "document_id": document_id,
            "session_id": session_id,
            "sources": sources
        }

rag_service = RAGService()

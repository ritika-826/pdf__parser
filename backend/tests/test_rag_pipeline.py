import pytest
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama
from app.rag_service import rag_service

def test_rag_pipeline_grounded_answer(monkeypatch, temp_vector_store):
    doc_id = "doc_rag_test"
    temp_vector_store.chunk_and_store(
        document_id=doc_id,
        filename="solar_system.pdf",
        file_size=1024,
        pages_data=[
            {"page_number": 1, "text": "Jupiter is the largest planet in our solar system, with a diameter of roughly 142,984 km."}
        ]
    )

    monkeypatch.setattr(rag_service, "is_ollama_available", lambda: (True, ["llama3:latest"]))

    mock_response = AIMessage(content="Jupiter is the largest planet in the solar system, with a diameter of roughly 142,984 km [Page 1].")
    monkeypatch.setattr(ChatOllama, "invoke", lambda self, messages: mock_response)

    res = rag_service.answer_question(
        document_id=doc_id,
        question="What is the largest planet and what is its diameter?",
        session_id="session_test_1"
    )

    assert "Jupiter" in res["answer"]
    assert "142,984" in res["answer"]
    assert res["document_id"] == doc_id
    assert len(res["sources"]) > 0
    assert res["sources"][0]["page"] == 1

def test_rag_pipeline_answer_absent_from_document(monkeypatch, temp_vector_store):
    doc_id = "doc_gardening"
    temp_vector_store.chunk_and_store(
        document_id=doc_id,
        filename="gardening.pdf",
        file_size=1024,
        pages_data=[
            {"page_number": 1, "text": "Tomatoes grow best in full sunlight with well-draining soil."}
        ]
    )

    monkeypatch.setattr(rag_service, "is_ollama_available", lambda: (True, ["llama3:latest"]))

    absent_phrase = "The uploaded document does not provide enough information to answer this question."
    monkeypatch.setattr(ChatOllama, "invoke", lambda self, messages: AIMessage(content=absent_phrase))

    res = rag_service.answer_question(
        document_id=doc_id,
        question="What is the quantum spin of an electron?",
        session_id="session_test_2"
    )

    assert res["answer"] == absent_phrase

def test_session_history_isolation(monkeypatch, temp_vector_store):
    doc_id = "doc_multi_session"
    temp_vector_store.chunk_and_store(
        document_id=doc_id,
        filename="multi_session.pdf",
        file_size=1024,
        pages_data=[{"page_number": 1, "text": "Python is a high-level programming language."}]
    )

    monkeypatch.setattr(rag_service, "is_ollama_available", lambda: (True, ["llama3:latest"]))
    monkeypatch.setattr(ChatOllama, "invoke", lambda self, messages: AIMessage(content="Python is high-level."))

    rag_service.answer_question(document_id=doc_id, question="What is Python?", session_id="session_A")
    history_A = rag_service.get_history(session_id="session_A", document_id=doc_id)
    assert len(history_A) == 2

    history_B = rag_service.get_history(session_id="session_B", document_id=doc_id)
    assert len(history_B) == 0

    rag_service.clear_session_history("session_A")
    history_A_after = rag_service.get_history(session_id="session_A", document_id=doc_id)
    assert len(history_A_after) == 0

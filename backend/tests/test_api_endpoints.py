from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama
from tests.conftest import create_text_pdf_bytes
from app.rag_service import rag_service

def test_health_endpoint(client, monkeypatch):
    monkeypatch.setattr(rag_service, "is_ollama_available", lambda: (True, ["llama3:latest"]))
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["ollama_connected"] is True
    assert "llama3:latest" in data["available_models"]

def test_upload_pdf_success(client, temp_vector_store):
    pdf_bytes = create_text_pdf_bytes(["FastAPI is a modern, high-performance web framework for building APIs."])
    response = client.post(
        "/upload-pdf",
        files={"file": ("framework.pdf", pdf_bytes, "application/pdf")}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["document"]["filename"] == "framework.pdf"
    assert len(data["stages"]) == 6
    assert data["stages"][-1]["status"] == "completed"

def test_upload_pdf_invalid_type(client):
    response = client.post(
        "/upload-pdf",
        files={"file": ("readme.txt", b"plain text", "text/plain")}
    )
    assert response.status_code == 400
    assert "Only PDF" in response.json()["detail"]

def test_list_and_delete_documents(client, temp_vector_store):
    # Store a document
    temp_vector_store.chunk_and_store(
        document_id="doc_xyz",
        filename="xyz.pdf",
        file_size=1024,
        pages_data=[{"page_number": 1, "text": "Sample content."}]
    )

    # List documents
    res_list = client.get("/documents")
    assert res_list.status_code == 200
    docs = res_list.json()["documents"]
    assert any(d["document_id"] == "doc_xyz" for d in docs)

    # Delete document
    res_del = client.delete("/documents/doc_xyz")
    assert res_del.status_code == 200
    assert res_del.json()["success"] is True

    # Deleting again returns 404
    res_del_404 = client.delete("/documents/doc_xyz")
    assert res_del_404.status_code == 404

def test_ask_question_endpoint(client, temp_vector_store, monkeypatch):
    doc_id = "doc_qa"
    temp_vector_store.chunk_and_store(
        document_id=doc_id,
        filename="qa.pdf",
        file_size=1024,
        pages_data=[{"page_number": 1, "text": "Antigravity is an AI pair programmer created by Google DeepMind."}]
    )

    monkeypatch.setattr(rag_service, "is_ollama_available", lambda: (True, ["llama3:latest"]))
    monkeypatch.setattr(ChatOllama, "invoke", lambda self, messages: AIMessage(content="Antigravity is created by Google DeepMind [Page 1]."))

    payload = {
        "document_id": doc_id,
        "question": "Who created Antigravity?",
        "session_id": "test_session_qa"
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "Google DeepMind" in data["answer"]
    assert len(data["sources"]) > 0

def test_clear_chat_endpoint(client):
    response = client.delete("/chat/my_session_123")
    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["session_id"] == "my_session_123"

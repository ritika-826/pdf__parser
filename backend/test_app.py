import io
import pytest
from pypdf import PdfWriter
from fastapi.testclient import TestClient
from fastapi import HTTPException
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama

from main import (
    app,
    extract_pdf_pages,
    vector_store,
    answer_rag,
    embedding_service,
    chat_history
)

def create_text_pdf(text: str) -> bytes:
    content = "BT /F1 12 Tf 50 250 Td (" + text + ") Tj ET"
    return f"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj << /Length {len(content)} >> stream
{content}
endstream endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000244 00000 n 
0000000{305 + len(content):03d} 00000 n 
trailer << /Size 6 /Root 1 0 R >>
startxref
{380 + len(content)}
%%EOF""".encode("latin1")

def create_blank_pdf() -> bytes:
    w = PdfWriter()
    w.add_blank_page(300, 300)
    b = io.BytesIO()
    w.write(b)
    return b.getvalue()

@pytest.fixture(autouse=True)
def mock_embeddings(monkeypatch):
    def fake_embed(texts):
        return [[(ord(c) % 10) / 10.0 for c in (t * 384)[:384]] for t in texts]
    monkeypatch.setattr(embedding_service, "embed_documents", fake_embed)
    monkeypatch.setattr(embedding_service, "embed_query", lambda t: fake_embed([t])[0])

@pytest.fixture
def client():
    return TestClient(app)

def test_extract_pdf_validation():
    with pytest.raises(HTTPException) as exc:
        extract_pdf_pages(b"%PDF-1.4", "file.txt")
    assert exc.value.status_code == 400

    with pytest.raises(HTTPException) as exc:
        extract_pdf_pages(b"NOT_A_PDF", "file.pdf")
    assert exc.value.status_code == 400

    with pytest.raises(HTTPException) as exc:
        extract_pdf_pages(create_blank_pdf(), "blank.pdf")
    assert exc.value.status_code == 422
    assert "scanned" in exc.value.detail

def test_extract_pdf_success():
    pdf = create_text_pdf("Artificial Intelligence and Document Retrieval.")
    pages = extract_pdf_pages(pdf, "test.pdf")
    assert len(pages) == 1
    assert pages[0]["page_number"] == 1
    assert "Artificial Intelligence" in pages[0]["text"]

def test_vector_store_operations():
    doc_id = "test_doc_1"
    pages = [{"page_number": 1, "text": "LangChain and ChromaDB work together."}]
    info = vector_store.store_document(doc_id, "test.pdf", 1024, pages)
    assert info["document_id"] == doc_id

    results = vector_store.search("LangChain", doc_id)
    assert len(results) > 0
    assert "LangChain" in results[0]["text"]

    assert vector_store.delete(doc_id) is True

def test_rag_answering(monkeypatch):
    doc_id = "test_doc_rag"
    vector_store.store_document(doc_id, "rag.pdf", 100, [{"page_number": 1, "text": "Apollo launch code is 442200."}])

    monkeypatch.setattr("main.check_ollama", lambda: (True, ["llama3:latest"]))
    monkeypatch.setattr(ChatOllama, "invoke", lambda self, msgs: AIMessage(content="Apollo launch code is 442200 [Page 1]."))

    res = answer_rag(doc_id, "What is the launch code?", session_id="s1")
    assert "442200" in res["answer"]
    assert len(res["sources"]) > 0

def test_api_full_flow(client, monkeypatch):
    monkeypatch.setattr("main.check_ollama", lambda: (True, ["llama3:latest"]))
    monkeypatch.setattr(ChatOllama, "invoke", lambda self, msgs: AIMessage(content="FastAPI is high performance."))

    # Health
    r = client.get("/health")
    assert r.status_code == 200

    # Upload
    pdf = create_text_pdf("FastAPI is a modern web framework.")
    r_up = client.post("/upload-pdf", files={"file": ("test.pdf", pdf, "application/pdf")})
    assert r_up.status_code == 200
    doc_id = r_up.json()["document"]["document_id"]

    # Ask
    r_ask = client.post("/ask", json={"document_id": doc_id, "question": "What is FastAPI?", "session_id": "test_s"})
    assert r_ask.status_code == 200
    assert "FastAPI" in r_ask.json()["answer"]

    # Documents
    r_docs = client.get("/documents")
    assert r_docs.status_code == 200
    assert any(d["document_id"] == doc_id for d in r_docs.json()["documents"])

    # Clear Chat
    r_clear = client.delete("/chat/test_s")
    assert r_clear.status_code == 200

    # Delete
    r_del = client.delete(f"/documents/{doc_id}")
    assert r_del.status_code == 200

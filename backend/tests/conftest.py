import io
from typing import List
import pytest
from pypdf import PdfWriter
from fastapi.testclient import TestClient

from app.vector_store import embedding_service, VectorStoreService
from app.main import app

def create_sample_pdf_bytes(pages_text: List[str]) -> bytes:
    """Generate a real valid PDF in memory with blank pages."""
    writer = PdfWriter()
    for _ in pages_text:
        writer.add_blank_page(width=300, height=300)
    
    pdf_buffer = io.BytesIO()
    writer.write(pdf_buffer)
    return pdf_buffer.getvalue()

def create_text_pdf_bytes(text_lines: List[str]) -> bytes:
    """Construct a valid single/multi-page PDF with extractable text streams."""
    content_stream = "BT /F1 12 Tf 50 250 Td (" + " ".join(text_lines) + ") Tj ET"
    stream_len = len(content_stream)
    
    pdf_content = f"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length {stream_len} >>
stream
{content_stream}
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000244 00000 n 
0000000{305 + stream_len:03d} 00000 n 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
{380 + stream_len}
%%EOF"""
    return pdf_content.encode("latin1")

class MockEmbeddings:
    """Deterministic mock embedding provider for tests."""
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        for text in texts:
            vec = [0.0] * 384
            for i, ch in enumerate(text[:384]):
                vec[i] = round((ord(ch) % 10) / 10.0, 3)
            mag = sum(x*x for x in vec) ** 0.5 or 1.0
            results.append([x / mag for x in vec])
        return results

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

@pytest.fixture
def mock_embeddings(monkeypatch):
    mock = MockEmbeddings()
    monkeypatch.setattr(embedding_service, "embed_documents", mock.embed_documents)
    monkeypatch.setattr(embedding_service, "embed_query", mock.embed_query)
    return mock

@pytest.fixture
def temp_vector_store(tmp_path, monkeypatch, mock_embeddings):
    """Provide a fresh VectorStoreService pointing to a temporary directory."""
    test_store = VectorStoreService(persist_directory=str(tmp_path / "test_chroma"))
    monkeypatch.setattr("app.vector_store.vector_store_service", test_store)
    monkeypatch.setattr("app.main.vector_store_service", test_store)
    monkeypatch.setattr("app.rag_service.vector_store_service", test_store)
    return test_store

@pytest.fixture
def client(temp_vector_store):
    return TestClient(app)

import pytest

def test_chunk_and_store_and_retrieve(temp_vector_store):
    doc_id = "doc_alpha"
    filename = "alpha_doc.pdf"
    pages_data = [
        {"page_number": 1, "text": "LangChain provides modern abstractions for prompt templates and retrieval chains."},
        {"page_number": 2, "text": "ChromaDB is an open-source embedding database for AI application development."}
    ]

    doc_info = temp_vector_store.chunk_and_store(
        document_id=doc_id,
        filename=filename,
        file_size=2048,
        pages_data=pages_data,
        chunk_size=100,
        chunk_overlap=20
    )

    assert doc_info["document_id"] == doc_id
    assert doc_info["chunk_count"] >= 2
    assert doc_info["page_count"] == 2

    results = temp_vector_store.similarity_search("ChromaDB embedding database", document_id=doc_id, top_k=2)
    assert len(results) > 0
    assert any("ChromaDB" in r["text"] for r in results)
    assert results[0]["page"] in [1, 2]

def test_document_scoping_prevents_leakage(temp_vector_store):
    # Store Doc 1
    temp_vector_store.chunk_and_store(
        document_id="doc_1",
        filename="first.pdf",
        file_size=1024,
        pages_data=[{"page_number": 1, "text": "Secret code for project Apollo is 998811."}]
    )

    # Store Doc 2
    temp_vector_store.chunk_and_store(
        document_id="doc_2",
        filename="second.pdf",
        file_size=1024,
        pages_data=[{"page_number": 1, "text": "Public announcements regarding holiday schedules."}]
    )

    # Search for Apollo in doc_2 (should return 0 results)
    results = temp_vector_store.similarity_search("Apollo secret code", document_id="doc_2")
    for r in results:
        assert "998811" not in r["text"]
        assert "Apollo" not in r["text"]

def test_delete_document(temp_vector_store):
    temp_vector_store.chunk_and_store(
        document_id="doc_to_delete",
        filename="temporary.pdf",
        file_size=500,
        pages_data=[{"page_number": 1, "text": "Temporary content to be deleted."}]
    )

    assert temp_vector_store.get_document("doc_to_delete") is not None
    deleted = temp_vector_store.delete_document("doc_to_delete")
    assert deleted is True

    assert temp_vector_store.get_document("doc_to_delete") is None
    results = temp_vector_store.similarity_search("Temporary content", document_id="doc_to_delete")
    assert len(results) == 0

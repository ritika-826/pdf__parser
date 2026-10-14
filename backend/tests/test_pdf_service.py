import pytest
from app.services.pdf_service import (
    PDFService,
    PDFSizeExceededError,
    PDFInvalidFormatError,
    PDFEmptyError,
)
from tests.conftest import create_text_pdf_bytes, create_sample_pdf_bytes

def test_validate_file_invalid_extension():
    with pytest.raises(PDFInvalidFormatError, match="Only PDF"):
        PDFService.validate_file("document.docx", 100, b"%PDF-1.4")

def test_validate_file_size_exceeded():
    with pytest.raises(PDFSizeExceededError, match="exceeds maximum allowed limit"):
        # 25 MB exceeds the default 20 MB limit
        large_bytes = 25 * 1024 * 1024 + 1
        PDFService.validate_file("large.pdf", large_bytes, b"%PDF-1.4")

def test_validate_file_invalid_header():
    with pytest.raises(PDFInvalidFormatError, match="Invalid PDF header"):
        PDFService.validate_file("fake.pdf", 100, b"NOT_A_PDF_CONTENT")

def test_extract_text_success():
    pdf_bytes = create_text_pdf_bytes(["Artificial Intelligence in Document Retrieval and Question Answering."])
    pages = PDFService.extract_text_per_page(pdf_bytes, "test.pdf")
    
    assert len(pages) >= 1
    assert pages[0]["page_number"] == 1
    assert "Artificial Intelligence" in pages[0]["text"]

def test_extract_text_empty_or_scanned():
    # Blank PDF with no text stream
    blank_pdf = create_sample_pdf_bytes([""])
    with pytest.raises(PDFEmptyError, match="No extractable text found in this PDF.*OCR"):
        PDFService.extract_text_per_page(blank_pdf, "blank.pdf")

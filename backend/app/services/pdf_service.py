import io
from typing import List, Dict, Any
import pdfplumber
from app.config import settings

class PDFProcessingError(Exception):
    """Base exception for PDF processing errors."""
    pass

class PDFSizeExceededError(PDFProcessingError):
    """Raised when the uploaded PDF exceeds the configured size limit."""
    pass

class PDFInvalidFormatError(PDFProcessingError):
    """Raised when the file does not have a valid PDF header or extension."""
    pass

class PDFEncryptedError(PDFProcessingError):
    """Raised when the PDF is password-protected or encrypted."""
    pass

class PDFCorruptedError(PDFProcessingError):
    """Raised when the PDF file is corrupted or unreadable."""
    pass

class PDFEmptyError(PDFProcessingError):
    """Raised when the PDF has no pages or contains no extractable text (e.g. scanned)."""
    pass

class PDFService:
    @staticmethod
    def validate_file(filename: str, file_size: int, file_bytes: bytes) -> None:
        """Validate filename, size, and magic bytes."""
        if not filename.lower().endswith(".pdf"):
            raise PDFInvalidFormatError("Only PDF (.pdf) files are supported.")
        
        max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
        if file_size > max_bytes:
            raise PDFSizeExceededError(
                f"File size ({file_size / (1024 * 1024):.1f} MB) exceeds maximum allowed limit of {settings.MAX_FILE_SIZE_MB} MB."
            )
        
        if not file_bytes.startswith(b"%PDF"):
            raise PDFInvalidFormatError("Invalid PDF header. File is not a valid PDF document.")

    @staticmethod
    def extract_text_per_page(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """
        Extract text from PDF on a per-page basis using PDFPlumber.
        Preserves page numbers and detects scanned/empty documents.
        """
        PDFService.validate_file(filename, len(file_bytes), file_bytes)

        pages_data: List[Dict[str, Any]] = []
        total_text_chars = 0

        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                if len(pdf.pages) == 0:
                    raise PDFEmptyError("The uploaded PDF contains no pages.")

                for idx, page in enumerate(pdf.pages):
                    page_number = idx + 1
                    try:
                        extracted = page.extract_text()
                    except Exception as pe:
                        # Handle page-level read errors
                        extracted = None

                    text = (extracted or "").strip()
                    if text:
                        total_text_chars += len(text)
                        pages_data.append({
                            "page_number": page_number,
                            "text": text
                        })

        except pdfplumber.pdfminer.pdfparser.PDFSyntaxError as e:
            raise PDFCorruptedError(f"PDF syntax error: Document is corrupted or unreadable ({str(e)}).")
        except pdfplumber.pdfminer.pdfdocument.PDFPasswordIncorrect:
            raise PDFEncryptedError("PDF document is password protected.")
        except pdfplumber.pdfminer.pdfdocument.PDFEncryptionError:
            raise PDFEncryptedError("PDF document is encrypted.")
        except (PDFProcessingError, PDFSizeExceededError, PDFInvalidFormatError, PDFEmptyError):
            raise
        except Exception as e:
            err_str = str(e).lower()
            if "password" in err_str or "encrypt" in err_str:
                raise PDFEncryptedError("PDF document is encrypted or password-protected.")
            raise PDFCorruptedError(f"Failed to process PDF: {str(e)}")

        if total_text_chars == 0 or len(pages_data) == 0:
            raise PDFEmptyError(
                "No extractable text found in this PDF. The document may be scanned or image-based, which requires OCR to extract text."
            )

        return pages_data

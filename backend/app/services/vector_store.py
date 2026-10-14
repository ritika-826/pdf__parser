import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.config import settings
from app.services.embedding_service import embedding_service
from app.models.schemas import DocumentInfo, SourceReference

logger = logging.getLogger(__name__)

class VectorStoreService:
    def __init__(self, persist_directory: Optional[str] = None):
        self.persist_directory = Path(persist_directory or settings.CHROMA_PERSIST_DIRECTORY)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.persist_directory / "registry.json"
        
        # Initialize ChromaDB persistent client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="pdf_documents",
            metadata={"description": "PDF_Parser document chunk embeddings"}
        )
        self._init_registry()

    def _init_registry(self) -> None:
        if not self.registry_file.exists():
            self._save_registry({})

    def _load_registry(self) -> Dict[str, Any]:
        try:
            if self.registry_file.exists():
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading registry: {e}")
        return {}

    def _save_registry(self, data: Dict[str, Any]) -> None:
        try:
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving registry: {e}")

    def list_documents(self) -> List[DocumentInfo]:
        """Return all tracked documents."""
        reg = self._load_registry()
        docs: List[DocumentInfo] = []
        for doc_id, info in reg.items():
            docs.append(DocumentInfo(
                document_id=doc_id,
                filename=info.get("filename", "unknown.pdf"),
                file_size_bytes=info.get("file_size_bytes", 0),
                page_count=info.get("page_count", 0),
                chunk_count=info.get("chunk_count", 0),
                uploaded_at=info.get("uploaded_at", datetime.utcnow().isoformat()),
                status=info.get("status", "ready")
            ))
        return docs

    def get_document(self, document_id: str) -> Optional[DocumentInfo]:
        reg = self._load_registry()
        info = reg.get(document_id)
        if not info:
            return None
        return DocumentInfo(
            document_id=document_id,
            filename=info.get("filename", "unknown.pdf"),
            file_size_bytes=info.get("file_size_bytes", 0),
            page_count=info.get("page_count", 0),
            chunk_count=info.get("chunk_count", 0),
            uploaded_at=info.get("uploaded_at", datetime.utcnow().isoformat()),
            status=info.get("status", "ready")
        )

    def chunk_and_store(
        self,
        document_id: str,
        filename: str,
        file_size: int,
        pages_data: List[Dict[str, Any]],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> DocumentInfo:
        """
        Split extracted pages into chunks, compute embeddings, and store in ChromaDB.
        """
        c_size = chunk_size or settings.CHUNK_SIZE
        c_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=c_size,
            chunk_overlap=c_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        all_chunks: List[str] = []
        all_metadatas: List[Dict[str, Any]] = []
        all_ids: List[str] = []

        total_pages = len(pages_data)
        global_chunk_idx = 0

        for page in pages_data:
            page_number = page["page_number"]
            page_text = page["text"]
            
            # Split text for current page
            page_chunks = text_splitter.split_text(page_text)
            
            for chunk_content in page_chunks:
                clean_chunk = chunk_content.strip()
                if not clean_chunk:
                    continue
                
                chunk_id = f"{document_id}_p{page_number}_c{global_chunk_idx}"
                all_ids.append(chunk_id)
                all_chunks.append(clean_chunk)
                all_metadatas.append({
                    "document_id": document_id,
                    "filename": filename,
                    "page_number": int(page_number),
                    "chunk_id": chunk_id
                })
                global_chunk_idx += 1

        if not all_chunks:
            raise ValueError("No valid text chunks could be produced from the document.")

        # If document already existed in chroma, delete old chunks first
        try:
            self.collection.delete(where={"document_id": document_id})
        except Exception:
            pass

        # Generate embeddings using HuggingFace
        embeddings = embedding_service.embed_documents(all_chunks)

        # Batch insert into ChromaDB
        batch_size = 100
        try:
            for i in range(0, len(all_chunks), batch_size):
                end_idx = i + batch_size
                self.collection.add(
                    ids=all_ids[i:end_idx],
                    embeddings=embeddings[i:end_idx],
                    documents=all_chunks[i:end_idx],
                    metadatas=all_metadatas[i:end_idx]
                )
        except Exception as e:
            if "dimension" in str(e).lower():
                logger.warning(f"Dimension mismatch detected ({e}). Resetting collection...")
                self.client.delete_collection("pdf_documents")
                self.collection = self.client.get_or_create_collection(
                    name="pdf_documents",
                    metadata={"description": "PDF_Parser document chunk embeddings"}
                )
                for i in range(0, len(all_chunks), batch_size):
                    end_idx = i + batch_size
                    self.collection.add(
                        ids=all_ids[i:end_idx],
                        embeddings=embeddings[i:end_idx],
                        documents=all_chunks[i:end_idx],
                        metadatas=all_metadatas[i:end_idx]
                    )
            else:
                raise

        # Update registry
        reg = self._load_registry()
        doc_info = {
            "document_id": document_id,
            "filename": filename,
            "file_size_bytes": file_size,
            "page_count": total_pages,
            "chunk_count": len(all_chunks),
            "uploaded_at": datetime.utcnow().isoformat(),
            "status": "ready"
        }
        reg[document_id] = doc_info
        self._save_registry(reg)

        return DocumentInfo(**doc_info)

    def similarity_search(
        self,
        query: str,
        document_id: str,
        top_k: Optional[int] = None
    ) -> List[SourceReference]:
        """
        Perform similarity search scoped strictly to document_id.
        """
        k = top_k or settings.TOP_K
        query_vec = embedding_service.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=k,
            where={"document_id": document_id}
        )

        sources: List[SourceReference] = []
        if not results or not results.get("documents") or not results["documents"][0]:
            return sources

        docs = results["documents"][0]
        metadatas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        for i, text in enumerate(docs):
            meta = metadatas[i] if i < len(metadatas) else {}
            dist = distances[i] if i < len(distances) else None
            # Convert cosine distance or L2 distance to an informative score if applicable
            score = round(1.0 - (dist / 2.0), 3) if dist is not None else None
            sources.append(SourceReference(
                page=meta.get("page_number", 1),
                chunk_id=meta.get("chunk_id", f"chunk_{i}"),
                text=text,
                score=score
            ))

        return sources

    def delete_document(self, document_id: str) -> bool:
        """Delete all vectors for document_id and remove from registry."""
        reg = self._load_registry()
        existed = document_id in reg

        try:
            self.collection.delete(where={"document_id": document_id})
        except Exception as e:
            logger.warning(f"Chroma delete for {document_id} error: {e}")

        if existed:
            del reg[document_id]
            self._save_registry(reg)
            return True
        return False

# Global singleton
vector_store_service = VectorStoreService()

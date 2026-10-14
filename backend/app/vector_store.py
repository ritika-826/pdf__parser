import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# HuggingFace Embedding Service
class HuggingFaceSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        if not text:
            return []
        model = self._get_model()
        embedding = model.encode([text], convert_to_numpy=True, show_progress_bar=False)
        return embedding[0].tolist()

embedding_service = HuggingFaceSentenceTransformerEmbeddings()

# ChromaDB Vector Store Service
class VectorStoreService:
    def __init__(self, persist_directory: Optional[str] = None):
        self.persist_directory = Path(persist_directory or os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db"))
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.persist_directory / "registry.json"
        
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

    def list_documents(self) -> List[Dict[str, Any]]:
        reg = self._load_registry()
        return list(reg.values())

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        reg = self._load_registry()
        return reg.get(document_id)

    def chunk_and_store(
        self,
        document_id: str,
        filename: str,
        file_size: int,
        pages_data: List[Dict[str, Any]],
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ) -> Dict[str, Any]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        all_chunks: List[str] = []
        all_metadatas: List[Dict[str, Any]] = []
        all_ids: List[str] = []

        global_chunk_idx = 0
        for page in pages_data:
            page_number = page["page_number"]
            page_chunks = text_splitter.split_text(page["text"])
            
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

        try:
            self.collection.delete(where={"document_id": document_id})
        except Exception:
            pass

        embeddings = embedding_service.embed_documents(all_chunks)

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
                logger.warning(f"Dimension mismatch ({e}), recreating collection...")
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

        doc_info = {
            "document_id": document_id,
            "filename": filename,
            "file_size_bytes": file_size,
            "page_count": len(pages_data),
            "chunk_count": len(all_chunks),
            "uploaded_at": datetime.utcnow().isoformat(),
            "status": "ready"
        }
        reg = self._load_registry()
        reg[document_id] = doc_info
        self._save_registry(reg)

        return doc_info

    def similarity_search(
        self,
        query: str,
        document_id: str,
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        query_vec = embedding_service.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            where={"document_id": document_id}
        )

        sources: List[Dict[str, Any]] = []
        if not results or not results.get("documents") or not results["documents"][0]:
            return sources

        docs = results["documents"][0]
        metadatas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        for i, text in enumerate(docs):
            meta = metadatas[i] if i < len(metadatas) else {}
            dist = distances[i] if i < len(distances) else None
            score = round(1.0 - (dist / 2.0), 3) if dist is not None else None
            sources.append({
                "page": meta.get("page_number", 1),
                "chunk_id": meta.get("chunk_id", f"chunk_{i}"),
                "text": text,
                "score": score
            })

        return sources

    def delete_document(self, document_id: str) -> bool:
        reg = self._load_registry()
        existed = document_id in reg
        try:
            self.collection.delete(where={"document_id": document_id})
        except Exception as e:
            logger.warning(f"Chroma delete for {document_id}: {e}")

        if existed:
            del reg[document_id]
            self._save_registry(reg)
            return True
        return False

vector_store_service = VectorStoreService()

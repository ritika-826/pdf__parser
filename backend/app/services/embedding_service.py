from typing import List, Optional
import logging
from langchain_core.embeddings import Embeddings
from app.config import settings

logger = logging.getLogger(__name__)

class HuggingFaceSentenceTransformerEmbeddings(Embeddings):
    """
    LangChain Embeddings implementation using Hugging Face Sentence Transformers.
    Lazy loads model and provides thread-safe document/query vectorization.
    """
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.error(f"Failed to load embedding model {self.model_name}: {e}")
                raise RuntimeError(
                    f"Could not initialize embedding model '{self.model_name}': {str(e)}"
                ) from e
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        model = self._get_model()
        try:
            embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating document embeddings: {e}")
            raise RuntimeError(f"Embedding generation failed: {str(e)}") from e

    def embed_query(self, text: str) -> List[float]:
        if not text:
            return []
        model = self._get_model()
        try:
            embedding = model.encode([text], convert_to_numpy=True, show_progress_bar=False)
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise RuntimeError(f"Query embedding generation failed: {str(e)}") from e

# Global singleton instance
embedding_service = HuggingFaceSentenceTransformerEmbeddings()

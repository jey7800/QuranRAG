"""BGE-M3 embedder via sentence-transformers.

BAAI/bge-m3 is a multilingual model (1024-dim) that handles Arabic + English
well. It is the primary/default embedding model for QuranRAG.
"""

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.embedding.base import Embedder


class BGEM3Embedder(Embedder):
    """BAAI/bge-m3 — multilingual, 1024-dim, handles Arabic+English well.

    Uses sentence-transformers for local inference (no API calls).
    First run downloads ~2 GB model to ~/.cache/huggingface/.
    """

    MODEL_ID = "BAAI/bge-m3"
    DIMENSION = 1024
    MAX_SEQ_LENGTH = 8192
    DEFAULT_BATCH_SIZE = 32

    def __init__(self, batch_size: int = DEFAULT_BATCH_SIZE, device: str | None = None):
        """Initialize the model.

        Args:
            batch_size: Batch size for encoding. 32 is safe for 16GB RAM.
            device: "cpu", "cuda", or None (auto-detect).
        """
        logger.info(f"Loading {self.MODEL_ID}...")
        self._model = SentenceTransformer(self.MODEL_ID, device=device)
        self._batch_size = batch_size
        logger.info(f"  Loaded on device: {self._model.device}")

    @property
    def name(self) -> str:
        return "bge-m3"

    @property
    def dimension(self) -> int:
        return self.DIMENSION

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed documents. No special prefix for BGE-M3 documents."""
        return self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,
        )

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query. sentence-transformers handles the query prefix internally."""
        return self._model.encode(
            [query],
            batch_size=1,
            normalize_embeddings=True,
        )[0]

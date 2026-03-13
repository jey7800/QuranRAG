"""Swan-Large Arabic-optimized embedder.

OALL/swan-large-embedding is optimized for Arabic text (1024-dim).
Stronger on pure Arabic queries than BGE-M3 but less multilingual.
"""

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.embedding.base import Embedder


class SwanLargeEmbedder(Embedder):
    """OALL/swan-large-embedding — Arabic-optimized, 1024-dim.

    Good for Arabic-only embedding or as a comparison baseline.
    """

    MODEL_ID = "OALL/swan-large-embedding"
    DIMENSION = 1024
    DEFAULT_BATCH_SIZE = 16  # Larger model, smaller batch

    def __init__(self, batch_size: int = DEFAULT_BATCH_SIZE, device: str | None = None):
        logger.info(f"Loading {self.MODEL_ID}...")
        self._model = SentenceTransformer(self.MODEL_ID, device=device)
        self._batch_size = batch_size
        logger.info(f"  Loaded on device: {self._model.device}")

    @property
    def name(self) -> str:
        return "swan-large"

    @property
    def dimension(self) -> int:
        return self.DIMENSION

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,
        )

    def embed_query(self, query: str) -> np.ndarray:
        return self._model.encode(
            [query],
            batch_size=1,
            normalize_embeddings=True,
        )[0]

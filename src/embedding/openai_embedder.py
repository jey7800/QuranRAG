"""OpenAI text-embedding-3-large embedder.

API-based, 3072-dim. Requires OPENAI_API_KEY in environment.
Install with: pip install 'quran-rag[openai]'
"""

import os

import numpy as np
from loguru import logger

from src.embedding.base import Embedder


class OpenAIEmbedder(Embedder):
    """OpenAI text-embedding-3-large — 3072-dim, API-based."""

    MODEL_ID = "text-embedding-3-large"
    DIMENSION = 3072
    MAX_BATCH_SIZE = 2048  # OpenAI limit

    def __init__(self, api_key: str | None = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not found. Install with: pip install 'quran-rag[openai]'"
            )

        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        logger.info(f"OpenAI embedder initialized: {self.MODEL_ID}")

    @property
    def name(self) -> str:
        return "openai-3-large"

    @property
    def dimension(self) -> int:
        return self.DIMENSION

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed documents via OpenAI API. Handles batching internally."""
        all_embeddings = []
        for i in range(0, len(texts), self.MAX_BATCH_SIZE):
            batch = texts[i : i + self.MAX_BATCH_SIZE]
            response = self._client.embeddings.create(model=self.MODEL_ID, input=batch)
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        return np.array(all_embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """OpenAI does not distinguish query vs document."""
        return self.embed_single(query)

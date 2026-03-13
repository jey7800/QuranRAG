"""Gemini Embedding via google-genai.

API-based, 768-dim. Requires GOOGLE_API_KEY in environment.
Install with: pip install 'quran-rag[google]'
"""

import os

import numpy as np
from loguru import logger

from src.embedding.base import Embedder


class GeminiEmbedder(Embedder):
    """Gemini text-embedding-004 — 768-dim, API-based."""

    MODEL_ID = "text-embedding-004"
    DIMENSION = 768
    MAX_BATCH_SIZE = 100

    def __init__(self, api_key: str | None = None):
        try:
            from google import genai  # noqa: F401
        except ImportError:
            raise ImportError(
                "google-genai package not found. Install with: pip install 'quran-rag[google]'"
            )

        self._client = genai.Client(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))
        logger.info(f"Gemini embedder initialized: {self.MODEL_ID}")

    @property
    def name(self) -> str:
        return "gemini-004"

    @property
    def dimension(self) -> int:
        return self.DIMENSION

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        from google import genai

        all_embeddings = []
        for i in range(0, len(texts), self.MAX_BATCH_SIZE):
            batch = texts[i : i + self.MAX_BATCH_SIZE]
            result = self._client.models.embed_content(
                model=self.MODEL_ID,
                contents=batch,
                config=genai.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            all_embeddings.extend([e.values for e in result.embeddings])
        return np.array(all_embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        from google import genai

        result = self._client.models.embed_content(
            model=self.MODEL_ID,
            contents=[query],
            config=genai.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return np.array(result.embeddings[0].values, dtype=np.float32)

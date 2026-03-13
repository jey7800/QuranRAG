"""Abstract embedding interface."""

from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    """Abstract base class for text embedding models.

    All embedding implementations must subclass this and implement:
    - name: human-readable model identifier
    - dimension: output vector dimensionality
    - embed_texts: batch document embedding
    - embed_query: single query embedding (may differ from document embedding)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name, e.g. 'bge-m3'."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the output vectors."""
        ...

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts (documents).

        Args:
            texts: List of strings to embed.

        Returns:
            numpy array of shape (len(texts), self.dimension).
        """
        ...

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text. Convenience wrapper.

        Returns:
            numpy array of shape (self.dimension,).
        """
        return self.embed_texts([text])[0]

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a search query.

        Some models (e.g. BGE-M3) use a different prompt/prefix for queries
        vs. documents. This method handles that distinction.

        Args:
            query: The search query string.

        Returns:
            numpy array of shape (self.dimension,).
        """
        ...

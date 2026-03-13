"""Abstract vector store interface."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from pydantic import BaseModel


class SearchResult(BaseModel):
    """A single search result from the vector store."""

    chunk_id: str
    score: float
    payload: dict[str, Any]  # Full metadata stored with the vector


class VectorStore(ABC):
    """Abstract interface for vector storage and retrieval."""

    @abstractmethod
    def create_collection(
        self,
        name: str,
        dimension: int,
        distance: str = "cosine",
        recreate: bool = False,
    ) -> None:
        """Create a named collection (or ensure it exists).

        Args:
            name: Collection name, e.g. "verse_chunks".
            dimension: Vector dimensionality.
            distance: Distance metric ("cosine", "dot", "euclidean").
            recreate: If True, drop and recreate if exists.
        """
        ...

    @abstractmethod
    def upsert(
        self,
        collection_name: str,
        ids: list[str],
        vectors: np.ndarray,
        payloads: list[dict[str, Any]],
    ) -> int:
        """Insert or update vectors with metadata.

        Args:
            collection_name: Target collection.
            ids: Unique IDs for each vector (chunk_id).
            vectors: (N, D) array of vectors.
            payloads: List of metadata dicts, one per vector.

        Returns:
            Number of vectors upserted.
        """
        ...

    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for nearest neighbors.

        Args:
            collection_name: Collection to search.
            query_vector: Query vector of shape (D,).
            top_k: Number of results to return.
            filters: Optional metadata filters (e.g. {"surah_number": 2}).

        Returns:
            List of SearchResult ordered by descending similarity.
        """
        ...

    @abstractmethod
    def count(self, collection_name: str) -> int:
        """Return the number of vectors in a collection."""
        ...

    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection entirely."""
        ...

    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        ...

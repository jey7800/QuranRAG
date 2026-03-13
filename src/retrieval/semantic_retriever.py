"""Semantic retriever — vector similarity search across Qdrant collections."""

from typing import Any

from loguru import logger

from src.config import (
    COLLECTION_SURAH_CHUNKS,
    COLLECTION_THEMATIC_CHUNKS,
    COLLECTION_VERSE_CHUNKS,
    DEFAULT_TOP_K,
    SEMANTIC_RETRIEVAL_K,
)
from src.embedding.base import Embedder
from src.vectorstore.base import SearchResult, VectorStore

ALL_COLLECTIONS = [
    COLLECTION_VERSE_CHUNKS,
    COLLECTION_THEMATIC_CHUNKS,
    COLLECTION_SURAH_CHUNKS,
]


class SemanticRetriever:
    """Search Qdrant collections by embedding similarity."""

    def __init__(self, vector_store: VectorStore, embedder: Embedder) -> None:
        self._store = vector_store
        self._embedder = embedder

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        collections: list[str] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Embed query and search across collections, returning merged results by score.

        Args:
            query: User query string (any language).
            top_k: Number of final results to return.
            collections: Which collections to search (defaults to all 3).
            filters: Metadata filters passed to Qdrant (e.g. {"surah_number": 2}).

        Returns:
            Merged and sorted list of SearchResult.
        """
        collections = collections or ALL_COLLECTIONS
        query_vector = self._embedder.embed_query(query)

        all_results: list[SearchResult] = []
        for collection in collections:
            if not self._store.collection_exists(collection):
                logger.warning(f"Collection {collection} does not exist, skipping")
                continue
            results = self._store.search(
                collection_name=collection,
                query_vector=query_vector,
                top_k=SEMANTIC_RETRIEVAL_K,
                filters=filters,
            )
            all_results.extend(results)

        # Sort by score descending and take top_k
        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results[:top_k]

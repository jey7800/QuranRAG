"""Hybrid retriever — orchestrates semantic + graph retrieval."""

from typing import Any

from loguru import logger

from src.api.schemas import EnrichedVerse
from src.config import COLLECTION_VERSE_CHUNKS, DEFAULT_TOP_K
from src.retrieval.context_enricher import ContextEnricher
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.semantic_retriever import SemanticRetriever
from src.vectorstore.base import SearchResult

# Score boost for verses confirmed by graph expansion
_GRAPH_BOOST = 0.05


class HybridRetriever:
    """Combines semantic vector search with ontology graph expansion.

    Strategy: search verse_chunks only (most precise), then use the graph
    to boost verses that are ontologically related to the query topic.
    """

    def __init__(
        self,
        semantic: SemanticRetriever,
        graph: GraphRetriever,
        enricher: ContextEnricher,
    ) -> None:
        self._semantic = semantic
        self._graph = graph
        self._enricher = enricher

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        filters: dict[str, Any] | None = None,
    ) -> list[EnrichedVerse]:
        """Run hybrid retrieval: semantic search + graph boost + enrichment.

        Returns enriched verses sorted by score, deduplicated.
        """
        # 1. Semantic search on verse_chunks only (avoids thematic/surah dilution)
        semantic_results = self._semantic.search(
            query,
            top_k=top_k * 2,
            collections=[COLLECTION_VERSE_CHUNKS],
            filters=filters,
        )

        # 2. Extract verse IDs and scores
        verse_scores: dict[str, float] = {}
        for result in semantic_results:
            vid = self._extract_verse_id(result)
            if vid and (vid not in verse_scores or result.score > verse_scores[vid]):
                verse_scores[vid] = result.score

        # 3. Graph expansion — boost existing results, don't inject new ones at 0
        all_tags: set[str] = set()
        for result in semantic_results[:10]:
            tags = result.payload.get("topic_tags", [])
            if isinstance(tags, list):
                all_tags.update(tags)

        if all_tags:
            graph_verse_ids = set(self._graph.expand(list(all_tags)))
            for vid in verse_scores:
                if vid in graph_verse_ids:
                    verse_scores[vid] += _GRAPH_BOOST

        # 4. Sort by score and take top_k
        sorted_ids = sorted(verse_scores, key=verse_scores.get, reverse=True)
        top_ids = sorted_ids[:top_k]

        # 5. Enrich
        enriched = self._enricher.enrich_verses(
            top_ids, scores=verse_scores, neighbor_range=2
        )

        logger.debug(
            f"Hybrid retrieval: {len(semantic_results)} semantic, "
            f"{len(all_tags)} tags, {len(verse_scores)} unique -> {len(enriched)} enriched"
        )
        return enriched

    @staticmethod
    def _extract_verse_id(result: SearchResult) -> str | None:
        """Extract verse ID from a SearchResult."""
        chunk_id = result.chunk_id
        if chunk_id.startswith("verse:"):
            return chunk_id[len("verse:"):]
        return result.payload.get("verse_id") or None

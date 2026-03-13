"""Hybrid retriever — orchestrates semantic + graph retrieval."""

from typing import Any

from loguru import logger

from src.api.schemas import EnrichedVerse
from src.config import DEFAULT_TOP_K
from src.retrieval.context_enricher import ContextEnricher
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.semantic_retriever import SemanticRetriever
from src.vectorstore.base import SearchResult


class HybridRetriever:
    """Combines semantic vector search with ontology graph expansion."""

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
        """Run hybrid retrieval: semantic search + graph expansion + enrichment.

        Returns enriched verses sorted by score, deduplicated.
        """
        # 1. Semantic search (fetch more than top_k to allow for dedup)
        semantic_results = self._semantic.search(
            query, top_k=top_k * 2, filters=filters
        )

        # 2. Extract verse IDs and scores from semantic results
        verse_scores: dict[str, float] = {}
        for result in semantic_results:
            verse_ids = self._extract_verse_ids(result)
            for vid in verse_ids:
                if vid not in verse_scores or result.score > verse_scores[vid]:
                    verse_scores[vid] = result.score

        # 3. Graph expansion using topic tags from semantic results
        all_tags: set[str] = set()
        for result in semantic_results[:10]:  # use top-10 for tag extraction
            tags = result.payload.get("topic_tags", [])
            if isinstance(tags, list):
                all_tags.update(tags)

        if all_tags:
            graph_verse_ids = self._graph.expand(list(all_tags))
            for vid in graph_verse_ids:
                if vid not in verse_scores:
                    verse_scores[vid] = 0.0  # graph-only results get score 0

        # 4. Sort by score and take top_k
        sorted_ids = sorted(verse_scores.keys(), key=lambda v: verse_scores[v], reverse=True)
        top_ids = sorted_ids[:top_k]

        # 5. Enrich
        enriched = self._enricher.enrich_verses(
            top_ids, scores=verse_scores, neighbor_range=2
        )

        logger.debug(
            f"Hybrid retrieval: {len(semantic_results)} semantic, "
            f"{len(all_tags)} tags, {len(verse_scores)} unique verses -> {len(enriched)} enriched"
        )
        return enriched

    @staticmethod
    def _extract_verse_ids(result: SearchResult) -> list[str]:
        """Extract individual verse IDs from a SearchResult based on chunk type."""
        chunk_id = result.chunk_id

        if chunk_id.startswith("verse:"):
            # "verse:2:255" -> "2:255"
            return [chunk_id[len("verse:"):]]

        if chunk_id.startswith("group:"):
            # Thematic group — get verse_ids from payload
            verse_ids = result.payload.get("verse_ids", [])
            if verse_ids:
                return verse_ids
            # Fallback: parse "group:2:1-5" -> surah 2, ayahs 1-5
            rest = chunk_id[len("group:"):]
            parts = rest.split(":")
            if len(parts) == 2:
                surah = parts[0]
                ayah_range = parts[1].split("-")
                if len(ayah_range) == 2:
                    start, end = int(ayah_range[0]), int(ayah_range[1])
                    return [f"{surah}:{a}" for a in range(start, end + 1)]
            return []

        if chunk_id.startswith("surah:"):
            # Surah summary — get verse_ids from payload (but limit to avoid noise)
            verse_ids = result.payload.get("verse_ids", [])
            return verse_ids[:5] if verse_ids else []

        return []

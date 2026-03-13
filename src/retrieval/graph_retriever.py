"""Graph retriever — ontology-based verse expansion via NetworkX."""

from pathlib import Path

import networkx as nx
from loguru import logger

from src.config import CONCEPTS_JSON, GRAPH_HOPS, GRAPH_TOP_CONCEPTS
from src.data.schemas import OntologyConcept
from src.retrieval.data_store import DataStore


class GraphRetriever:
    """Expand retrieval results using the Quranic ontology concept graph."""

    def __init__(self, data_store: DataStore) -> None:
        self._data_store = data_store
        self._graph: nx.Graph = nx.Graph()
        self._build_graph()

    def _build_graph(self) -> None:
        """Build NetworkX graph from concepts in the DataStore."""
        concepts = self._data_store.get_all_concepts()
        for cid, concept in concepts.items():
            self._graph.add_node(cid, verses=concept.verses, name=concept.name_en)
            for child in concept.child_concepts:
                self._graph.add_edge(cid, child, relation="parent_child")
            for related in concept.related_concepts:
                self._graph.add_edge(cid, related, relation="related")
        logger.info(
            f"Concept graph built: {self._graph.number_of_nodes()} nodes, "
            f"{self._graph.number_of_edges()} edges"
        )

    def expand(
        self,
        topic_tags: list[str],
        hops: int = GRAPH_HOPS,
        max_concepts: int = GRAPH_TOP_CONCEPTS,
    ) -> list[str]:
        """Given topic tags, find related verse IDs via graph traversal.

        1. Match topic_tags to concept nodes
        2. Keep top max_concepts by number of linked verses
        3. BFS expansion by `hops` hops
        4. Collect all verse IDs from reached concepts

        Returns:
            Deduplicated list of verse_ids.
        """
        # Match tags to graph nodes
        matched = [tag for tag in topic_tags if tag in self._graph]
        if not matched:
            return []

        # Rank by number of linked verses, take top N
        matched.sort(
            key=lambda c: len(self._graph.nodes[c].get("verses", [])), reverse=True
        )
        seed_concepts = matched[:max_concepts]

        # BFS expansion
        reached: set[str] = set()
        for seed in seed_concepts:
            neighbors = nx.single_source_shortest_path_length(
                self._graph, seed, cutoff=hops
            )
            reached.update(neighbors.keys())

        # Collect verse IDs from all reached concepts
        verse_ids: set[str] = set()
        for concept_id in reached:
            node_data = self._graph.nodes.get(concept_id, {})
            for vid in node_data.get("verses", []):
                verse_ids.add(vid)

        return list(verse_ids)

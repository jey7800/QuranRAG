"""MCP server exposing QuranRAG retrieval as tools for LLM clients."""

import threading

from loguru import logger
from mcp.server.fastmcp import FastMCP

# Module-level state, initialized lazily on first tool call
_data_store = None
_hybrid = None
_enricher = None
_graph = None
_init_lock = threading.Lock()
_initialized = False

mcp_server = FastMCP(
    "quran-rag",
    instructions="Search and explore the Quran with enriched context, citations, and translations",
)


def init_mcp_state(data_store, hybrid, enricher, graph) -> None:
    """Set module-level state (called by FastAPI lifespan or lazy init)."""
    global _data_store, _hybrid, _enricher, _graph, _initialized
    _data_store = data_store
    _hybrid = hybrid
    _enricher = enricher
    _graph = graph
    _initialized = True


def _lazy_init() -> None:
    """Load all components on first tool call (keeps MCP startup instant)."""
    global _initialized
    if _initialized:
        return

    with _init_lock:
        if _initialized:
            return

        logger.info("Lazy-loading QuranRAG components...")

        from src.config import QDRANT_DB_PATH, QDRANT_MODE
        from src.embedding.factory import get_embedder
        from src.retrieval.context_enricher import ContextEnricher
        from src.retrieval.data_store import DataStore
        from src.retrieval.graph_retriever import GraphRetriever
        from src.retrieval.hybrid_retriever import HybridRetriever
        from src.retrieval.semantic_retriever import SemanticRetriever
        from src.vectorstore.qdrant_store import QdrantVectorStore

        data_store = DataStore()
        data_store.load()

        vector_store = QdrantVectorStore(mode=QDRANT_MODE, path=QDRANT_DB_PATH)
        embedder = get_embedder()

        semantic = SemanticRetriever(vector_store, embedder)
        graph = GraphRetriever(data_store)
        enricher = ContextEnricher(data_store)
        hybrid = HybridRetriever(semantic, graph, enricher)

        init_mcp_state(data_store, hybrid, enricher, graph)
        logger.info("QuranRAG components loaded successfully")


def _ensure_state() -> None:
    _lazy_init()


@mcp_server.tool()
def search_verses(query: str, top_k: int = 10) -> str:
    """Search the Quran for verses relevant to a question or topic.

    Returns enriched verses with Arabic text, translations, historical context,
    polysemy alerts, and thematic links. Use this for any Quran-related question.

    Args:
        query: The search query in any language (English, Arabic, French).
        top_k: Number of results to return (default 10).
    """
    _ensure_state()
    from src.mcp.formatters import format_search_results

    results = _hybrid.retrieve(query, top_k=top_k)
    return format_search_results(results, query)


@mcp_server.tool()
def get_verse(verse_ref: str) -> str:
    """Get complete details for a specific Quran verse by reference.

    Returns Arabic text, translations, morphology roots, polysemy alerts,
    historical context (asbab al-nuzul), and thematic links.

    Args:
        verse_ref: Verse reference in 'surah:ayah' format, e.g. '2:255' for Ayat al-Kursi.
    """
    _ensure_state()
    from src.mcp.formatters import format_verse

    enriched = _enricher.enrich_verse(verse_ref)
    if not enriched:
        return f"Verse {verse_ref} not found. Use format 'surah:ayah', e.g. '2:255'."
    return format_verse(enriched)


@mcp_server.tool()
def explore_theme(theme: str, top_k: int = 20) -> str:
    """Explore all Quran verses related to a theme or concept.

    Returns verses ordered by relevance, showing how the Quran addresses the topic.
    Examples: 'justice', 'creation', 'day-of-resurrection', 'noah', 'charity'.

    Args:
        theme: Theme or concept ID to explore (e.g. 'justice', 'prayer', 'moses').
        top_k: Maximum number of verses to return.
    """
    _ensure_state()
    from src.api.schemas import ThemeResponse
    from src.mcp.formatters import format_search_results, format_theme

    concept = _data_store.get_concept(theme)

    if concept:
        expanded_ids = _graph.expand([theme], hops=1)
        all_ids = list(dict.fromkeys(concept.verses + expanded_ids))[:top_k]
        verses = _enricher.enrich_verses(all_ids, neighbor_range=0)
        response = ThemeResponse(
            concept_id=concept.concept_id,
            name_en=concept.name_en,
            description=concept.description,
            related_concepts=concept.related_concepts,
            verses=verses,
        )
        return format_theme(response)

    results = _hybrid.retrieve(theme, top_k=top_k)
    return format_search_results(results, f"theme: {theme}")


@mcp_server.tool()
def compare_translations(verse_ref: str) -> str:
    """Compare available translations for a Quran verse.

    Shows Arabic text alongside English (Asad) and French (Hamidullah) translations.
    Highlights polysemous Arabic words that may explain translation differences.

    Args:
        verse_ref: Verse reference in 'surah:ayah' format, e.g. '2:255'.
    """
    _ensure_state()
    from src.api.schemas import CompareResponse
    from src.mcp.formatters import format_comparison

    verse = _data_store.get_verse(verse_ref)
    if not verse:
        return f"Verse {verse_ref} not found. Use format 'surah:ayah', e.g. '2:255'."

    translations = {"en_asad": verse.text_en_asad}
    if verse.text_fr_hamidullah:
        translations["fr_hamidullah"] = verse.text_fr_hamidullah
    if verse.transliteration:
        translations["transliteration"] = verse.transliteration

    comp = CompareResponse(
        verse_id=verse.verse_id,
        text_arabic=verse.text_arabic,
        translations=translations,
    )

    result = format_comparison(comp)
    polysemy = _data_store.get_polysemy_for_verse(verse)
    if polysemy:
        result += "\n\n**Polysemy alerts:**"
        for p in polysemy:
            senses = "; ".join(s.get("meaning_en", "") for s in p.senses)
            result += f"\n  - {p.word_arabic} ({p.root}): {senses}"
            if p.scholarly_note:
                result += f"\n    Note: {p.scholarly_note}"

    return result


@mcp_server.tool()
def get_context(verse_ref: str, range: int = 5) -> str:
    """Get surrounding context for a Quran verse.

    Shows neighboring verses, revelation circumstances, period, and related verses.
    Useful for understanding a verse in its textual and historical context.

    Args:
        verse_ref: Verse reference in 'surah:ayah' format, e.g. '2:255'.
        range: Number of verses before and after to include (default 5).
    """
    _ensure_state()
    from src.api.schemas import ContextResponse, VerseSnippet
    from src.mcp.formatters import format_context

    enriched = _enricher.enrich_verse(verse_ref, neighbor_range=0)
    if not enriched:
        return f"Verse {verse_ref} not found. Use format 'surah:ayah', e.g. '2:255'."

    neighbors = _data_store.get_neighbors(verse_ref, range_=range)
    neighbor_snippets = [
        VerseSnippet(
            verse_id=n.verse_id,
            text_arabic=n.text_arabic,
            text_en=n.text_en_asad,
        )
        for n in neighbors
    ]

    ctx = ContextResponse(
        center_verse=enriched,
        range=range,
        neighbors=neighbor_snippets,
    )
    return format_context(ctx)

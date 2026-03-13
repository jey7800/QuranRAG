"""REST API route handlers."""

from fastapi import APIRouter, HTTPException, Query, Request

from src.api.schemas import (
    CompareResponse,
    ContextResponse,
    EnrichedVerse,
    SearchResponse,
    StatsResponse,
    SurahResponse,
    ThemeResponse,
    VerseSnippet,
)
from src.config import (
    COLLECTION_SURAH_CHUNKS,
    COLLECTION_THEMATIC_CHUNKS,
    COLLECTION_VERSE_CHUNKS,
)

router = APIRouter()


@router.get("/search", response_model=SearchResponse)
async def search(
    request: Request,
    q: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(10, ge=1, le=100),
    surah: int | None = Query(None, ge=1, le=114),
    period: str | None = Query(None, pattern="^(meccan|medinan)$"),
):
    """Semantic search across the Quran with optional filters."""
    filters = {}
    if surah is not None:
        filters["surah_number"] = surah
    if period is not None:
        filters["revelation_period"] = period

    hybrid = request.app.state.hybrid_retriever
    results = hybrid.retrieve(q, top_k=top_k, filters=filters or None)
    return SearchResponse(query=q, total=len(results), results=results)


@router.get("/verse/{surah}/{ayah}", response_model=EnrichedVerse)
async def get_verse(request: Request, surah: int, ayah: int):
    """Get a single verse with full enrichment."""
    verse_id = f"{surah}:{ayah}"
    enricher = request.app.state.enricher
    enriched = enricher.enrich_verse(verse_id)
    if not enriched:
        raise HTTPException(status_code=404, detail=f"Verse {verse_id} not found")
    return enriched


@router.get("/surah/{surah_number}", response_model=SurahResponse)
async def get_surah(request: Request, surah_number: int):
    """Get surah metadata and all its verses."""
    data_store = request.app.state.data_store
    chapter = data_store.get_chapter(surah_number)
    if not chapter:
        raise HTTPException(status_code=404, detail=f"Surah {surah_number} not found")

    enricher = request.app.state.enricher
    verses = enricher.enrich_verses(chapter.verse_ids, neighbor_range=0)

    return SurahResponse(
        surah_number=chapter.surah_number,
        name_en=chapter.name_en,
        name_ar=chapter.name_ar,
        revelation_type=chapter.revelation_type,
        revelation_order=chapter.revelation_order,
        number_of_ayahs=chapter.number_of_ayahs,
        verses=verses,
    )


@router.get("/theme/{concept_id}", response_model=ThemeResponse)
async def get_theme(request: Request, concept_id: str):
    """Get all verses linked to an ontology concept."""
    data_store = request.app.state.data_store
    concept = data_store.get_concept(concept_id)
    if not concept:
        raise HTTPException(status_code=404, detail=f"Concept '{concept_id}' not found")

    # Get verses directly linked + 1-hop expansion
    graph = request.app.state.graph_retriever
    expanded_ids = graph.expand([concept_id], hops=1)
    # Also include directly linked verses
    all_ids = list(dict.fromkeys(concept.verses + expanded_ids))

    enricher = request.app.state.enricher
    verses = enricher.enrich_verses(all_ids, neighbor_range=0)

    return ThemeResponse(
        concept_id=concept.concept_id,
        name_en=concept.name_en,
        description=concept.description,
        related_concepts=concept.related_concepts,
        verses=verses,
    )


@router.get("/compare/{surah}/{ayah}", response_model=CompareResponse)
async def compare_translations(request: Request, surah: int, ayah: int):
    """Compare available translations for a verse."""
    verse_id = f"{surah}:{ayah}"
    data_store = request.app.state.data_store
    verse = data_store.get_verse(verse_id)
    if not verse:
        raise HTTPException(status_code=404, detail=f"Verse {verse_id} not found")

    translations = {"en_asad": verse.text_en_asad}
    if verse.text_fr_hamidullah:
        translations["fr_hamidullah"] = verse.text_fr_hamidullah
    if verse.transliteration:
        translations["transliteration"] = verse.transliteration

    return CompareResponse(
        verse_id=verse.verse_id,
        text_arabic=verse.text_arabic,
        translations=translations,
    )


@router.get("/context/{surah}/{ayah}", response_model=ContextResponse)
async def get_context(
    request: Request,
    surah: int,
    ayah: int,
    range: int = Query(5, ge=1, le=20, alias="range"),
):
    """Get surrounding verses for context."""
    verse_id = f"{surah}:{ayah}"
    enricher = request.app.state.enricher
    center = enricher.enrich_verse(verse_id, neighbor_range=0)
    if not center:
        raise HTTPException(status_code=404, detail=f"Verse {verse_id} not found")

    data_store = request.app.state.data_store
    neighbors = data_store.get_neighbors(verse_id, range_=range)
    neighbor_snippets = [
        VerseSnippet(
            verse_id=n.verse_id,
            text_arabic=n.text_arabic,
            text_en=n.text_en_asad,
        )
        for n in neighbors
    ]

    return ContextResponse(
        center_verse=center,
        range=range,
        neighbors=neighbor_snippets,
    )


@router.get("/stats", response_model=StatsResponse)
async def get_stats(request: Request):
    """Get dataset statistics."""
    data_store = request.app.state.data_store
    vector_store = request.app.state.vector_store
    stats = data_store.get_stats()

    chunks = {}
    for collection in [
        COLLECTION_VERSE_CHUNKS,
        COLLECTION_THEMATIC_CHUNKS,
        COLLECTION_SURAH_CHUNKS,
    ]:
        if vector_store.collection_exists(collection):
            chunks[collection] = vector_store.count(collection)
        else:
            chunks[collection] = 0

    return StatsResponse(
        total_chunks=chunks,
        **stats,
    )

"""Context enricher — attaches full verse data, polysemy, abrogation, neighbors."""

from src.api.schemas import (
    AbrogationDetail,
    EnrichedVerse,
    PolysemyInfo,
    VerseSnippet,
)
from src.data.schemas import Verse
from src.retrieval.data_store import DataStore


class ContextEnricher:
    """Enrich verse IDs with full context from the DataStore."""

    def __init__(self, data_store: DataStore) -> None:
        self._store = data_store

    def enrich_verse(
        self,
        verse_id: str,
        score: float | None = None,
        neighbor_range: int = 2,
    ) -> EnrichedVerse | None:
        """Enrich a single verse by ID."""
        verse = self._store.get_verse(verse_id)
        if not verse:
            return None
        return self._build_enriched(verse, score, neighbor_range)

    def enrich_verses(
        self,
        verse_ids: list[str],
        scores: dict[str, float] | None = None,
        neighbor_range: int = 2,
    ) -> list[EnrichedVerse]:
        """Enrich a list of verse IDs. Skips missing verses."""
        results = []
        for vid in verse_ids:
            score = scores.get(vid) if scores else None
            enriched = self.enrich_verse(vid, score=score, neighbor_range=neighbor_range)
            if enriched:
                results.append(enriched)
        return results

    def _build_enriched(
        self, verse: Verse, score: float | None, neighbor_range: int
    ) -> EnrichedVerse:
        # Polysemy
        polysemy_entries = self._store.get_polysemy_for_verse(verse)
        polysemy_info = [
            PolysemyInfo(
                word_arabic=pe.word_arabic,
                root=pe.root,
                senses=pe.senses,
                scholarly_note=pe.scholarly_note,
            )
            for pe in polysemy_entries
        ]

        # Abrogation
        abr = self._store.get_abrogation(verse.verse_id)
        abrogation_info = None
        if abr:
            abrogation_info = AbrogationDetail(
                abrogated_by=abr.abrogated_by,
                abrogates=abr.abrogates,
                topic=abr.topic,
                scholarly_consensus=abr.scholarly_consensus,
                note=abr.note,
            )

        # Neighbors
        neighbors = self._store.get_neighbors(verse.verse_id, range_=neighbor_range)
        neighbor_snippets = [
            VerseSnippet(
                verse_id=n.verse_id,
                text_arabic=n.text_arabic,
                text_en=n.text_en_asad,
            )
            for n in neighbors
        ]

        return EnrichedVerse(
            verse_id=verse.verse_id,
            surah_number=verse.surah_number,
            ayah_number=verse.ayah_number,
            surah_name_en=verse.surah_name_en,
            surah_name_ar=verse.surah_name_ar,
            text_arabic=verse.text_arabic,
            text_en_asad=verse.text_en_asad,
            text_fr_hamidullah=verse.text_fr_hamidullah,
            transliteration=verse.transliteration,
            revelation_period=verse.revelation_period,
            revelation_order=verse.revelation_order,
            juz=verse.juz,
            hizb=verse.hizb,
            topic_tags=verse.topic_tags,
            related_verses=verse.related_verses,
            asbab_al_nuzul=verse.asbab_al_nuzul,
            asbab_status=verse.asbab_status,
            polysemy_info=polysemy_info,
            abrogation_info=abrogation_info,
            neighbor_verses=neighbor_snippets,
            score=score,
        )

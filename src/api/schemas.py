"""Pydantic response models for the REST API."""

from typing import Any, Optional

from pydantic import BaseModel


class VerseSnippet(BaseModel):
    """Lightweight verse representation for neighbors and related verses."""

    verse_id: str
    text_arabic: str
    text_en: str


class PolysemyInfo(BaseModel):
    word_arabic: str
    root: str
    senses: list[dict[str, str]]
    scholarly_note: str | None = None


class AbrogationDetail(BaseModel):
    abrogated_by: str | None = None
    abrogates: str | None = None
    topic: str
    scholarly_consensus: str
    note: str | None = None


class EnrichedVerse(BaseModel):
    """A verse with full enrichment data attached."""

    verse_id: str
    surah_number: int
    ayah_number: int
    surah_name_en: str
    surah_name_ar: str
    text_arabic: str
    text_en_asad: str
    text_fr_hamidullah: str | None = None
    transliteration: str | None = None
    revelation_period: str
    revelation_order: int
    juz: int
    hizb: int
    topic_tags: list[str] = []
    related_verses: list[str] = []
    asbab_al_nuzul: str | None = None
    asbab_status: str = "not_documented"
    polysemy_info: list[PolysemyInfo] = []
    abrogation_info: AbrogationDetail | None = None
    neighbor_verses: list[VerseSnippet] = []
    score: float | None = None


class SearchResponse(BaseModel):
    query: str
    total: int
    results: list[EnrichedVerse]


class SurahResponse(BaseModel):
    surah_number: int
    name_en: str
    name_ar: str
    revelation_type: str
    revelation_order: int
    number_of_ayahs: int
    verses: list[EnrichedVerse]


class ThemeResponse(BaseModel):
    concept_id: str
    name_en: str
    description: str | None = None
    related_concepts: list[str] = []
    verses: list[EnrichedVerse]


class CompareResponse(BaseModel):
    verse_id: str
    text_arabic: str
    translations: dict[str, str]


class ContextResponse(BaseModel):
    center_verse: EnrichedVerse
    range: int
    neighbors: list[VerseSnippet]


class StatsResponse(BaseModel):
    total_verses: int
    total_surahs: int
    total_concepts: int
    total_chunks: dict[str, int]
    layer1_coverage: float
    layer2_coverage: float
    layer3_coverage: float
    layer4_coverage: float
    polysemy_entries: int
    abrogation_entries: int

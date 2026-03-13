"""Pydantic models for the Quran RAG dataset."""

from pydantic import BaseModel
from typing import Optional


class MorphSegment(BaseModel):
    form: str
    tag: str
    type: str  # PREFIX, STEM, SUFFIX


class WordMorphology(BaseModel):
    word_index: int
    arabic: str
    root: Optional[str] = None
    lemma: Optional[str] = None
    pos: str
    features: Optional[str] = None
    segments: list[MorphSegment] = []


class PolysemyEntry(BaseModel):
    word_arabic: str
    root: str
    senses: list[dict]  # {"meaning_en": "...", "meaning_ar": "..."}
    scholarly_note: Optional[str] = None


class AbrogationInfo(BaseModel):
    abrogated_by: Optional[str] = None
    abrogates: Optional[str] = None
    topic: str
    scholarly_consensus: str  # "majority_agree", "debated", "minority_view"
    note: Optional[str] = None


class Verse(BaseModel):
    # Layer 1: Text
    verse_id: str  # "2:255"
    surah_number: int
    ayah_number: int
    text_arabic: str
    text_en_asad: str
    text_fr_hamidullah: Optional[str] = None
    transliteration: Optional[str] = None

    # Layer 2: Linguistics
    morphology: list[WordMorphology] = []
    polysemous_words: list[PolysemyEntry] = []

    # Layer 3: Historical context
    asbab_al_nuzul: Optional[str] = None
    asbab_status: str = "not_documented"  # "documented", "not_documented", "debated"
    revelation_period: str  # "meccan" or "medinan"
    revelation_order: int  # chronological order of the surah

    # Layer 4: Thematic links
    topic_tags: list[str] = []
    related_verses: list[str] = []
    abrogation: Optional[AbrogationInfo] = None

    # Metadata
    surah_name_ar: str
    surah_name_en: str
    juz: int
    hizb: int
    page: Optional[int] = None


class Chapter(BaseModel):
    surah_number: int
    name_ar: str
    name_en: str
    revelation_type: str  # "meccan" or "medinan"
    revelation_order: int
    number_of_ayahs: int
    verse_ids: list[str] = []


class OntologyConcept(BaseModel):
    concept_id: str
    name_en: str
    description: Optional[str] = None
    parent_concepts: list[str] = []
    child_concepts: list[str] = []
    related_concepts: list[str] = []
    verses: list[str] = []


class DatasetStats(BaseModel):
    total_verses: int
    total_surahs: int
    layer1_coverage: float  # fraction of verses with text
    layer2_coverage: float  # fraction of verses with morphology
    layer3_coverage: float  # fraction of verses with asbab
    layer4_coverage: float  # fraction of verses with topic_tags
    polysemy_entries: int
    abrogation_entries: int

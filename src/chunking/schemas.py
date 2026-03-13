"""Pydantic models for chunks at all three granularity levels."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class ChunkType(str, Enum):
    VERSE = "verse"
    THEMATIC_GROUP = "thematic_group"
    SURAH_SUMMARY = "surah_summary"


class VerseChunk(BaseModel):
    """Level 1: A single verse with its full enriched context serialized for embedding."""

    chunk_id: str  # "verse:2:255"
    chunk_type: ChunkType = ChunkType.VERSE
    verse_id: str  # "2:255"
    surah_number: int
    ayah_number: int
    surah_name_en: str
    surah_name_ar: str

    # The text that gets embedded (constructed by the chunker)
    text_for_embedding: str

    # Metadata stored alongside the vector (not embedded, used for filtering/display)
    revelation_period: str  # "meccan" / "medinan"
    juz: int
    hizb: int
    topic_tags: list[str] = []
    has_asbab: bool = False
    page: Optional[int] = None


class ThematicGroupChunk(BaseModel):
    """Level 2: A cluster of consecutive verses grouped by semantic similarity."""

    chunk_id: str  # "group:2:1-5"
    chunk_type: ChunkType = ChunkType.THEMATIC_GROUP
    surah_number: int
    surah_name_en: str
    surah_name_ar: str
    start_ayah: int
    end_ayah: int
    verse_ids: list[str]  # ["2:1", "2:2", ..., "2:5"]

    text_for_embedding: str  # Combined text of all verses in the group

    # Aggregated metadata
    revelation_period: str
    juz: int  # juz of the first verse
    topic_tags: list[str] = []  # Union of all topic tags in the group
    verse_count: int


class SurahSummaryChunk(BaseModel):
    """Level 3: A summary representation of an entire surah."""

    chunk_id: str  # "surah:2"
    chunk_type: ChunkType = ChunkType.SURAH_SUMMARY
    surah_number: int
    surah_name_en: str
    surah_name_ar: str
    revelation_period: str
    revelation_order: int
    number_of_ayahs: int
    verse_ids: list[str]  # All verse IDs in the surah

    text_for_embedding: str  # The summary text

    # Aggregated metadata
    juz_range: list[int] = []  # [1, 2] if surah spans juz 1-2
    topic_tags: list[str] = []  # Union of all topic tags across the surah


# Union type for any chunk
Chunk = VerseChunk | ThematicGroupChunk | SurahSummaryChunk

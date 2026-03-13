"""Tests for the chunking module."""

import pytest

from src.chunking.schemas import ChunkType, SurahSummaryChunk, ThematicGroupChunk, VerseChunk
from src.chunking.surah_summarizer import build_surah_summary_text, create_surah_summaries
from src.chunking.thematic_grouper import group_consecutive_verses
from src.chunking.verse_chunker import build_verse_text, create_verse_chunks
from src.data.schemas import Verse, WordMorphology


@pytest.fixture
def sample_verse() -> Verse:
    """A minimal valid Verse for testing."""
    return Verse(
        verse_id="2:255",
        surah_number=2,
        ayah_number=255,
        text_arabic="\u0627\u0644\u0644\u0651\u064e\u0647\u064f \u0644\u064e\u0627 \u0625\u0650\u0644\u064e\u0670\u0647\u064e \u0625\u0650\u0644\u0651\u064e\u0627 \u0647\u064f\u0648\u064e \u0627\u0644\u0652\u062d\u064e\u064a\u0651\u064f \u0627\u0644\u0652\u0642\u064e\u064a\u0651\u064f\u0648\u0645\u064f",
        text_en_asad="God - there is no deity save Him, the Ever-Living, the Self-Subsistent Fount of All Being.",
        revelation_period="medinan",
        revelation_order=87,
        surah_name_ar="\u0633\u064f\u0648\u0631\u064e\u0629\u064f \u0627\u0644\u0628\u064e\u0642\u064e\u0631\u064e\u0629\u0650",
        surah_name_en="Al-Baqara",
        juz=3,
        hizb=17,
        topic_tags=["allah", "monotheism"],
        morphology=[
            WordMorphology(
                word_index=1,
                arabic="{ll~ahu",
                root="Alh",
                lemma="{ll~ah",
                pos="PN",
                segments=[],
            ),
            WordMorphology(
                word_index=2,
                arabic="laA",
                root=None,
                lemma="laA",
                pos="NEG",
                segments=[],
            ),
        ],
    )


@pytest.fixture
def sample_verses() -> list[Verse]:
    """Multiple minimal verses for a test surah."""
    verses = []
    for i in range(1, 8):
        verses.append(
            Verse(
                verse_id=f"1:{i}",
                surah_number=1,
                ayah_number=i,
                text_arabic=f"Arabic text {i}",
                text_en_asad=f"English text {i}",
                revelation_period="meccan",
                revelation_order=5,
                surah_name_ar="Al-Fatiha",
                surah_name_en="Al-Faatiha",
                juz=1,
                hizb=1,
                topic_tags=["allah"] if i <= 3 else ["guidance"],
            )
        )
    return verses


class TestVerseChunker:
    def test_build_verse_text_includes_arabic_and_english(self, sample_verse):
        text = build_verse_text(sample_verse)
        assert sample_verse.text_arabic in text
        assert "Ever-Living" in text

    def test_build_verse_text_includes_metadata(self, sample_verse):
        text = build_verse_text(sample_verse)
        assert "Al-Baqara" in text
        assert "255" in text
        assert "Medinan" in text
        assert "Juz 3" in text

    def test_build_verse_text_includes_topics(self, sample_verse):
        text = build_verse_text(sample_verse)
        assert "Topics:" in text
        assert "allah" in text
        assert "monotheism" in text

    def test_build_verse_text_includes_roots(self, sample_verse):
        text = build_verse_text(sample_verse)
        assert "Roots:" in text
        assert "Alh" in text

    def test_create_verse_chunks_count(self, sample_verse):
        chunks = create_verse_chunks([sample_verse])
        assert len(chunks) == 1
        assert chunks[0].chunk_type == ChunkType.VERSE
        assert chunks[0].chunk_id == "verse:2:255"

    def test_verse_chunk_has_text_for_embedding(self, sample_verse):
        chunks = create_verse_chunks([sample_verse])
        assert len(chunks[0].text_for_embedding) > 0

    def test_verse_chunk_metadata(self, sample_verse):
        chunks = create_verse_chunks([sample_verse])
        chunk = chunks[0]
        assert chunk.surah_number == 2
        assert chunk.ayah_number == 255
        assert chunk.revelation_period == "medinan"
        assert chunk.juz == 3
        assert chunk.has_asbab is False


class TestThematicGrouper:
    def test_single_group_for_small_input(self):
        """Input smaller than MIN_GROUP_SIZE should become one group."""
        import numpy as np

        embeddings = np.random.rand(2, 64).astype(np.float32)
        groups = group_consecutive_verses(embeddings, min_group=3)
        assert len(groups) == 1
        assert groups[0] == [0, 1]

    def test_group_respects_max_size(self):
        """No group should exceed MAX_GROUP_SIZE."""
        import numpy as np

        # Create 20 similar embeddings (all ones) -> should be one group initially
        # but max_group=7 forces splitting
        embeddings = np.ones((20, 64), dtype=np.float32)
        # Add small noise to avoid division by zero
        embeddings += np.random.rand(20, 64) * 0.01
        groups = group_consecutive_verses(embeddings, max_group=7, min_group=3)
        for g in groups:
            assert len(g) <= 7

    def test_groups_cover_all_verses(self):
        """All verse indices should appear exactly once across all groups."""
        import numpy as np

        n = 15
        embeddings = np.random.rand(n, 64).astype(np.float32)
        groups = group_consecutive_verses(embeddings)
        all_indices = []
        for g in groups:
            all_indices.extend(g)
        assert sorted(all_indices) == list(range(n))


class TestSurahSummarizer:
    def test_summary_includes_surah_name(self, sample_verses):
        text = build_surah_summary_text(
            surah_number=1,
            surah_name_en="Al-Faatiha",
            surah_name_ar="Al-Fatiha",
            revelation_period="meccan",
            revelation_order=5,
            surah_verses=sample_verses,
        )
        assert "Al-Faatiha" in text
        assert "7 verses" in text

    def test_summary_includes_revelation_info(self, sample_verses):
        text = build_surah_summary_text(
            surah_number=1,
            surah_name_en="Al-Faatiha",
            surah_name_ar="Al-Fatiha",
            revelation_period="meccan",
            revelation_order=5,
            surah_verses=sample_verses,
        )
        assert "Meccan" in text
        assert "Revelation order: 5" in text

    def test_create_surah_summaries_count(self, sample_verses):
        chunks = create_surah_summaries({1: sample_verses})
        assert len(chunks) == 1
        assert chunks[0].chunk_type == ChunkType.SURAH_SUMMARY
        assert chunks[0].chunk_id == "surah:1"
        assert chunks[0].number_of_ayahs == 7

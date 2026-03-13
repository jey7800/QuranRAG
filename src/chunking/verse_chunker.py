"""Level 1: Build one chunk per verse with enriched context.

Each verse becomes a VerseChunk whose `text_for_embedding` combines:
- Structural header (surah name, verse number, period, juz)
- Arabic text
- English translation (Asad)
- Topic tags
- Unique morphological roots

French text and transliteration are excluded from embedding text to avoid
diluting the vector space — they are available in metadata for display.
"""

from loguru import logger
from tqdm import tqdm

from src.chunking.schemas import VerseChunk
from src.data.schemas import Verse


def build_verse_text(verse: Verse) -> str:
    """Construct the embedding text for a single verse.

    Format:
        [Surah Al-Baqara (2), Verse 255 | Medinan | Juz 3]
        Arabic: <arabic text>
        English: <english translation>
        Topics: monotheism, divine-attributes
        Roots: Alh, Hyy, qwm
    """
    parts = []

    # Structural header
    period = verse.revelation_period.capitalize()
    header = (
        f"[Surah {verse.surah_name_en} ({verse.surah_number}), "
        f"Verse {verse.ayah_number} | {period} | Juz {verse.juz}]"
    )
    parts.append(header)

    # Arabic text
    parts.append(f"Arabic: {verse.text_arabic}")

    # English translation
    parts.append(f"English: {verse.text_en_asad}")

    # Topic tags (if any)
    if verse.topic_tags:
        unique_tags = sorted(set(verse.topic_tags))
        parts.append(f"Topics: {', '.join(unique_tags)}")

    # Unique morphological roots (strong cross-lingual signal)
    if verse.morphology:
        roots = sorted(
            {w.root for w in verse.morphology if w.root},
        )
        if roots:
            parts.append(f"Roots: {', '.join(roots)}")

    return "\n".join(parts)


def create_verse_chunks(verses: list[Verse]) -> list[VerseChunk]:
    """Convert all verses to VerseChunk objects.

    Args:
        verses: Parsed Verse objects from verses.jsonl.

    Returns:
        List of VerseChunk objects (one per verse, 6,236 expected).
    """
    chunks = []
    for verse in tqdm(verses, desc="Building verse chunks", disable=len(verses) < 100):
        text = build_verse_text(verse)
        chunk = VerseChunk(
            chunk_id=f"verse:{verse.verse_id}",
            verse_id=verse.verse_id,
            surah_number=verse.surah_number,
            ayah_number=verse.ayah_number,
            surah_name_en=verse.surah_name_en,
            surah_name_ar=verse.surah_name_ar,
            text_for_embedding=text,
            revelation_period=verse.revelation_period,
            juz=verse.juz,
            hizb=verse.hizb,
            topic_tags=list(set(verse.topic_tags)),
            has_asbab=verse.asbab_status == "documented",
            page=verse.page,
        )
        chunks.append(chunk)

    logger.info(f"Created {len(chunks)} verse chunks")
    return chunks

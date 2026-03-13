"""Level 3: Build one summary chunk per surah.

Each surah becomes a SurahSummaryChunk whose `text_for_embedding` captures:
- Surah identity (name, number, verse count, period, revelation order)
- Key themes (topic tags sorted by frequency)
- Opening verses (first 3 verses in English)
"""

from collections import Counter

from loguru import logger

from src.chunking.schemas import SurahSummaryChunk
from src.data.schemas import Verse


def build_surah_summary_text(
    surah_number: int,
    surah_name_en: str,
    surah_name_ar: str,
    revelation_period: str,
    revelation_order: int,
    surah_verses: list[Verse],
) -> str:
    """Build the embedding text for a surah summary.

    Format:
        Surah Al-Baqara (2) - The Cow
        286 verses | Medinan | Revelation order: 87 | Juz: 1-3

        Key themes: monotheism, divine-attributes, jurisprudence, ...

        Opening:
        Verse 2:1: Alif. Lam. Mim.
        Verse 2:2: THIS DIVINE WRIT - let there be no doubt about it...
        Verse 2:3: who believe in [the existence of] that which is beyond...

        Core topics: <all unique tags sorted by frequency>
    """
    parts = []

    # Identity header
    period = revelation_period.capitalize()
    num_verses = len(surah_verses)

    # Juz range
    juz_values = sorted({v.juz for v in surah_verses})
    if len(juz_values) == 1:
        juz_str = f"Juz {juz_values[0]}"
    else:
        juz_str = f"Juz {juz_values[0]}-{juz_values[-1]}"

    parts.append(f"Surah {surah_name_en} ({surah_number})")
    parts.append(f"{num_verses} verses | {period} | Revelation order: {revelation_order} | {juz_str}")

    # Key themes by frequency
    tag_counter = Counter()
    for v in surah_verses:
        tag_counter.update(v.topic_tags)

    if tag_counter:
        top_tags = [tag for tag, _ in tag_counter.most_common(10)]
        parts.append(f"\nKey themes: {', '.join(top_tags)}")

    # Opening verses (first 3)
    opening_verses = surah_verses[:3]
    if opening_verses:
        parts.append("\nOpening:")
        for v in opening_verses:
            parts.append(f"Verse {v.verse_id}: {v.text_en_asad}")

    # All unique topics
    if tag_counter:
        all_tags = [tag for tag, _ in tag_counter.most_common()]
        parts.append(f"\nCore topics: {', '.join(all_tags)}")

    return "\n".join(parts)


def create_surah_summaries(
    verses_by_surah: dict[int, list[Verse]],
) -> list[SurahSummaryChunk]:
    """Build summary chunks for all 114 surahs.

    Args:
        verses_by_surah: Dict mapping surah_number -> list of Verse objects (sorted by ayah).

    Returns:
        List of 114 SurahSummaryChunk objects.
    """
    chunks = []

    for surah_num in sorted(verses_by_surah.keys()):
        surah_verses = verses_by_surah[surah_num]
        if not surah_verses:
            continue

        first_verse = surah_verses[0]
        surah_name_en = first_verse.surah_name_en
        surah_name_ar = first_verse.surah_name_ar
        revelation_period = first_verse.revelation_period
        revelation_order = first_verse.revelation_order

        text = build_surah_summary_text(
            surah_number=surah_num,
            surah_name_en=surah_name_en,
            surah_name_ar=surah_name_ar,
            revelation_period=revelation_period,
            revelation_order=revelation_order,
            surah_verses=surah_verses,
        )

        # Aggregate metadata
        juz_values = sorted({v.juz for v in surah_verses})
        all_tags = list({tag for v in surah_verses for tag in v.topic_tags})
        verse_ids = [v.verse_id for v in surah_verses]

        chunk = SurahSummaryChunk(
            chunk_id=f"surah:{surah_num}",
            surah_number=surah_num,
            surah_name_en=surah_name_en,
            surah_name_ar=surah_name_ar,
            revelation_period=revelation_period,
            revelation_order=revelation_order,
            number_of_ayahs=len(surah_verses),
            verse_ids=verse_ids,
            text_for_embedding=text,
            juz_range=juz_values,
            topic_tags=all_tags,
        )
        chunks.append(chunk)

    logger.info(f"Created {len(chunks)} surah summary chunks")
    return chunks

"""Validate the merged dataset and compute statistics."""

import json

from loguru import logger
from pydantic import ValidationError

from src.config import PROCESSED_DIR, STATS_JSON, TOTAL_SURAHS, TOTAL_VERSES, VERSES_JSONL
from src.data.schemas import Verse


def validate_dataset() -> dict:
    """Validate verses.jsonl and compute stats. Returns the stats dict."""
    if not VERSES_JSONL.exists():
        raise FileNotFoundError(f"Dataset not found: {VERSES_JSONL}")

    verses = []
    errors = []

    with open(VERSES_JSONL, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            raw = json.loads(line)
            try:
                verse = Verse(**raw)
                verses.append(verse)
            except ValidationError as e:
                errors.append(f"Line {i} ({raw.get('verse_id', '?')}): {e}")

    # Basic assertions
    verse_ids = [v.verse_id for v in verses]
    surahs = {v.surah_number for v in verses}

    checks = {
        "total_verses": len(verses),
        "expected_verses": TOTAL_VERSES,
        "verses_match": len(verses) == TOTAL_VERSES,
        "total_surahs": len(surahs),
        "expected_surahs": TOTAL_SURAHS,
        "surahs_match": len(surahs) == TOTAL_SURAHS,
        "duplicate_verse_ids": len(verse_ids) - len(set(verse_ids)),
        "validation_errors": len(errors),
    }

    # Coverage stats
    has_text = sum(1 for v in verses if v.text_arabic and v.text_en_asad)
    has_morphology = sum(1 for v in verses if v.morphology)
    has_asbab = sum(1 for v in verses if v.asbab_status == "documented")
    has_topics = sum(1 for v in verses if v.topic_tags)
    has_transliteration = sum(1 for v in verses if v.transliteration)
    has_french = sum(1 for v in verses if v.text_fr_hamidullah)
    poly_count = sum(1 for v in verses if v.polysemous_words)
    abrogation_count = sum(1 for v in verses if v.abrogation)

    total = len(verses) or 1  # avoid division by zero

    stats = {
        "total_verses": len(verses),
        "total_surahs": len(surahs),
        "layer1_coverage": round(has_text / total, 4),
        "layer1_french_coverage": round(has_french / total, 4),
        "layer1_transliteration_coverage": round(has_transliteration / total, 4),
        "layer2_coverage": round(has_morphology / total, 4),
        "layer3_coverage": round(has_asbab / total, 4),
        "layer4_coverage": round(has_topics / total, 4),
        "polysemy_entries": poly_count,
        "abrogation_entries": abrogation_count,
        "validation_errors": len(errors),
    }

    # Log results
    logger.info("=== Dataset Validation Results ===")
    for key, val in checks.items():
        status = "OK" if val is True or (isinstance(val, int) and val == 0) else str(val)
        logger.info(f"  {key}: {status}")

    logger.info("=== Coverage Statistics ===")
    logger.info(f"  Layer 1 (Text):       {stats['layer1_coverage']*100:.1f}%")
    logger.info(f"  Layer 1 (French):     {stats['layer1_french_coverage']*100:.1f}%")
    logger.info(f"  Layer 1 (Translit):   {stats['layer1_transliteration_coverage']*100:.1f}%")
    logger.info(f"  Layer 2 (Morphology): {stats['layer2_coverage']*100:.1f}%")
    logger.info(f"  Layer 3 (Asbab):      {stats['layer3_coverage']*100:.1f}%")
    logger.info(f"  Layer 4 (Themes):     {stats['layer4_coverage']*100:.1f}%")
    logger.info(f"  Polysemy entries:     {stats['polysemy_entries']}")
    logger.info(f"  Abrogation entries:   {stats['abrogation_entries']}")

    if errors:
        logger.warning(f"{len(errors)} validation errors:")
        for e in errors[:10]:
            logger.warning(f"  {e}")
        if len(errors) > 10:
            logger.warning(f"  ... and {len(errors) - 10} more")

    # Save stats
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    STATS_JSON.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info(f"Stats saved to {STATS_JSON}")

    return stats

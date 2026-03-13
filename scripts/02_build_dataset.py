"""Orchestrator: merge all layers and validate the dataset."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from src.config import CHAPTERS_JSON, RAW_ONTOLOGY_DIR, RAW_QURAN_API_DIR
from src.data.fetch_quran_api import parse_surah_response
from src.data.fetch_quran_md import fetch_quran_md
from src.data.merge_dataset import merge_dataset
from src.data.parse_asbab import parse_asbab_pdf
from src.data.parse_morphology import parse_morphology_tsv
from src.data.validate_dataset import validate_dataset


def load_cached_api_data() -> tuple[list[dict], list[dict]]:
    """Load previously fetched API data from cache."""
    chapters = []
    verses = []
    for surah_num in range(1, 115):
        path = RAW_QURAN_API_DIR / f"surah_{surah_num}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Cached surah {surah_num} not found. Run 01_fetch_data.py first."
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        ch, vs = parse_surah_response(raw)
        chapters.append(ch)
        verses.extend(vs)
    return chapters, verses


def main():
    logger.info("=== Phase 1 Step 2: Building dataset ===")

    # Load Layer 1 from cache
    logger.info("[1/6] Loading cached API data...")
    chapters, verses = load_cached_api_data()
    logger.info(f"  → {len(verses)} verses from {len(chapters)} surahs")

    # Load transliterations
    logger.info("[2/6] Loading transliterations...")
    transliterations = fetch_quran_md()
    logger.info(f"  → {len(transliterations)} transliterations")

    # Parse morphology (Layer 2)
    logger.info("[3/6] Parsing morphology (Layer 2)...")
    morphology = parse_morphology_tsv()
    logger.info(f"  → {len(morphology)} verses with morphology")

    # Parse asbab al-nuzul (Layer 3)
    logger.info("[4/6] Parsing asbab al-nuzul (Layer 3)...")
    asbab = parse_asbab_pdf()
    logger.info(f"  → {len(asbab)} verses with asbab")

    # Load ontology (Layer 4)
    logger.info("[5/6] Loading ontology (Layer 4)...")
    concepts_path = RAW_ONTOLOGY_DIR / "concepts_raw.json"
    if concepts_path.exists():
        concepts = json.loads(concepts_path.read_text(encoding="utf-8"))
    else:
        logger.warning("No ontology data found. Run 01_fetch_data.py first.")
        concepts = []
    logger.info(f"  → {len(concepts)} concepts")

    # Merge all layers
    logger.info("[6/6] Merging all layers...")
    merged = merge_dataset(chapters, verses, transliterations, morphology, asbab, concepts)
    logger.info(f"  → {len(merged)} verses merged")

    # Validate
    logger.info("")
    logger.info("=== Validation ===")
    stats = validate_dataset()

    logger.info("")
    logger.info("=== Phase 1 complete ===")


if __name__ == "__main__":
    main()

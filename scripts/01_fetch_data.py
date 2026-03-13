"""Orchestrator: fetch all raw data sources for Phase 1."""

import asyncio
import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from src.data.fetch_quran_api import fetch_all_surahs
from src.data.fetch_quran_md import fetch_quran_md
from src.data.scrape_ontology import fetch_ontology_from_github


async def main():
    logger.info("=== Phase 1 Step 1: Fetching raw data ===")

    # Layer 1: Quran text from alquran.cloud
    logger.info("[1/3] Fetching Quran text from alquran.cloud API...")
    chapters, verses = await fetch_all_surahs()
    logger.info(f"  → {len(chapters)} surahs, {len(verses)} verses")

    # Layer 1 supplement: transliteration from HuggingFace
    logger.info("[2/3] Fetching transliteration from HuggingFace...")
    transliterations = fetch_quran_md()
    logger.info(f"  → {len(transliterations)} transliterations")

    # Layer 4: Ontology from GitHub
    logger.info("[3/3] Fetching ontology from GitHub...")
    concepts = await fetch_ontology_from_github()
    logger.info(f"  → {len(concepts)} concepts")

    # Info about manual steps
    logger.info("")
    logger.info("=== Manual steps needed ===")
    logger.info("[MANUAL] Layer 2 - Morphology:")
    logger.info("  Download from https://corpus.quran.com/download/")
    logger.info("  Place quranic-corpus-morphology-0.4.txt in data/raw/quranic_corpus/")
    logger.info("")
    logger.info("[MANUAL] Layer 3 - Asbab al-Nuzul:")
    logger.info("  Download from https://www.altafsir.com/Books/Asbab%20Al-Nuzul%20by%20Al-Wahidi.pdf")
    logger.info("  Place al_wahidi_en.pdf in data/raw/asbab_al_nuzul/")
    logger.info("")
    logger.info("=== Fetching complete. Run 02_build_dataset.py next. ===")


if __name__ == "__main__":
    asyncio.run(main())

"""Orchestrate all three chunking levels.

Loads verses from the Phase 1 dataset, builds all chunk types,
and saves them to JSONL files in data/processed/chunks/.
"""

import json
from collections import defaultdict

import numpy as np
from loguru import logger

from src.chunking.schemas import SurahSummaryChunk, ThematicGroupChunk, VerseChunk
from src.chunking.surah_summarizer import create_surah_summaries
from src.chunking.thematic_grouper import create_thematic_groups
from src.chunking.verse_chunker import build_verse_text, create_verse_chunks
from src.config import (
    CHAPTERS_JSON,
    CHUNKS_DIR,
    SURAH_CHUNKS_JSONL,
    THEMATIC_CHUNKS_JSONL,
    VERSE_CHUNKS_JSONL,
    VERSES_JSONL,
)
from src.data.schemas import Verse
from src.embedding.base import Embedder


def load_verses() -> list[Verse]:
    """Load and parse all verses from the JSONL dataset."""
    if not VERSES_JSONL.exists():
        raise FileNotFoundError(
            f"Verse dataset not found: {VERSES_JSONL}. Run scripts/02_build_dataset.py first."
        )

    verses = []
    with open(VERSES_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                verses.append(Verse.model_validate_json(line))

    logger.info(f"Loaded {len(verses)} verses from {VERSES_JSONL}")
    return verses


def group_verses_by_surah(verses: list[Verse]) -> dict[int, list[Verse]]:
    """Group verses by surah number, sorted by ayah."""
    by_surah: dict[int, list[Verse]] = defaultdict(list)
    for v in verses:
        by_surah[v.surah_number].append(v)

    # Sort each surah's verses by ayah number
    for surah_num in by_surah:
        by_surah[surah_num].sort(key=lambda v: v.ayah_number)

    return dict(by_surah)


def save_chunks(chunks: list, filepath) -> int:
    """Save chunks to a JSONL file.

    Returns the number of chunks saved.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")
    logger.info(f"Saved {len(chunks)} chunks to {filepath}")
    return len(chunks)


def load_cached_chunks(filepath, model_cls):
    """Load chunks from a cached JSONL file."""
    if not filepath.exists():
        return None
    chunks = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(model_cls.model_validate_json(line))
    logger.info(f"Loaded {len(chunks)} cached chunks from {filepath}")
    return chunks


def run_chunking_pipeline(
    embedder: Embedder,
    skip_verse: bool = False,
    skip_thematic: bool = False,
    skip_surah: bool = False,
) -> dict[str, list]:
    """Run the full chunking pipeline.

    Args:
        embedder: Embedder instance (needed for thematic grouping).
        skip_verse: If True, skip verse-level chunks (load from cache).
        skip_thematic: If True, skip thematic group chunks (load from cache).
        skip_surah: If True, skip surah summary chunks (load from cache).

    Returns:
        Dict with keys "verse", "thematic", "surah" -> list of chunks.

    Side effects:
        Saves chunks to data/processed/chunks/ as JSONL files.
    """
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    # Load verse data
    verses = load_verses()
    verses_by_surah = group_verses_by_surah(verses)

    result: dict[str, list] = {}

    # Level 1: Verse chunks
    if skip_verse:
        cached = load_cached_chunks(VERSE_CHUNKS_JSONL, VerseChunk)
        if cached:
            result["verse"] = cached
        else:
            logger.warning("No cached verse chunks found, building from scratch")
            result["verse"] = create_verse_chunks(verses)
            save_chunks(result["verse"], VERSE_CHUNKS_JSONL)
    else:
        logger.info("=== Level 1: Verse chunks ===")
        result["verse"] = create_verse_chunks(verses)
        save_chunks(result["verse"], VERSE_CHUNKS_JSONL)

    # Level 2: Thematic groups (needs embedder for similarity computation)
    if skip_thematic:
        cached = load_cached_chunks(THEMATIC_CHUNKS_JSONL, ThematicGroupChunk)
        if cached:
            result["thematic"] = cached
        else:
            logger.warning("No cached thematic chunks found, building from scratch")
            thematic_chunks, _ = create_thematic_groups(verses_by_surah, embedder)
            result["thematic"] = thematic_chunks
            save_chunks(result["thematic"], THEMATIC_CHUNKS_JSONL)
    else:
        logger.info("=== Level 2: Thematic group chunks ===")
        thematic_chunks, surah_embeddings = create_thematic_groups(verses_by_surah, embedder)
        result["thematic"] = thematic_chunks
        save_chunks(result["thematic"], THEMATIC_CHUNKS_JSONL)

    # Level 3: Surah summaries
    if skip_surah:
        cached = load_cached_chunks(SURAH_CHUNKS_JSONL, SurahSummaryChunk)
        if cached:
            result["surah"] = cached
        else:
            logger.warning("No cached surah chunks found, building from scratch")
            result["surah"] = create_surah_summaries(verses_by_surah)
            save_chunks(result["surah"], SURAH_CHUNKS_JSONL)
    else:
        logger.info("=== Level 3: Surah summary chunks ===")
        result["surah"] = create_surah_summaries(verses_by_surah)
        save_chunks(result["surah"], SURAH_CHUNKS_JSONL)

    logger.info(
        f"Chunking complete: {len(result['verse'])} verse, "
        f"{len(result['thematic'])} thematic, {len(result['surah'])} surah"
    )

    return result

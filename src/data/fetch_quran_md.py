"""Fetch transliteration data from HuggingFace Quran-MD dataset.

The dataset is ~35GB (includes audio). We stream it and only extract text columns
to avoid downloading the full dataset.
"""

import json

from loguru import logger
from tqdm import tqdm

from src.config import QURAN_MD_DATASET, RAW_QURAN_MD_DIR, TOTAL_VERSES


def fetch_quran_md() -> dict[str, str]:
    """Load Quran-MD dataset via streaming and return verse_id → transliteration mapping."""
    from datasets import load_dataset

    RAW_QURAN_MD_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_QURAN_MD_DIR / "transliterations.json"

    if cache_path.exists():
        logger.info("Using cached transliteration data")
        return json.loads(cache_path.read_text(encoding="utf-8"))

    logger.info(f"Streaming HuggingFace dataset: {QURAN_MD_DATASET} (text only, skipping audio)")

    # Stream to avoid downloading the full 35GB dataset
    # select specific columns to skip audio decoding entirely
    ds = load_dataset(
        QURAN_MD_DATASET,
        split="train",
        streaming=True,
    ).select_columns(["surah_id", "ayah_id", "ayah_tr"])

    transliterations = {}
    rows_seen = 0

    for row in tqdm(ds, desc="Streaming transliterations", total=TOTAL_VERSES):
        rows_seen += 1
        surah = row.get("surah_id") or row.get("surah_number") or row.get("surah")
        ayah = row.get("ayah_id") or row.get("ayah_number") or row.get("ayah")
        translit = row.get("ayah_tr") or row.get("transliteration") or row.get("translit")

        if surah and ayah and translit:
            verse_id = f"{surah}:{ayah}"
            if verse_id not in transliterations:
                transliterations[verse_id] = translit

        # Early stop: dataset has 30 reciters × 6236 verses = 187k rows
        # We only need one transliteration per verse.
        # Stop once we've seen enough rows to cover one full reciter cycle.
        if rows_seen >= TOTAL_VERSES and len(transliterations) > 0:
            logger.info(
                f"Processed {rows_seen} rows — collected {len(transliterations)} "
                f"unique transliterations. Stopping early."
            )
            break

    # Cache
    cache_path.write_text(
        json.dumps(transliterations, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        f"Loaded {len(transliterations)} unique transliterations "
        f"(expected ~{TOTAL_VERSES})"
    )
    return transliterations


def run():
    """Entry point."""
    return fetch_quran_md()


if __name__ == "__main__":
    run()

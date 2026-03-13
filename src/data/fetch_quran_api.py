"""Fetch Quran text from alquran.cloud API (Arabic Uthmani + EN Asad + FR Hamidullah)."""

import asyncio
import json
import time

import httpx
from loguru import logger
from tqdm import tqdm

from src.config import (
    ALQURAN_CLOUD_BASE_URL,
    ALQURAN_EDITIONS,
    API_SLEEP_SECONDS,
    RAW_QURAN_API_DIR,
    TOTAL_SURAHS,
)


async def fetch_surah(client: httpx.AsyncClient, surah_number: int) -> dict:
    """Fetch a single surah with multiple editions."""
    url = f"{ALQURAN_CLOUD_BASE_URL}/surah/{surah_number}/editions/{ALQURAN_EDITIONS}"
    resp = await client.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


async def fetch_meta(client: httpx.AsyncClient) -> dict:
    """Fetch Quran metadata."""
    url = f"{ALQURAN_CLOUD_BASE_URL}/meta"
    resp = await client.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def parse_surah_response(raw: dict) -> tuple[dict, list[dict]]:
    """Parse a multi-edition surah response into chapter metadata + verse list."""
    editions = raw["data"]
    # editions[0] = Arabic Uthmani, editions[1] = EN Asad, editions[2] = FR Hamidullah
    arabic_edition = editions[0]
    en_edition = editions[1]
    fr_edition = editions[2] if len(editions) > 2 else None

    surah_meta = {
        "surah_number": arabic_edition["number"],
        "name_ar": arabic_edition["name"],
        "name_en": arabic_edition["englishName"],
        "revelation_type": arabic_edition["revelationType"].lower(),
        "number_of_ayahs": arabic_edition["numberOfAyahs"],
    }

    verses = []
    for i, ayah_ar in enumerate(arabic_edition["ayahs"]):
        ayah_en = en_edition["ayahs"][i]
        ayah_fr = fr_edition["ayahs"][i] if fr_edition else None

        verse = {
            "verse_id": f"{surah_meta['surah_number']}:{ayah_ar['numberInSurah']}",
            "surah_number": surah_meta["surah_number"],
            "ayah_number": ayah_ar["numberInSurah"],
            "text_arabic": ayah_ar["text"],
            "text_en_asad": ayah_en["text"],
            "text_fr_hamidullah": ayah_fr["text"] if ayah_fr else None,
            "juz": ayah_ar["juz"],
            "hizb": ayah_ar["hizbQuarter"],
            "page": ayah_ar.get("page"),
            "surah_name_ar": surah_meta["name_ar"],
            "surah_name_en": surah_meta["name_en"],
            "revelation_type": surah_meta["revelation_type"],
        }
        verses.append(verse)

    return surah_meta, verses


async def fetch_all_surahs() -> tuple[list[dict], list[dict]]:
    """Fetch all 114 surahs. Returns (chapters_list, all_verses_list)."""
    RAW_QURAN_API_DIR.mkdir(parents=True, exist_ok=True)

    all_chapters = []
    all_verses = []

    async with httpx.AsyncClient() as client:
        # Fetch metadata
        logger.info("Fetching Quran metadata...")
        meta = await fetch_meta(client)
        meta_path = RAW_QURAN_API_DIR / "meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        # Fetch each surah
        for surah_num in tqdm(range(1, TOTAL_SURAHS + 1), desc="Fetching surahs"):
            cache_path = RAW_QURAN_API_DIR / f"surah_{surah_num}.json"

            if cache_path.exists():
                logger.debug(f"Using cached surah {surah_num}")
                raw = json.loads(cache_path.read_text(encoding="utf-8"))
            else:
                logger.debug(f"Fetching surah {surah_num}...")
                raw = await fetch_surah(client, surah_num)
                cache_path.write_text(
                    json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                await asyncio.sleep(API_SLEEP_SECONDS)

            chapter, verses = parse_surah_response(raw)
            all_chapters.append(chapter)
            all_verses.extend(verses)

    logger.info(f"Fetched {len(all_chapters)} surahs, {len(all_verses)} verses")
    return all_chapters, all_verses


def run():
    """Entry point."""
    chapters, verses = asyncio.run(fetch_all_surahs())
    return chapters, verses


if __name__ == "__main__":
    run()

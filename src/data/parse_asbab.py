"""Parse Al-Wahidi's Asbab al-Nuzul PDF into structured JSON.

The PDF is structured as:
  - Surah headers in bold: (Al-Baqarah)
  - Verse section headers: [2:14] or [1:1-7] (ranges)
  - Narration text follows each header until the next [surah:verse] header
"""

import json
import re

from loguru import logger
from tqdm import tqdm

from src.config import ASBAB_PDF, ASBAB_PARSED, RAW_ASBAB_DIR


# Matches [2:14], [1:1-7], [2:1-2] — section headers in the PDF
# Group 1: surah, Group 2: start verse, Group 3: optional end verse
SECTION_HEADER_RE = re.compile(r"\[(\d{1,3}):(\d{1,3})(?:-(\d{1,3}))?\]")


def expand_verse_refs(surah: int, start: int, end: int | None) -> list[str]:
    """Expand a verse reference (possibly a range) into individual verse_ids."""
    if end is None or end == start:
        return [f"{surah}:{start}"]
    return [f"{surah}:{v}" for v in range(start, end + 1)]


def parse_asbab_pdf() -> dict[str, str]:
    """Parse the Asbab al-Nuzul PDF into verse_id → context text mapping.

    Returns dict mapping verse_id → asbab narration text.
    """
    if not ASBAB_PDF.exists():
        logger.warning(
            f"Asbab al-Nuzul PDF not found: {ASBAB_PDF}\n"
            "[MANUAL] Please download it from "
            "https://www.altafsir.com/Books/Asbab%20Al-Nuzul%20by%20Al-Wahidi.pdf "
            "and place it in data/raw/asbab_al_nuzul/"
        )
        return {}

    # Check for cached parsed version
    if ASBAB_PARSED.exists():
        logger.info("Using cached asbab parsed data")
        return json.loads(ASBAB_PARSED.read_text(encoding="utf-8"))

    import pdfplumber

    logger.info(f"Parsing Asbab al-Nuzul from {ASBAB_PDF}")

    # Extract full text from PDF
    full_text = ""
    with pdfplumber.open(ASBAB_PDF) as pdf:
        for page in tqdm(pdf.pages, desc="Extracting PDF pages"):
            text = page.extract_text()
            if text:
                full_text += text + "\n"

    logger.info(f"Extracted {len(full_text):,} characters from PDF")

    # Strategy: split the text into sections using [surah:verse] headers as delimiters.
    # Each section's text = the asbab narration for that verse (or verse range).
    #
    # The PDF format is:
    #   [2:14]
    #   (And when they fall in with those who believe...) [2:14]. Ahmad ibn Muhammad...
    #   ... narration text ...
    #
    #   [2:21]
    #   (O mankind! Worship your Lord ...) [2:21]. Sa'id ibn Muhammad...
    #   ... narration text ...

    asbab_entries: dict[str, str] = {}

    # Find all section header positions
    headers = list(SECTION_HEADER_RE.finditer(full_text))
    logger.info(f"Found {len(headers)} verse section headers in PDF")

    for i, match in enumerate(headers):
        surah = int(match.group(1))
        start_verse = int(match.group(2))
        end_verse = int(match.group(3)) if match.group(3) else None

        # Extract text from after this header to before the next header
        text_start = match.end()
        text_end = headers[i + 1].start() if i + 1 < len(headers) else len(full_text)

        section_text = full_text[text_start:text_end].strip()

        # Clean up: remove excessive whitespace, keep paragraph structure
        section_text = re.sub(r"\n{3,}", "\n\n", section_text)
        section_text = section_text.strip()

        if not section_text:
            continue

        # Assign to all verses in the range
        verse_ids = expand_verse_refs(surah, start_verse, end_verse)
        for verse_id in verse_ids:
            # Don't overwrite if we already have a more specific entry
            if verse_id not in asbab_entries:
                asbab_entries[verse_id] = section_text

    # Cache
    RAW_ASBAB_DIR.mkdir(parents=True, exist_ok=True)
    ASBAB_PARSED.write_text(
        json.dumps(asbab_entries, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info(f"Parsed asbab al-nuzul for {len(asbab_entries)} verse(s)")
    return asbab_entries


def run():
    """Entry point."""
    return parse_asbab_pdf()


if __name__ == "__main__":
    run()

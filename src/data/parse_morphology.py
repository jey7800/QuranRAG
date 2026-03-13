"""Parse Quranic Arabic Corpus morphological data (TSV) into per-verse annotations."""

import json
import re
from collections import defaultdict

from loguru import logger

from src.config import MORPHOLOGY_FILE, RAW_CORPUS_DIR


# Regex for the location field: (surah:ayah:word:segment)
LOCATION_RE = re.compile(r"\((\d+):(\d+):(\d+):(\d+)\)")


def parse_morphology_tsv() -> dict[str, list[dict]]:
    """Parse the Quranic Arabic Corpus TSV file.

    Returns a dict mapping verse_id → list of word-level morphology entries.
    """
    if not MORPHOLOGY_FILE.exists():
        logger.warning(
            f"Morphology file not found: {MORPHOLOGY_FILE}\n"
            "[MANUAL] Please download it from https://corpus.quran.com/download/ "
            "and place it in data/raw/quranic_corpus/"
        )
        return {}

    logger.info(f"Parsing morphology from {MORPHOLOGY_FILE}")

    # Group segments by (surah, ayah, word)
    word_segments: dict[tuple, list[dict]] = defaultdict(list)

    with open(MORPHOLOGY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 4:
                continue

            location, form, tag, features = parts[0], parts[1], parts[2], parts[3]
            match = LOCATION_RE.match(location)
            if not match:
                continue

            surah, ayah, word, segment = (
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
                int(match.group(4)),
            )

            # Determine segment type
            if "PREFIX" in features or features.startswith("PREFIX"):
                seg_type = "PREFIX"
            elif "SUFFIX" in features or features.startswith("SUFFIX"):
                seg_type = "SUFFIX"
            else:
                seg_type = "STEM"

            seg_entry = {
                "form": form,
                "tag": tag,
                "type": seg_type,
            }

            word_segments[(surah, ayah, word)].append((segment, seg_entry, features))

    # Build per-verse morphology
    verse_morphology: dict[str, list[dict]] = defaultdict(list)

    # Group by (surah, ayah) and then by word
    words_by_verse: dict[tuple, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for (surah, ayah, word), segs in word_segments.items():
        words_by_verse[(surah, ayah)][word] = sorted(segs, key=lambda x: x[0])

    for (surah, ayah), words in words_by_verse.items():
        verse_id = f"{surah}:{ayah}"
        for word_idx in sorted(words.keys()):
            segments = words[word_idx]

            # Reconstruct the full Arabic word from segments
            arabic_word = "".join(s[1]["form"] for s in segments)

            # Extract root, lemma, POS from STEM segment
            root = None
            lemma = None
            pos = "UNK"
            features_str = None

            for _, seg_entry, features in segments:
                if seg_entry["type"] == "STEM":
                    pos = seg_entry["tag"]
                    # Parse features like ROOT:سمو|LEM:ٱسْم|M|GEN
                    for feat in features.split("|"):
                        if feat.startswith("ROOT:"):
                            root = feat[5:]
                        elif feat.startswith("LEM:"):
                            lemma = feat[4:]
                    # Collect remaining features
                    remaining = [
                        f
                        for f in features.split("|")
                        if not f.startswith(("ROOT:", "LEM:", "POS:", "STEM"))
                    ]
                    features_str = "|".join(remaining) if remaining else None

            word_entry = {
                "word_index": word_idx,
                "arabic": arabic_word,
                "root": root,
                "lemma": lemma,
                "pos": pos,
                "features": features_str,
                "segments": [s[1] for s in segments],
            }
            verse_morphology[verse_id].append(word_entry)

    logger.info(f"Parsed morphology for {len(verse_morphology)} verses")

    # Cache parsed data
    cache_path = RAW_CORPUS_DIR / "morphology_parsed.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(verse_morphology, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return verse_morphology


def run():
    """Entry point."""
    return parse_morphology_tsv()


if __name__ == "__main__":
    run()

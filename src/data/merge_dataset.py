"""Merge all 4 layers into the final verses.jsonl dataset."""

import json

from loguru import logger

from src.config import (
    ABROGATION_FILE,
    CHAPTERS_JSON,
    CONCEPTS_JSON,
    POLYSEMY_CATALOG,
    PROCESSED_DIR,
    SURAH_TO_REVELATION_ORDER,
    VERSES_JSONL,
)


def build_verse_to_topics(concepts: list[dict]) -> dict[str, list[str]]:
    """Build a mapping from verse_id → list of topic tags from ontology concepts."""
    verse_topics: dict[str, list[str]] = {}
    for concept in concepts:
        for verse_id in concept.get("verses", []):
            if verse_id not in verse_topics:
                verse_topics[verse_id] = []
            verse_topics[verse_id].append(concept["concept_id"])
    return verse_topics


def build_verse_to_related(concepts: list[dict]) -> dict[str, list[str]]:
    """Build a mapping from verse_id → related verse_ids (same concept cluster)."""
    verse_related: dict[str, set[str]] = {}
    for concept in concepts:
        verses = concept.get("verses", [])
        for v in verses:
            if v not in verse_related:
                verse_related[v] = set()
            verse_related[v].update(vid for vid in verses if vid != v)
    return {k: list(v)[:10] for k, v in verse_related.items()}  # cap at 10


def load_polysemy_catalog() -> dict[str, dict]:
    """Load the polysemy catalog keyed by Arabic word."""
    if not POLYSEMY_CATALOG.exists():
        logger.debug("No polysemy catalog found — skipping polysemy layer")
        return {}
    return json.loads(POLYSEMY_CATALOG.read_text(encoding="utf-8"))


def load_abrogation_data() -> dict[str, dict]:
    """Load abrogation data keyed by verse_id."""
    if not ABROGATION_FILE.exists():
        logger.debug("No abrogation file found — skipping abrogation layer")
        return {}
    raw = json.loads(ABROGATION_FILE.read_text(encoding="utf-8"))
    result = {}
    for entry in raw:
        abrogated = entry.get("abrogated_verse")
        abrogating = entry.get("abrogating_verse")
        if abrogated:
            result[abrogated] = {
                "abrogated_by": abrogating,
                "abrogates": None,
                "topic": entry.get("topic", ""),
                "scholarly_consensus": entry.get("scholarly_consensus", "debated"),
                "note": entry.get("note"),
            }
        if abrogating:
            result.setdefault(abrogating, {
                "abrogated_by": None,
                "abrogates": abrogated,
                "topic": entry.get("topic", ""),
                "scholarly_consensus": entry.get("scholarly_consensus", "debated"),
                "note": entry.get("note"),
            })
    return result


def merge_dataset(
    chapters: list[dict],
    verses: list[dict],
    transliterations: dict[str, str],
    morphology: dict[str, list[dict]],
    asbab: dict[str, str],
    concepts: list[dict],
) -> list[dict]:
    """Merge all layers into the final verse dataset."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    verse_topics = build_verse_to_topics(concepts)
    verse_related = build_verse_to_related(concepts)
    polysemy = load_polysemy_catalog()
    abrogation = load_abrogation_data()

    merged_verses = []
    warnings = []

    for v in verses:
        verse_id = v["verse_id"]
        surah_num = v["surah_number"]

        # Layer 1: Text (already present)
        merged = {
            "verse_id": verse_id,
            "surah_number": surah_num,
            "ayah_number": v["ayah_number"],
            "text_arabic": v["text_arabic"],
            "text_en_asad": v["text_en_asad"],
            "text_fr_hamidullah": v.get("text_fr_hamidullah"),
            "transliteration": transliterations.get(verse_id),
        }

        # Layer 2: Linguistics
        morph = morphology.get(verse_id, [])
        merged["morphology"] = morph

        # Detect polysemous words in this verse
        poly_words = []
        for word in morph:
            arabic = word.get("arabic", "")
            if arabic in polysemy:
                poly_words.append(polysemy[arabic])
        merged["polysemous_words"] = poly_words

        # Layer 3: Historical context
        asbab_text = asbab.get(verse_id)
        merged["asbab_al_nuzul"] = asbab_text
        merged["asbab_status"] = "documented" if asbab_text else "not_documented"
        merged["revelation_period"] = v.get("revelation_type", "unknown")
        merged["revelation_order"] = SURAH_TO_REVELATION_ORDER.get(surah_num, 0)

        # Layer 4: Thematic links
        merged["topic_tags"] = verse_topics.get(verse_id, [])
        merged["related_verses"] = verse_related.get(verse_id, [])
        merged["abrogation"] = abrogation.get(verse_id)

        # Metadata
        merged["surah_name_ar"] = v.get("surah_name_ar", "")
        merged["surah_name_en"] = v.get("surah_name_en", "")
        merged["juz"] = v.get("juz", 0)
        merged["hizb"] = v.get("hizb", 0)
        merged["page"] = v.get("page")

        merged_verses.append(merged)

    # Write verses.jsonl
    with open(VERSES_JSONL, "w", encoding="utf-8") as f:
        for verse in merged_verses:
            f.write(json.dumps(verse, ensure_ascii=False) + "\n")

    # Write chapters.json
    enriched_chapters = []
    for ch in chapters:
        surah_num = ch["surah_number"]
        ch_verses = [v["verse_id"] for v in merged_verses if v["surah_number"] == surah_num]
        enriched_chapters.append({
            "surah_number": surah_num,
            "name_ar": ch["name_ar"],
            "name_en": ch["name_en"],
            "revelation_type": ch["revelation_type"],
            "revelation_order": SURAH_TO_REVELATION_ORDER.get(surah_num, 0),
            "number_of_ayahs": ch["number_of_ayahs"],
            "verse_ids": ch_verses,
        })
    CHAPTERS_JSON.write_text(
        json.dumps(enriched_chapters, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Write concepts.json
    CONCEPTS_JSON.write_text(
        json.dumps(concepts, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info(
        f"Merged dataset: {len(merged_verses)} verses, {len(enriched_chapters)} chapters, "
        f"{len(concepts)} concepts"
    )
    if warnings:
        for w in warnings:
            logger.warning(w)

    return merged_verses

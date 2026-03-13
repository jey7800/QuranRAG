"""In-memory store for all Quranic data. Loaded once at startup."""

import json

from loguru import logger

from src.config import (
    CHAPTERS_JSON,
    CONCEPTS_JSON,
    POLYSEMY_CATALOG,
    ABROGATION_FILE,
    VERSES_JSONL,
)
from src.data.schemas import (
    AbrogationInfo,
    Chapter,
    OntologyConcept,
    PolysemyEntry,
    Verse,
)


class DataStore:
    """Holds all static Quranic data in memory for fast O(1) lookups."""

    def __init__(self) -> None:
        self._verses: dict[str, Verse] = {}
        self._chapters: dict[int, Chapter] = {}
        self._concepts: dict[str, OntologyConcept] = {}
        self._polysemy: dict[str, PolysemyEntry] = {}  # keyed by root
        self._abrogation: dict[str, AbrogationInfo] = {}  # keyed by verse_id

    def load(self) -> None:
        """Load all data files into memory. Call once at startup."""
        self._load_verses()
        self._load_chapters()
        self._load_concepts()
        self._load_polysemy()
        self._load_abrogation()
        logger.info(
            f"DataStore loaded: {len(self._verses)} verses, "
            f"{len(self._chapters)} chapters, {len(self._concepts)} concepts, "
            f"{len(self._polysemy)} polysemy entries, {len(self._abrogation)} abrogation entries"
        )

    def _load_verses(self) -> None:
        with open(VERSES_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                v = Verse.model_validate_json(line)
                self._verses[v.verse_id] = v

    def _load_chapters(self) -> None:
        with open(CHAPTERS_JSON, encoding="utf-8") as f:
            data = json.load(f)
        for ch in data:
            chapter = Chapter.model_validate(ch)
            self._chapters[chapter.surah_number] = chapter

    def _load_concepts(self) -> None:
        with open(CONCEPTS_JSON, encoding="utf-8") as f:
            data = json.load(f)
        for c in data:
            concept = OntologyConcept.model_validate(c)
            self._concepts[concept.concept_id] = concept

    def _load_polysemy(self) -> None:
        if not POLYSEMY_CATALOG.exists():
            logger.warning(f"Polysemy catalog not found: {POLYSEMY_CATALOG}")
            return
        with open(POLYSEMY_CATALOG, encoding="utf-8") as f:
            data = json.load(f)
        for _word, entry in data.items():
            pe = PolysemyEntry.model_validate(entry)
            self._polysemy[pe.root] = pe

    def _load_abrogation(self) -> None:
        if not ABROGATION_FILE.exists():
            logger.warning(f"Abrogation file not found: {ABROGATION_FILE}")
            return
        with open(ABROGATION_FILE, encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            abrogated = entry["abrogated_verse"]
            abrogating = entry["abrogating_verse"]
            base = {
                "topic": entry["topic"],
                "scholarly_consensus": entry["scholarly_consensus"],
                "note": entry.get("note"),
            }
            # The abrogated verse points to what abrogates it
            self._abrogation[abrogated] = AbrogationInfo(
                abrogated_by=abrogating, **base
            )
            # The abrogating verse points to what it abrogates
            self._abrogation[abrogating] = AbrogationInfo(
                abrogates=abrogated, **base
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def get_verse(self, verse_id: str) -> Verse | None:
        return self._verses.get(verse_id)

    def get_chapter(self, surah_number: int) -> Chapter | None:
        return self._chapters.get(surah_number)

    def get_concept(self, concept_id: str) -> OntologyConcept | None:
        return self._concepts.get(concept_id)

    def get_all_concepts(self) -> dict[str, OntologyConcept]:
        return self._concepts

    def get_verses_for_surah(self, surah_number: int) -> list[Verse]:
        chapter = self._chapters.get(surah_number)
        if not chapter:
            return []
        return [self._verses[vid] for vid in chapter.verse_ids if vid in self._verses]

    def get_neighbors(self, verse_id: str, range_: int = 2) -> list[Verse]:
        """Get ±range_ neighboring verses, excluding the verse itself."""
        parts = verse_id.split(":")
        if len(parts) != 2:
            return []
        surah_num, ayah_num = int(parts[0]), int(parts[1])
        chapter = self._chapters.get(surah_num)
        if not chapter:
            return []

        neighbors = []
        for offset in range(-range_, range_ + 1):
            if offset == 0:
                continue
            neighbor_ayah = ayah_num + offset
            if 1 <= neighbor_ayah <= chapter.number_of_ayahs:
                nid = f"{surah_num}:{neighbor_ayah}"
                v = self._verses.get(nid)
                if v:
                    neighbors.append(v)
        return neighbors

    def get_polysemy_for_verse(self, verse: Verse) -> list[PolysemyEntry]:
        """Find polysemy entries relevant to this verse by matching morphology roots."""
        if not self._polysemy:
            return []
        # First check if the verse already has polysemous_words populated
        if verse.polysemous_words:
            return verse.polysemous_words
        # Fall back to matching roots from morphology
        found = []
        verse_roots = {w.root for w in verse.morphology if w.root}
        for root, entry in self._polysemy.items():
            if root in verse_roots:
                found.append(entry)
        return found

    def get_abrogation(self, verse_id: str) -> AbrogationInfo | None:
        return self._abrogation.get(verse_id)

    def get_stats(self) -> dict:
        total = len(self._verses)
        if total == 0:
            return {"total_verses": 0, "total_surahs": 0}
        with_morphology = sum(1 for v in self._verses.values() if v.morphology)
        with_asbab = sum(
            1 for v in self._verses.values() if v.asbab_status == "documented"
        )
        with_tags = sum(1 for v in self._verses.values() if v.topic_tags)
        return {
            "total_verses": total,
            "total_surahs": len(self._chapters),
            "total_concepts": len(self._concepts),
            "layer1_coverage": 1.0,  # all verses have text
            "layer2_coverage": round(with_morphology / total, 3),
            "layer3_coverage": round(with_asbab / total, 3),
            "layer4_coverage": round(with_tags / total, 3),
            "polysemy_entries": len(self._polysemy),
            "abrogation_entries": len(self._abrogation) // 2,  # each pair counted once
        }

"""Generate 200 benchmark queries from the ontology and verse data.

Ground truth is derived from two independent sources:
1. concepts.json — concept.verses (scholarly ontology)
2. verses.jsonl — inverted topic_tags index

This dual-sourcing eliminates developer bias.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Any

from loguru import logger

from src.config import (
    BENCHMARK_SEED,
    CONCEPTS_JSON,
    VERSES_JSONL,
)

# ── Data structures ───────────────────────────────────────────────────────────


class BenchmarkQuery:
    """A single benchmark query with ground truth."""

    def __init__(
        self,
        *,
        id: str,
        query: str,
        expected_verses: list[str],
        category: str,
        language: str,
        difficulty: str,
        source_concept: str = "",
    ) -> None:
        self.id = id
        self.query = query
        self.expected_verses = expected_verses
        self.category = category
        self.language = language
        self.difficulty = difficulty
        self.source_concept = source_concept

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "expected_verses": self.expected_verses,
            "category": self.category,
            "language": self.language,
            "difficulty": self.difficulty,
            "source_concept": self.source_concept,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BenchmarkQuery:
        return cls(**d)


# ── French concept translations (top 30 Quranic concepts) ────────────────────

CONCEPT_FR_MAP: dict[str, str] = {
    "allah": "Allah",
    "paradise": "le paradis",
    "hell": "l'enfer",
    "satan": "Satan",
    "day-of-resurrection": "le jour de la résurrection",
    "last-day": "le jour dernier",
    "heart": "le cœur",
    "earth": "la terre",
    "musa": "Moïse",
    "ibrahim": "Abraham",
    "children-of-israel": "les enfants d'Israël",
    "pharaoh": "Pharaon",
    "quran": "le Coran",
    "islam": "l'islam",
    "nuh": "Noé",
    "maryam": "Marie",
    "sun": "le soleil",
    "jinn": "les djinns",
    "isa": "Jésus",
    "moon": "la lune",
    "fire": "le feu",
    "water": "l'eau",
    "angel": "les anges",
    "soul": "l'âme",
    "prayer": "la prière",
    "messenger": "le messager",
    "prophet": "le prophète",
    "ship": "le navire",
    "star": "les étoiles",
    "adam": "Adam",
    "david": "David",
    "solomon": "Salomon",
    "jacob": "Jacob",
    "joseph": "Joseph",
    "noah's-ark": "l'arche de Noé",
    "mountain": "la montagne",
    "rain": "la pluie",
    "sea": "la mer",
    "wind": "le vent",
    "night": "la nuit",
    "jesus": "Jésus",
    "torah": "la Torah",
    "tree": "l'arbre",
    "muhammad": "Muhammad",
    "christianity": "le christianisme",
    "judaism": "le judaïsme",
    "kaaba": "la Kaaba",
    "lightning": "la foudre",
    "cloud": "les nuages",
    "gold": "l'or",
    "camel": "le chameau",
    "clay": "l'argile",
    "allah's-throne": "le trône d'Allah",
    "iblis": "Iblis",
    "children-of-adam": "les enfants d'Adam",
    "garden-of-eden": "le jardin d'Éden",
    "king": "le roi",
    "israel": "Israël",
    "ishmael": "Ismaël",
    "dawn": "l'aube",
    "dust": "la poussière",
    "tongue": "la langue",
    "snake": "le serpent",
    "fish": "le poisson",
    "silk": "la soie",
    "bone": "les os",
    "pearl": "la perle",
    "grain": "le grain",
    "night-of-decree": "la nuit du destin",
    "masjid-al-haram": "la mosquée sacrée",
}

# ── Paraphrase queries (hand-crafted indirect reformulations) ────────────────

PARAPHRASE_QUERIES: list[dict[str, Any]] = [
    {
        "query": "What happens to people after they die according to the Quran?",
        "concept": "day-of-resurrection",
    },
    {
        "query": "How should wealth be shared among family members?",
        "concept": "inheritance",
    },
    {
        "query": "What rewards await the righteous in the hereafter?",
        "concept": "paradise",
    },
    {
        "query": "Who was the prophet that built the ark?",
        "concept": "nuh",
    },
    {
        "query": "What are the consequences of wrongdoing in the Quran?",
        "concept": "hell",
    },
    {
        "query": "How does the Quran describe the creation of the universe?",
        "concept": "earth",
    },
    {
        "query": "What role do supernatural beings play besides angels?",
        "concept": "jinn",
    },
    {
        "query": "Which woman is most revered in the Quran?",
        "concept": "maryam",
    },
    {
        "query": "What is the story of the golden calf worship?",
        "concept": "children-of-israel",
    },
    {
        "query": "How does the Quran describe the enemy of mankind?",
        "concept": "satan",
    },
    {
        "query": "What celestial bodies does the Quran mention?",
        "concept": "sun",
    },
    {
        "query": "Who escaped from the tyrant king of Egypt?",
        "concept": "musa",
    },
    {
        "query": "What does the Quran say about the inner spiritual state of humans?",
        "concept": "heart",
    },
    {
        "query": "Which prophet was willing to sacrifice his son?",
        "concept": "ibrahim",
    },
    {
        "query": "What is the Quran's view on lending money at interest?",
        "concept": "usury",
    },
]

# ── Negative queries (topics NOT in the Quran) ──────────────────────────────

NEGATIVE_QUERIES: list[str] = [
    "What does the Quran say about nuclear energy?",
    "Quran verses about the internet and social media",
    "What does the Quran say about democracy and voting?",
    "Quran teachings about cryptocurrency and bitcoin",
    "What does the Quran say about space travel?",
    "Quran verses mentioning dinosaurs",
    "What does the Quran say about artificial intelligence?",
    "Quran guidance on climate change policy",
    "What does the Quran say about television?",
    "Quran verses about vaccination",
    "What does the Quran say about communism?",
    "Quran teachings about stock markets",
    "What does the Quran say about plastic pollution?",
    "Quran verses about quantum physics",
    "What does the Quran say about genetic engineering?",
]


# ── Main generator ────────────────────────────────────────────────────────────


def _build_inverted_tag_index(verses: list[dict]) -> dict[str, set[str]]:
    """Build tag -> set of verse_ids from verses.jsonl data."""
    index: dict[str, set[str]] = defaultdict(set)
    for v in verses:
        for tag in v.get("topic_tags", []):
            index[tag].add(v["verse_id"])
    return index


def _build_ground_truth(
    concepts: list[dict], tag_index: dict[str, set[str]], min_verses: int = 5
) -> dict[str, list[str]]:
    """Merge concept.verses with inverted tag index for robust ground truth."""
    ground_truth: dict[str, list[str]] = {}
    for concept in concepts:
        cid = concept["concept_id"]
        gt = set(concept.get("verses", []))
        gt.update(tag_index.get(cid, set()))
        if len(gt) >= min_verses:
            ground_truth[cid] = sorted(gt, key=_verse_sort_key)
    return ground_truth


def _verse_sort_key(verse_id: str) -> tuple[int, int]:
    """Sort key for verse IDs like '2:255'."""
    parts = verse_id.split(":")
    return (int(parts[0]), int(parts[1]))


def _extract_distinctive_phrase(text: str, min_words: int = 5, max_words: int = 10) -> str:
    """Extract a distinctive phrase from English translation text."""
    # Remove parenthetical notes and brackets
    cleaned = re.sub(r"\([^)]*\)", "", text)
    cleaned = re.sub(r"\[[^\]]*\]", "", cleaned)
    words = cleaned.split()
    if len(words) < min_words:
        return " ".join(words)
    # Take a window from the middle for distinctiveness
    start = max(0, len(words) // 2 - max_words // 2)
    end = min(len(words), start + max_words)
    return " ".join(words[start:end]).strip(" ,;:.")


def _extract_arabic_phrase(text: str, max_words: int = 5) -> str:
    """Extract a short Arabic phrase from verse text."""
    words = text.split()
    if len(words) <= max_words:
        return text
    # Take from the beginning (more distinctive in Arabic)
    return " ".join(words[:max_words])


def _concept_name_to_query(name_en: str) -> str:
    """Convert concept name_en to a natural English query."""
    name = name_en.lower().replace("'s ", "'s ").replace("-", " ")
    return f"What does the Quran say about {name}?"


def generate_queries(
    concepts_path: Path = CONCEPTS_JSON,
    verses_path: Path = VERSES_JSONL,
    seed: int = BENCHMARK_SEED,
) -> list[BenchmarkQuery]:
    """Generate the full benchmark query set (target: ~200 queries).

    The generation is deterministic given the seed.
    """
    rng = Random(seed)

    # Load data
    with open(concepts_path, encoding="utf-8") as f:
        concepts = json.load(f)
    with open(verses_path, encoding="utf-8") as f:
        verses = [json.loads(line) for line in f]

    # Build ground truth
    tag_index = _build_inverted_tag_index(verses)
    ground_truth = _build_ground_truth(concepts, tag_index, min_verses=5)
    concept_map = {c["concept_id"]: c for c in concepts}

    logger.info(f"Ground truth built: {len(ground_truth)} concepts with ≥5 verses")

    queries: list[BenchmarkQuery] = []
    idx = 0

    # ── 1. Concept queries (English) — 60 ────────────────────────────────────
    eligible = sorted(ground_truth.keys())
    # Exclude overly broad concepts (>200 verses) for meaningful recall
    eligible = [c for c in eligible if len(ground_truth[c]) <= 200]
    selected_concepts = rng.sample(eligible, min(70, len(eligible)))

    for cid in selected_concepts[:60]:
        concept = concept_map[cid]
        queries.append(BenchmarkQuery(
            id=f"concept_en_{idx:03d}",
            query=_concept_name_to_query(concept["name_en"]),
            expected_verses=ground_truth[cid],
            category="concept_en",
            language="en",
            difficulty="medium",
            source_concept=cid,
        ))
        idx += 1

    # ── 2. Concept queries (French) — 30 ─────────────────────────────────────
    fr_eligible = [c for c in selected_concepts if c in CONCEPT_FR_MAP]
    rng.shuffle(fr_eligible)
    for cid in fr_eligible[:30]:
        fr_name = CONCEPT_FR_MAP[cid]
        queries.append(BenchmarkQuery(
            id=f"concept_fr_{idx:03d}",
            query=f"Que dit le Coran sur {fr_name}\u202f?",
            expected_verses=ground_truth[cid],
            category="concept_fr",
            language="fr",
            difficulty="medium",
            source_concept=cid,
        ))
        idx += 1

    # ── 3. Concept queries (Arabic keyword) — 20 ─────────────────────────────
    # Use the Arabic text of a tagged verse as the query keyword
    ar_eligible = [c for c in selected_concepts if c in ground_truth]
    rng.shuffle(ar_eligible)
    verse_by_id = {v["verse_id"]: v for v in verses}
    ar_count = 0
    for cid in ar_eligible:
        if ar_count >= 20:
            break
        gt_verses = ground_truth[cid]
        # Pick a random verse from this concept's ground truth
        sample_vid = rng.choice(gt_verses)
        verse_data = verse_by_id.get(sample_vid)
        if not verse_data:
            continue
        ar_phrase = _extract_arabic_phrase(verse_data["text_arabic"])
        if len(ar_phrase) < 5:
            continue
        queries.append(BenchmarkQuery(
            id=f"concept_ar_{idx:03d}",
            query=ar_phrase,
            expected_verses=gt_verses,
            category="concept_ar",
            language="ar",
            difficulty="medium",
            source_concept=cid,
        ))
        idx += 1
        ar_count += 1

    # ── 4. Verse phrase queries (English) — 25 ───────────────────────────────
    tagged_verses = [v for v in verses if v.get("topic_tags") and len(v["text_en_asad"]) > 50]
    sampled_verses_en = rng.sample(tagged_verses, min(30, len(tagged_verses)))
    en_phrase_count = 0
    for v in sampled_verses_en:
        if en_phrase_count >= 25:
            break
        phrase = _extract_distinctive_phrase(v["text_en_asad"])
        if len(phrase.split()) < 4:
            continue
        queries.append(BenchmarkQuery(
            id=f"verse_phrase_en_{idx:03d}",
            query=phrase,
            expected_verses=[v["verse_id"]],
            category="verse_phrase_en",
            language="en",
            difficulty="easy",
        ))
        idx += 1
        en_phrase_count += 1

    # ── 5. Verse phrase queries (Arabic) — 15 ────────────────────────────────
    sampled_verses_ar = rng.sample(tagged_verses, min(20, len(tagged_verses)))
    ar_phrase_count = 0
    for v in sampled_verses_ar:
        if ar_phrase_count >= 15:
            break
        phrase = _extract_arabic_phrase(v["text_arabic"])
        if len(phrase.split()) < 3:
            continue
        queries.append(BenchmarkQuery(
            id=f"verse_phrase_ar_{idx:03d}",
            query=phrase,
            expected_verses=[v["verse_id"]],
            category="verse_phrase_ar",
            language="ar",
            difficulty="easy",
        ))
        idx += 1
        ar_phrase_count += 1

    # ── 6. Cross-reference queries — 20 ──────────────────────────────────────
    verses_with_refs = [
        v for v in verses
        if len(v.get("related_verses", [])) >= 3 and len(v["text_en_asad"]) > 50
    ]
    sampled_xref = rng.sample(verses_with_refs, min(25, len(verses_with_refs)))
    xref_count = 0
    for v in sampled_xref:
        if xref_count >= 20:
            break
        phrase = _extract_distinctive_phrase(v["text_en_asad"], min_words=6, max_words=12)
        if len(phrase.split()) < 5:
            continue
        expected = [v["verse_id"]] + v["related_verses"][:5]
        queries.append(BenchmarkQuery(
            id=f"cross_ref_{idx:03d}",
            query=phrase,
            expected_verses=expected,
            category="cross_reference",
            language="en",
            difficulty="hard",
        ))
        idx += 1
        xref_count += 1

    # ── 7. Negative queries — 15 ─────────────────────────────────────────────
    for neg_query in NEGATIVE_QUERIES:
        queries.append(BenchmarkQuery(
            id=f"negative_{idx:03d}",
            query=neg_query,
            expected_verses=[],
            category="negative",
            language="en",
            difficulty="negative",
        ))
        idx += 1

    # ── 8. Paraphrase queries — 15 ───────────────────────────────────────────
    for pq in PARAPHRASE_QUERIES:
        cid = pq["concept"]
        gt = ground_truth.get(cid, [])
        if not gt:
            # Fallback: try tag index only
            gt = sorted(tag_index.get(cid, set()), key=_verse_sort_key)
        queries.append(BenchmarkQuery(
            id=f"paraphrase_{idx:03d}",
            query=pq["query"],
            expected_verses=gt,
            category="paraphrase",
            language="en",
            difficulty="hard",
            source_concept=cid,
        ))
        idx += 1

    logger.info(
        f"Generated {len(queries)} benchmark queries: "
        + ", ".join(
            f"{cat}={sum(1 for q in queries if q.category == cat)}"
            for cat in sorted({q.category for q in queries})
        )
    )
    return queries


def save_queries(queries: list[BenchmarkQuery], path: Path) -> None:
    """Save query set to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": "2.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": BENCHMARK_SEED,
        "n_queries": len(queries),
        "queries": [q.to_dict() for q in queries],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(queries)} queries to {path}")


def load_queries(path: Path) -> list[BenchmarkQuery]:
    """Load query set from JSON."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    queries = [BenchmarkQuery.from_dict(q) for q in data["queries"]]
    logger.info(f"Loaded {len(queries)} queries from {path} (v{data['version']})")
    return queries

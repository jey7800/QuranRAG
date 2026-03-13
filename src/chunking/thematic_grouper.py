"""Level 2: Cluster consecutive verses by semantic similarity.

Groups consecutive verses within each surah into thematic chunks.
The algorithm:
  1. Compute cosine similarity between consecutive verse embeddings
  2. Cut where similarity < threshold
  3. Post-process: merge small groups, split large groups
  4. Build ThematicGroupChunk for each group

This preserves the Quran's sequential structure (no cross-surah groups,
no reordering) while capturing thematic continuity.
"""

from collections import Counter

import numpy as np
from loguru import logger
from tqdm import tqdm

from src.chunking.schemas import ThematicGroupChunk
from src.chunking.verse_chunker import build_verse_text
from src.config import (
    DEFAULT_CHUNK_SIMILARITY_THRESHOLD,
    MAX_GROUP_SIZE,
    MIN_GROUP_SIZE,
)
from src.data.schemas import Verse
from src.embedding.base import Embedder


def compute_verse_embeddings(
    verses: list[Verse],
    embedder: Embedder,
) -> np.ndarray:
    """Embed all verses and return an (N, D) matrix.

    Uses the same build_verse_text as Level 1 chunking so that
    the embeddings are identical and can be reused.

    Args:
        verses: The verse objects.
        embedder: An initialized Embedder instance.

    Returns:
        numpy array of shape (len(verses), embedding_dim).
    """
    texts = [build_verse_text(v) for v in verses]
    return embedder.embed_texts(texts)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _centroid(embeddings: np.ndarray, indices: list[int]) -> np.ndarray:
    """Compute the centroid of a subset of embeddings."""
    return embeddings[indices].mean(axis=0)


def group_consecutive_verses(
    embeddings: np.ndarray,
    similarity_threshold: float = DEFAULT_CHUNK_SIMILARITY_THRESHOLD,
    min_group: int = MIN_GROUP_SIZE,
    max_group: int = MAX_GROUP_SIZE,
) -> list[list[int]]:
    """Segment a surah's verses into thematic groups.

    Args:
        embeddings: Pre-computed embeddings for the surah's verses, shape (N, D).
        similarity_threshold: Minimum cosine similarity to keep verses in the same group.
        min_group: Minimum number of verses per group.
        max_group: Maximum number of verses per group.

    Returns:
        List of groups, each a list of 0-based verse indices within the surah.
    """
    n = len(embeddings)

    # Edge case: surah too small for grouping
    if n <= min_group:
        return [list(range(n))]

    # Step 1: Compute consecutive similarities
    similarities = []
    for i in range(n - 1):
        sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
        similarities.append(sim)

    # Step 2: Find break points where similarity drops below threshold
    breaks = set()
    for i, sim in enumerate(similarities):
        if sim < similarity_threshold:
            breaks.add(i + 1)  # break BEFORE index i+1

    # Step 3: Build initial groups from break points
    groups: list[list[int]] = []
    current_group: list[int] = [0]

    for i in range(1, n):
        if i in breaks:
            groups.append(current_group)
            current_group = [i]
        else:
            current_group.append(i)
    groups.append(current_group)

    # Step 4: Merge groups that are too small
    groups = _merge_small_groups(groups, embeddings, min_group)

    # Step 5: Split groups that are too large
    groups = _split_large_groups(groups, embeddings, similarities, max_group)

    return groups


def _merge_small_groups(
    groups: list[list[int]],
    embeddings: np.ndarray,
    min_group: int,
) -> list[list[int]]:
    """Merge groups smaller than min_group with their most similar neighbor."""
    if len(groups) <= 1:
        return groups

    merged = True
    while merged:
        merged = False
        new_groups: list[list[int]] = []
        i = 0
        while i < len(groups):
            if len(groups[i]) < min_group and len(groups) > 1:
                # Find most similar neighbor (previous or next)
                best_merge = None
                best_sim = -1.0

                centroid_i = _centroid(embeddings, groups[i])

                if i > 0 and (not new_groups or len(new_groups[-1]) + len(groups[i]) <= 15):
                    prev_idx = len(new_groups) - 1
                    if prev_idx >= 0:
                        centroid_prev = _centroid(embeddings, new_groups[prev_idx])
                        sim = _cosine_similarity(centroid_i, centroid_prev)
                        if sim > best_sim:
                            best_sim = sim
                            best_merge = "prev"

                if i + 1 < len(groups):
                    centroid_next = _centroid(embeddings, groups[i + 1])
                    sim = _cosine_similarity(centroid_i, centroid_next)
                    if sim > best_sim:
                        best_sim = sim
                        best_merge = "next"

                if best_merge == "prev" and new_groups:
                    new_groups[-1].extend(groups[i])
                    merged = True
                elif best_merge == "next" and i + 1 < len(groups):
                    groups[i + 1] = groups[i] + groups[i + 1]
                    merged = True
                else:
                    new_groups.append(groups[i])
                i += 1
            else:
                new_groups.append(groups[i])
                i += 1
        groups = new_groups

    return groups


def _split_large_groups(
    groups: list[list[int]],
    embeddings: np.ndarray,
    similarities: list[float],
    max_group: int,
) -> list[list[int]]:
    """Split groups larger than max_group at the point of lowest internal similarity."""
    result: list[list[int]] = []

    for group in groups:
        if len(group) <= max_group:
            result.append(group)
            continue

        # Recursively split
        to_split = [group]
        while to_split:
            current = to_split.pop()
            if len(current) <= max_group:
                result.append(current)
                continue

            # Find the split point: lowest consecutive similarity within the group
            min_sim = float("inf")
            split_at = len(current) // 2  # fallback

            for j in range(len(current) - 1):
                idx = current[j]
                if idx < len(similarities):
                    sim = similarities[idx]
                    if sim < min_sim:
                        min_sim = sim
                        split_at = j + 1

            left = current[:split_at]
            right = current[split_at:]

            # If split didn't help, force a midpoint split
            if not left or not right:
                mid = len(current) // 2
                left = current[:mid]
                right = current[mid:]

            to_split.append(right)
            to_split.append(left)

    return result


def build_thematic_group_text(
    surah_number: int,
    surah_name_en: str,
    revelation_period: str,
    group_verses: list[Verse],
) -> str:
    """Build the embedding text for a thematic group.

    Format:
        [Surah Al-Baqara (2), Verses 1-5 | Medinan | Juz 1]

        Verse 2:1: <english>
        Verse 2:2: <english>
        ...
        Verse 2:5: <english>

        Arabic:
        <verse 1 arabic> <verse 2 arabic> ... <verse 5 arabic>

        Topics: <union of topic tags>
    """
    parts = []

    first = group_verses[0]
    last = group_verses[-1]
    period = revelation_period.capitalize()

    header = (
        f"[Surah {surah_name_en} ({surah_number}), "
        f"Verses {first.ayah_number}-{last.ayah_number} | {period} | Juz {first.juz}]"
    )
    parts.append(header)

    # English translations
    parts.append("")
    for v in group_verses:
        parts.append(f"Verse {v.verse_id}: {v.text_en_asad}")

    # Arabic text (continuous)
    parts.append("\nArabic:")
    arabic_texts = " ".join(v.text_arabic for v in group_verses)
    parts.append(arabic_texts)

    # Union of topic tags
    all_tags = sorted({tag for v in group_verses for tag in v.topic_tags})
    if all_tags:
        parts.append(f"\nTopics: {', '.join(all_tags)}")

    return "\n".join(parts)


def create_thematic_groups(
    verses_by_surah: dict[int, list[Verse]],
    embedder: Embedder,
) -> tuple[list[ThematicGroupChunk], dict[int, np.ndarray]]:
    """Build all thematic group chunks across all surahs.

    Process:
        1. Group verses by surah
        2. For each surah, compute verse embeddings
        3. Run the grouping algorithm
        4. Build ThematicGroupChunk for each group

    Surahs with fewer than min_group verses become a single group.

    Args:
        verses_by_surah: Dict mapping surah_number -> sorted list of Verse objects.
        embedder: Embedder instance for computing verse similarities.

    Returns:
        Tuple of:
            - List of ThematicGroupChunk objects
            - Dict mapping surah_number -> embeddings array (for reuse by verse embedding)
    """
    chunks: list[ThematicGroupChunk] = []
    all_embeddings: dict[int, np.ndarray] = {}

    for surah_num in tqdm(sorted(verses_by_surah.keys()), desc="Thematic grouping"):
        surah_verses = verses_by_surah[surah_num]
        if not surah_verses:
            continue

        # Compute embeddings for this surah's verses
        surah_embeddings = compute_verse_embeddings(surah_verses, embedder)
        all_embeddings[surah_num] = surah_embeddings

        # Run grouping algorithm
        groups = group_consecutive_verses(surah_embeddings)

        # Build chunks for each group
        first_verse = surah_verses[0]
        for group_indices in groups:
            group_verses = [surah_verses[i] for i in group_indices]
            start_ayah = group_verses[0].ayah_number
            end_ayah = group_verses[-1].ayah_number

            text = build_thematic_group_text(
                surah_number=surah_num,
                surah_name_en=first_verse.surah_name_en,
                revelation_period=first_verse.revelation_period,
                group_verses=group_verses,
            )

            all_tags = sorted({tag for v in group_verses for tag in v.topic_tags})

            chunk = ThematicGroupChunk(
                chunk_id=f"group:{surah_num}:{start_ayah}-{end_ayah}",
                surah_number=surah_num,
                surah_name_en=first_verse.surah_name_en,
                surah_name_ar=first_verse.surah_name_ar,
                start_ayah=start_ayah,
                end_ayah=end_ayah,
                verse_ids=[v.verse_id for v in group_verses],
                text_for_embedding=text,
                revelation_period=first_verse.revelation_period,
                juz=group_verses[0].juz,
                topic_tags=all_tags,
                verse_count=len(group_verses),
            )
            chunks.append(chunk)

    logger.info(f"Created {len(chunks)} thematic group chunks across {len(all_embeddings)} surahs")
    return chunks, all_embeddings

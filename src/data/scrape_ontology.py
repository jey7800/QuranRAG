"""Fetch Quranic ontology data from GitHub (seelenbrecher/islamic-agent)."""

import json

import httpx
from loguru import logger

from src.config import ONTOLOGY_GITHUB_BASE, RAW_ONTOLOGY_DIR


# Known files in the GitHub repo that contain ontology/concept data
ONTOLOGY_FILES = [
    "quran_ontology.json",
    "quran_concepts.json",
    "quran_topics.json",
]


async def _fetch_file(client: httpx.AsyncClient, filename: str) -> dict | list | None:
    """Try to fetch a JSON file from the GitHub repo."""
    url = f"{ONTOLOGY_GITHUB_BASE}/{filename}"
    try:
        resp = await client.get(url, timeout=30, follow_redirects=True)
        if resp.status_code == 200:
            return resp.json()
        logger.debug(f"File not found (HTTP {resp.status_code}): {url}")
    except Exception as e:
        logger.debug(f"Failed to fetch {url}: {e}")
    return None


async def fetch_ontology_from_github() -> list[dict]:
    """Fetch ontology data from GitHub and return list of concept dicts."""
    RAW_ONTOLOGY_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_ONTOLOGY_DIR / "concepts_raw.json"

    if cache_path.exists():
        logger.info("Using cached ontology data")
        return json.loads(cache_path.read_text(encoding="utf-8"))

    logger.info("Fetching ontology from GitHub...")

    concepts = []
    async with httpx.AsyncClient() as client:
        # Try to get the repo file listing first
        repo_api = "https://api.github.com/repos/seelenbrecher/islamic-agent/contents/data/quran_data"
        try:
            resp = await client.get(repo_api, timeout=30)
            if resp.status_code == 200:
                files = resp.json()
                for f in files:
                    if f["name"].endswith(".json"):
                        logger.debug(f"Downloading {f['name']}...")
                        data = await _fetch_file(client, f["name"])
                        if data:
                            raw_path = RAW_ONTOLOGY_DIR / f["name"]
                            raw_path.write_text(
                                json.dumps(data, ensure_ascii=False, indent=2),
                                encoding="utf-8",
                            )
                            if isinstance(data, list):
                                concepts.extend(data)
                            elif isinstance(data, dict):
                                concepts.append(data)
        except Exception as e:
            logger.warning(f"GitHub API failed: {e}. Trying known filenames...")
            for filename in ONTOLOGY_FILES:
                data = await _fetch_file(client, filename)
                if data:
                    raw_path = RAW_ONTOLOGY_DIR / filename
                    raw_path.write_text(
                        json.dumps(data, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    if isinstance(data, list):
                        concepts.extend(data)
                    elif isinstance(data, dict):
                        concepts.append(data)

    # Normalize concepts into standard format
    normalized = normalize_concepts(concepts)

    cache_path.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"Fetched {len(normalized)} ontology concepts")
    return normalized


def normalize_concepts(raw_concepts: list) -> list[dict]:
    """Normalize various ontology formats into our standard concept schema.

    The seelenbrecher ontology.json is a flat dict where each key is a concept_id
    and each value has: Definition, Subcategories, Related Concepts, Verses List.
    This function is called with a list containing one such dict.
    """
    normalized = []
    seen_ids = set()

    for item in raw_concepts:
        if not isinstance(item, dict):
            continue

        # The ontology is a single nested dict — flatten it
        _flatten_ontology_dict(item, normalized, seen_ids, parent_id=None)

    return normalized


def _flatten_ontology_dict(
    node: dict, output: list, seen: set, parent_id: str | None
) -> None:
    """Recursively extract concepts from the seelenbrecher ontology format."""
    # Skip metadata keys
    skip_keys = {"Definition", "Subcategories", "Related Concepts", "Verses List"}

    for key, value in node.items():
        if key in skip_keys or not isinstance(value, dict):
            continue
        if key in seen:
            continue

        concept_id = key
        seen.add(concept_id)

        # Extract fields from this concept node
        definition = value.get("Definition", "").strip()
        subcategories = value.get("Subcategories", [])
        related_raw = value.get("Related Concepts", [])
        verses_raw = value.get("Verses List", [])

        # Format name from id
        name_en = concept_id.replace("-", " ").title()

        # Build verse references as "surah:ayah"
        verses = []
        for v in verses_raw:
            if isinstance(v, dict) and "surah_id" in v and "verse_id" in v:
                verses.append(f"{v['surah_id']}:{v['verse_id']}")

        # Related concepts
        related_concepts = []
        for r in related_raw:
            if isinstance(r, dict) and "id" in r:
                related_concepts.append(r["id"])

        # Child concepts from subcategories
        child_concepts = []
        for sub in subcategories:
            if isinstance(sub, dict) and "id" in sub:
                child_concepts.append(sub["id"])

        concept = {
            "concept_id": concept_id,
            "name_en": name_en,
            "description": definition if definition else None,
            "parent_concepts": [parent_id] if parent_id else [],
            "child_concepts": child_concepts,
            "related_concepts": related_concepts,
            "verses": verses,
        }
        output.append(concept)

        # Recurse into this concept's nested sub-concepts
        _flatten_ontology_dict(value, output, seen, parent_id=concept_id)


def run():
    """Entry point."""
    import asyncio
    return asyncio.run(fetch_ontology_from_github())


if __name__ == "__main__":
    run()

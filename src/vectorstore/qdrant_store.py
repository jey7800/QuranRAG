"""Qdrant vector store implementation.

Supports three modes:
    1. In-memory (mode="memory") — for tests and quick experiments
    2. On-disk (mode="disk") — persistent local storage, no Docker needed
    3. Remote (mode="remote") — connect to a Qdrant server
"""

import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from qdrant_client import QdrantClient, models

from src.vectorstore.base import SearchResult, VectorStore

# Default path for on-disk persistent storage
DEFAULT_QDRANT_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "qdrant_db"

# Distance metric mapping
DISTANCE_MAP = {
    "cosine": models.Distance.COSINE,
    "dot": models.Distance.DOT,
    "euclidean": models.Distance.EUCLID,
}

# UUID namespace for deterministic ID generation from chunk_id strings
_UUID_NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


def _chunk_id_to_uuid(chunk_id: str) -> str:
    """Convert a chunk_id string to a deterministic UUID string."""
    return str(uuid.uuid5(_UUID_NAMESPACE, chunk_id))


class QdrantVectorStore(VectorStore):
    """Qdrant implementation of the VectorStore interface."""

    def __init__(
        self,
        mode: str = "disk",
        path: Path | None = None,
        url: str | None = None,
        api_key: str | None = None,
    ):
        if mode == "memory":
            self._client = QdrantClient(location=":memory:")
            logger.info("Qdrant initialized in-memory mode")
        elif mode == "disk":
            db_path = path or DEFAULT_QDRANT_PATH
            db_path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(db_path))
            logger.info(f"Qdrant initialized with on-disk storage: {db_path}")
        elif mode == "remote":
            remote_url = url or os.environ.get("QDRANT_URL", "http://localhost:6333")
            remote_key = api_key or os.environ.get("QDRANT_API_KEY")
            self._client = QdrantClient(url=remote_url, api_key=remote_key)
            logger.info(f"Qdrant connected to remote: {remote_url}")
        else:
            raise ValueError(f"Unknown Qdrant mode: {mode!r}. Use 'memory', 'disk', or 'remote'.")

    def create_collection(
        self,
        name: str,
        dimension: int,
        distance: str = "cosine",
        recreate: bool = False,
    ) -> None:
        if recreate and self.collection_exists(name):
            self._client.delete_collection(name)
            logger.info(f"Deleted existing collection: {name}")

        if not self.collection_exists(name):
            self._client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=DISTANCE_MAP[distance],
                ),
            )
            logger.info(f"Created collection: {name} (dim={dimension}, dist={distance})")
        else:
            logger.info(f"Collection already exists: {name}")

    def upsert(
        self,
        collection_name: str,
        ids: list[str],
        vectors: np.ndarray,
        payloads: list[dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """Upsert vectors in batches.

        Uses deterministic UUIDs generated from chunk_id strings.
        The original chunk_id is also stored in the payload under '_chunk_id'.
        """
        points = []
        for id_str, vector, payload in zip(ids, vectors, payloads):
            payload_copy = dict(payload)
            payload_copy["_chunk_id"] = id_str
            points.append(
                models.PointStruct(
                    id=_chunk_id_to_uuid(id_str),
                    vector=vector.tolist(),
                    payload=payload_copy,
                )
            )

        total = 0
        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            self._client.upsert(collection_name=collection_name, points=batch)
            total += len(batch)

        logger.info(f"Upserted {total} vectors to {collection_name}")
        return total

    def search(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        qdrant_filter = None
        if filters:
            must_conditions = []
            for key, value in filters.items():
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                )
            qdrant_filter = models.Filter(must=must_conditions)

        results = self._client.query_points(
            collection_name=collection_name,
            query=query_vector.tolist(),
            limit=top_k,
            query_filter=qdrant_filter,
        )

        return [
            SearchResult(
                chunk_id=hit.payload.get("_chunk_id", str(hit.id)),
                score=hit.score,
                payload=hit.payload,
            )
            for hit in results.points
        ]

    def count(self, collection_name: str) -> int:
        info = self._client.get_collection(collection_name)
        return info.points_count

    def delete_collection(self, collection_name: str) -> None:
        self._client.delete_collection(collection_name)
        logger.info(f"Deleted collection: {collection_name}")

    def collection_exists(self, collection_name: str) -> bool:
        collections = self._client.get_collections().collections
        return any(c.name == collection_name for c in collections)

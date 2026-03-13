"""Tests for the vector store module.

Uses in-memory Qdrant for fast, isolated testing.
"""

import numpy as np
import pytest

from src.vectorstore.qdrant_store import QdrantVectorStore


@pytest.fixture
def store():
    """In-memory Qdrant store for testing."""
    return QdrantVectorStore(mode="memory")


class TestQdrantStore:
    def test_create_collection(self, store):
        store.create_collection("test", dimension=128)
        assert store.collection_exists("test")

    def test_collection_not_exists(self, store):
        assert not store.collection_exists("nonexistent")

    def test_upsert_and_count(self, store):
        store.create_collection("test", dimension=4)
        vectors = np.random.rand(5, 4).astype(np.float32)
        ids = [f"chunk_{i}" for i in range(5)]
        payloads = [{"surah": i} for i in range(5)]
        store.upsert("test", ids, vectors, payloads)
        assert store.count("test") == 5

    def test_search_returns_results(self, store):
        store.create_collection("test", dimension=4)
        vectors = np.eye(4, dtype=np.float32)
        ids = [f"chunk_{i}" for i in range(4)]
        payloads = [{"index": i} for i in range(4)]
        store.upsert("test", ids, vectors, payloads)

        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = store.search("test", query, top_k=2)
        assert len(results) == 2
        # First result should be most similar (identity vector at index 0)
        assert results[0].payload["index"] == 0

    def test_search_scores_are_ordered(self, store):
        store.create_collection("test", dimension=4)
        vectors = np.eye(4, dtype=np.float32)
        ids = [f"chunk_{i}" for i in range(4)]
        payloads = [{"index": i} for i in range(4)]
        store.upsert("test", ids, vectors, payloads)

        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = store.search("test", query, top_k=4)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_with_filter(self, store):
        store.create_collection("test", dimension=4)
        vectors = np.eye(4, dtype=np.float32)
        ids = [f"chunk_{i}" for i in range(4)]
        payloads = [{"surah": i % 2} for i in range(4)]
        store.upsert("test", ids, vectors, payloads)

        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = store.search("test", query, top_k=4, filters={"surah": 0})
        assert all(r.payload["surah"] == 0 for r in results)

    def test_delete_collection(self, store):
        store.create_collection("test", dimension=4)
        store.delete_collection("test")
        assert not store.collection_exists("test")

    def test_recreate_collection(self, store):
        store.create_collection("test", dimension=4)
        vectors = np.random.rand(3, 4).astype(np.float32)
        store.upsert("test", ["a", "b", "c"], vectors, [{}, {}, {}])
        assert store.count("test") == 3

        store.create_collection("test", dimension=4, recreate=True)
        assert store.count("test") == 0

    def test_chunk_id_preserved_in_payload(self, store):
        store.create_collection("test", dimension=4)
        vectors = np.random.rand(1, 4).astype(np.float32)
        store.upsert("test", ["verse:2:255"], vectors, [{"surah": 2}])

        query = vectors[0]
        results = store.search("test", query, top_k=1)
        assert len(results) == 1
        assert results[0].chunk_id == "verse:2:255"
        assert results[0].payload["_chunk_id"] == "verse:2:255"

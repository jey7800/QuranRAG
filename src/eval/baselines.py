"""Baseline retrievers (BM25, TF-IDF) and adapter for the hybrid retriever."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from loguru import logger

from src.vectorstore.base import SearchResult


class BenchmarkRetriever(Protocol):
    """Common interface all retrievers must satisfy for benchmarking."""

    def search(self, query: str, top_k: int) -> list[SearchResult]: ...


# ── BM25 ──────────────────────────────────────────────────────────────────────


class BM25Retriever:
    """BM25 keyword baseline using rank_bm25."""

    def __init__(self, chunks_path: Path) -> None:
        from rank_bm25 import BM25Okapi

        self._chunk_ids: list[str] = []
        self._verse_ids: list[str] = []
        corpus: list[list[str]] = []

        with open(chunks_path, encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                self._chunk_ids.append(chunk["chunk_id"])
                self._verse_ids.append(chunk.get("verse_id", ""))
                corpus.append(chunk["text_for_embedding"].lower().split())

        self._bm25 = BM25Okapi(corpus)
        logger.info(f"BM25 baseline: indexed {len(corpus)} chunks")

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        scores = self._bm25.get_scores(query.lower().split())
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            SearchResult(
                chunk_id=self._chunk_ids[i],
                score=float(scores[i]),
                payload={"verse_id": self._verse_ids[i]},
            )
            for i in top_idx
            if scores[i] > 0
        ]


# ── TF-IDF ────────────────────────────────────────────────────────────────────


class TfidfRetriever:
    """TF-IDF sparse vector baseline using scikit-learn."""

    def __init__(self, chunks_path: Path) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        self._cosine_similarity = cosine_similarity
        self._chunk_ids: list[str] = []
        self._verse_ids: list[str] = []
        texts: list[str] = []

        with open(chunks_path, encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                self._chunk_ids.append(chunk["chunk_id"])
                self._verse_ids.append(chunk.get("verse_id", ""))
                texts.append(chunk["text_for_embedding"])

        self._vectorizer = TfidfVectorizer(max_features=10000, sublinear_tf=True)
        self._matrix = self._vectorizer.fit_transform(texts)
        logger.info(f"TF-IDF baseline: indexed {len(texts)} chunks, vocab={self._matrix.shape[1]}")

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        query_vec = self._vectorizer.transform([query])
        scores = self._cosine_similarity(query_vec, self._matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            SearchResult(
                chunk_id=self._chunk_ids[i],
                score=float(scores[i]),
                payload={"verse_id": self._verse_ids[i]},
            )
            for i in top_idx
            if scores[i] > 0
        ]


# ── Hybrid adapter ────────────────────────────────────────────────────────────


class HybridRetrieverAdapter:
    """Wraps HybridRetriever.retrieve() to conform to BenchmarkRetriever protocol."""

    def __init__(self, hybrid_retriever: Any) -> None:
        self._hybrid = hybrid_retriever

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        enriched_verses = self._hybrid.retrieve(query, top_k=top_k)
        return [
            SearchResult(
                chunk_id=f"verse:{ev.verse_id}",
                score=ev.score,
                payload={"verse_id": ev.verse_id},
            )
            for ev in enriched_verses
        ]


# ── Semantic retriever wrappers ───────────────────────────────────────────────


class DenseVerseRetriever:
    """Dense retriever restricted to verse_chunks collection only."""

    def __init__(self, semantic_retriever: Any) -> None:
        from src.config import COLLECTION_VERSE_CHUNKS

        self._semantic = semantic_retriever
        self._collections = [COLLECTION_VERSE_CHUNKS]

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        return self._semantic.search(query, top_k=top_k, collections=self._collections)


class DenseAllRetriever:
    """Dense retriever across all 3 collections (default behavior)."""

    def __init__(self, semantic_retriever: Any) -> None:
        self._semantic = semantic_retriever

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        return self._semantic.search(query, top_k=top_k)

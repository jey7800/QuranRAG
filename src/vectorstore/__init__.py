"""Vector store module — abstract interface + Qdrant implementation."""

from src.vectorstore.base import SearchResult, VectorStore
from src.vectorstore.qdrant_store import QdrantVectorStore

__all__ = ["VectorStore", "SearchResult", "QdrantVectorStore"]

"""Embedding module — abstract interface + concrete implementations."""

from src.embedding.base import Embedder
from src.embedding.factory import get_embedder

__all__ = ["Embedder", "get_embedder"]

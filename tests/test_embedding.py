"""Tests for the embedding module.

Note: Tests that require model loading are marked with @pytest.mark.slow
and require the bge-m3 model to be downloaded (~2GB).
Run with: pytest tests/test_embedding.py -m "not slow"
"""

import pytest

from src.embedding.base import Embedder
from src.embedding.factory import get_embedder


class TestFactory:
    def test_unknown_embedder_raises(self):
        with pytest.raises(ValueError, match="Unknown embedder"):
            get_embedder("nonexistent-model")

    def test_factory_returns_embedder(self):
        """Test that the factory returns a proper Embedder subclass (requires model download)."""
        embedder = get_embedder("bge-m3", batch_size=4)
        assert isinstance(embedder, Embedder)
        assert embedder.name == "bge-m3"
        assert embedder.dimension == 1024


class TestBGEM3:
    """Integration tests for BGE-M3 (requires model download)."""

    @pytest.fixture(scope="class")
    def embedder(self):
        return get_embedder("bge-m3", batch_size=4)

    def test_embed_single_returns_correct_shape(self, embedder):
        import numpy as np

        vec = embedder.embed_single("test text")
        assert vec.shape == (embedder.dimension,)
        assert vec.dtype in (np.float32, np.float64)

    def test_embed_texts_batch(self, embedder):
        texts = [
            "hello world",
            "\u0628\u0633\u0645 \u0627\u0644\u0644\u0647 \u0627\u0644\u0631\u062d\u0645\u0646 \u0627\u0644\u0631\u062d\u064a\u0645",
            "test",
        ]
        vecs = embedder.embed_texts(texts)
        assert vecs.shape == (3, embedder.dimension)

    def test_embed_query_returns_correct_shape(self, embedder):
        vec = embedder.embed_query("What does the Quran say about mercy?")
        assert vec.shape == (embedder.dimension,)

    def test_similar_texts_have_high_cosine(self, embedder):
        import numpy as np

        v1 = embedder.embed_single("God is merciful")
        v2 = embedder.embed_single("Allah is compassionate and merciful")
        cosine = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        assert cosine > 0.5

    def test_embeddings_are_normalized(self, embedder):
        import numpy as np

        vec = embedder.embed_single("test normalization")
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-4

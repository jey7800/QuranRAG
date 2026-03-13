"""Factory for constructing embedder instances by name."""

from src.embedding.base import Embedder

AVAILABLE_EMBEDDERS = ["bge-m3", "swan-large", "openai-3-large", "gemini-004"]


def get_embedder(name: str = "bge-m3", **kwargs) -> Embedder:
    """Create an embedder by name.

    Args:
        name: One of "bge-m3", "swan-large", "openai-3-large", "gemini-004".
        **kwargs: Passed to the embedder constructor.

    Returns:
        An initialized Embedder instance.

    Raises:
        ValueError: If the name is not recognized.
        ImportError: If optional dependencies are missing.
    """
    # Lazy imports to avoid loading all models at startup
    if name == "bge-m3":
        from src.embedding.bge_m3 import BGEM3Embedder

        return BGEM3Embedder(**kwargs)
    elif name == "swan-large":
        from src.embedding.swan_large import SwanLargeEmbedder

        return SwanLargeEmbedder(**kwargs)
    elif name == "openai-3-large":
        from src.embedding.openai_embedder import OpenAIEmbedder

        return OpenAIEmbedder(**kwargs)
    elif name == "gemini-004":
        from src.embedding.gemini_embedder import GeminiEmbedder

        return GeminiEmbedder(**kwargs)
    else:
        raise ValueError(
            f"Unknown embedder: {name!r}. Available: {', '.join(AVAILABLE_EMBEDDERS)}"
        )

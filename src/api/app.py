"""FastAPI application factory with lifespan for resource management."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.routes import router
from src.config import QDRANT_DB_PATH, QDRANT_MODE
from src.embedding.factory import get_embedder
from src.retrieval.context_enricher import ContextEnricher
from src.retrieval.data_store import DataStore
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.semantic_retriever import SemanticRetriever
from src.vectorstore.qdrant_store import QdrantVectorStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load data, embedder, vector store, retrievers."""
    logger.info("Starting QuranRAG server...")

    # 1. Load all static data
    data_store = DataStore()
    data_store.load()

    # 2. Initialize vector store
    vector_store = QdrantVectorStore(mode=QDRANT_MODE, path=QDRANT_DB_PATH)

    # 3. Initialize embedder (loads model to GPU/CPU)
    logger.info("Loading embedding model...")
    embedder = get_embedder()

    # 4. Build retrievers
    semantic = SemanticRetriever(vector_store, embedder)
    graph = GraphRetriever(data_store)
    enricher = ContextEnricher(data_store)
    hybrid = HybridRetriever(semantic, graph, enricher)

    # 5. Store in app state
    app.state.data_store = data_store
    app.state.vector_store = vector_store
    app.state.embedder = embedder
    app.state.semantic_retriever = semantic
    app.state.graph_retriever = graph
    app.state.enricher = enricher
    app.state.hybrid_retriever = hybrid

    logger.info("QuranRAG server ready")
    yield
    logger.info("Shutting down QuranRAG server")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="QuranRAG API",
        description="Enriched Quranic data retrieval with verse citations",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api")

    return app

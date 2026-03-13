"""Phase 2: Chunk the dataset, embed all chunks, and store in Qdrant.

Usage:
    python scripts/03_embed.py                          # Full pipeline with bge-m3
    python scripts/03_embed.py --embedder swan-large    # Use Swan-Large instead
    python scripts/03_embed.py --skip-chunking          # Only re-embed cached chunks
    python scripts/03_embed.py --benchmark              # Run benchmark after indexing
    python scripts/03_embed.py --recreate-collections   # Drop and recreate Qdrant collections
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from loguru import logger
from tqdm import tqdm

from src.config import (
    CHUNKS_DIR,
    COLLECTION_SURAH_CHUNKS,
    COLLECTION_THEMATIC_CHUNKS,
    COLLECTION_VERSE_CHUNKS,
    DEFAULT_EMBEDDER,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EVAL_QUERIES_PATH,
    QDRANT_DB_PATH,
    QDRANT_MODE,
    SURAH_CHUNKS_JSONL,
    THEMATIC_CHUNKS_JSONL,
    VERSE_CHUNKS_JSONL,
)
from src.chunking.pipeline import (
    load_cached_chunks,
    run_chunking_pipeline,
)
from src.chunking.schemas import SurahSummaryChunk, ThematicGroupChunk, VerseChunk
from src.embedding.factory import get_embedder
from src.vectorstore.qdrant_store import QdrantVectorStore


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: Chunking & Embedding")
    parser.add_argument(
        "--embedder",
        type=str,
        default=DEFAULT_EMBEDDER,
        choices=["bge-m3", "swan-large", "openai-3-large", "gemini-004"],
        help=f"Embedding model to use (default: {DEFAULT_EMBEDDER})",
    )
    parser.add_argument(
        "--qdrant-mode",
        type=str,
        default=QDRANT_MODE,
        choices=["memory", "disk", "remote"],
        help=f"Qdrant storage mode (default: {QDRANT_MODE})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=EMBEDDING_BATCH_SIZE,
        help=f"Batch size for embedding (default: {EMBEDDING_BATCH_SIZE})",
    )
    parser.add_argument(
        "--skip-chunking",
        action="store_true",
        help="Skip chunking, use cached chunk files",
    )
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Skip embedding, only run chunking",
    )
    parser.add_argument(
        "--recreate-collections",
        action="store_true",
        help="Drop and recreate Qdrant collections",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run embedding benchmark after indexing",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Only run the benchmark (skip chunking and embedding)",
    )
    return parser.parse_args()


def embed_and_store(
    chunks: list,
    collection_name: str,
    embedder,
    store: QdrantVectorStore,
    batch_size: int = 32,
    recreate: bool = False,
):
    """Embed a list of chunks and store them in Qdrant.

    Steps:
        1. Create collection if needed
        2. Extract text_for_embedding from each chunk
        3. Embed in batches
        4. Build payloads from chunk metadata
        5. Upsert to Qdrant
    """
    if not chunks:
        logger.warning(f"No chunks to embed for {collection_name}")
        return

    # Create collection
    store.create_collection(
        name=collection_name,
        dimension=embedder.dimension,
        recreate=recreate,
    )

    # Extract texts and embed
    texts = [c.text_for_embedding for c in chunks]
    logger.info(f"Embedding {len(texts)} chunks for {collection_name}...")

    t0 = time.time()
    vectors = embedder.embed_texts(texts)
    elapsed = time.time() - t0
    logger.info(f"  Embedded in {elapsed:.1f}s ({len(texts) / elapsed:.0f} chunks/s)")

    # Build payloads (all fields except text_for_embedding to save storage)
    ids = []
    payloads = []
    for chunk in chunks:
        chunk_dict = chunk.model_dump()
        # Remove the large text field from payload (it's encoded in the vector)
        chunk_dict.pop("text_for_embedding", None)
        ids.append(chunk_dict.pop("chunk_id"))
        payloads.append(chunk_dict)

    # Upsert
    store.upsert(collection_name, ids, vectors, payloads, batch_size=100)


def run_benchmark(embedder, store: QdrantVectorStore):
    """Run the intrinsic embedding benchmark.

    Loads data/eval/embedding_benchmark.json, queries each collection,
    and reports Recall@K and MRR for each chunk level.
    """
    if not EVAL_QUERIES_PATH.exists():
        logger.warning(f"Benchmark file not found: {EVAL_QUERIES_PATH}")
        return

    benchmark = json.loads(EVAL_QUERIES_PATH.read_text(encoding="utf-8"))
    queries = benchmark.get("queries", [])
    recall_ks = benchmark.get("metrics", {}).get("recall_at_k", [5, 10, 20])

    if not queries:
        logger.warning("No queries in benchmark file")
        return

    logger.info(f"Running benchmark with {len(queries)} queries...")

    # Test against verse_chunks collection
    collection = COLLECTION_VERSE_CHUNKS
    if not store.collection_exists(collection):
        logger.warning(f"Collection {collection} does not exist, skipping benchmark")
        return

    all_recalls = {k: [] for k in recall_ks}
    reciprocal_ranks = []

    for q in tqdm(queries, desc="Benchmarking"):
        query_text = q["query"]
        expected_verses = set(q["expected_verses"])

        # Embed query
        query_vec = embedder.embed_query(query_text)

        # Search
        max_k = max(recall_ks)
        results = store.search(collection, query_vec, top_k=max_k)

        # Extract verse_ids from results
        result_verse_ids = []
        for r in results:
            # chunk_id is "verse:2:255" -> verse_id is "2:255"
            chunk_id = r.chunk_id
            if chunk_id.startswith("verse:"):
                result_verse_ids.append(chunk_id[6:])
            else:
                result_verse_ids.append(r.payload.get("verse_id", ""))

        # Compute Recall@K for each K
        for k in recall_ks:
            top_k_results = set(result_verse_ids[:k])
            hits = len(expected_verses & top_k_results)
            recall = hits / len(expected_verses) if expected_verses else 0.0
            all_recalls[k].append(recall)

        # Compute MRR
        rr = 0.0
        for rank, vid in enumerate(result_verse_ids, 1):
            if vid in expected_verses:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    # Report
    logger.info("")
    logger.info(f"=== Benchmark Results ({embedder.name}) ===")
    logger.info(f"Collection: {collection} ({store.count(collection)} vectors)")
    logger.info(f"Queries: {len(queries)}")
    for k in recall_ks:
        avg_recall = sum(all_recalls[k]) / len(all_recalls[k])
        logger.info(f"  Recall@{k}: {avg_recall:.3f}")
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    logger.info(f"  MRR: {mrr:.3f}")


def main():
    args = parse_args()

    # Fast path: benchmark-only mode
    if args.benchmark_only:
        logger.info("=== Benchmark-only mode ===")
        logger.info(f"  Embedder: {args.embedder}")

        embedder_kwargs = {"batch_size": args.batch_size}
        if args.embedder in ("bge-m3", "swan-large") and EMBEDDING_DEVICE:
            embedder_kwargs["device"] = EMBEDDING_DEVICE
        embedder = get_embedder(args.embedder, **embedder_kwargs)
        logger.info(f"  Model: {embedder.name}, Dimension: {embedder.dimension}")

        store = QdrantVectorStore(mode=args.qdrant_mode, path=QDRANT_DB_PATH)
        run_benchmark(embedder, store)
        return

    logger.info("=== Phase 2: Chunking & Embedding ===")
    logger.info(f"  Embedder: {args.embedder}")
    logger.info(f"  Qdrant mode: {args.qdrant_mode}")
    logger.info(f"  Batch size: {args.batch_size}")

    # Step 1: Initialize embedder
    logger.info(f"\n[1/4] Initializing embedder: {args.embedder}")
    embedder_kwargs = {"batch_size": args.batch_size}
    if args.embedder in ("bge-m3", "swan-large") and EMBEDDING_DEVICE:
        embedder_kwargs["device"] = EMBEDDING_DEVICE
    embedder = get_embedder(args.embedder, **embedder_kwargs)
    logger.info(f"  Model: {embedder.name}, Dimension: {embedder.dimension}")

    # Step 2: Chunking
    if args.skip_chunking:
        logger.info("\n[2/4] Loading cached chunks...")
        verse_chunks = load_cached_chunks(VERSE_CHUNKS_JSONL, VerseChunk) or []
        thematic_chunks = load_cached_chunks(THEMATIC_CHUNKS_JSONL, ThematicGroupChunk) or []
        surah_chunks = load_cached_chunks(SURAH_CHUNKS_JSONL, SurahSummaryChunk) or []

        if not verse_chunks:
            logger.error("No cached verse chunks found! Run without --skip-chunking first.")
            sys.exit(1)
    else:
        logger.info("\n[2/4] Running chunking pipeline...")
        chunk_results = run_chunking_pipeline(embedder=embedder)
        verse_chunks = chunk_results["verse"]
        thematic_chunks = chunk_results["thematic"]
        surah_chunks = chunk_results["surah"]

    logger.info(
        f"  Chunks: {len(verse_chunks)} verse, "
        f"{len(thematic_chunks)} thematic, {len(surah_chunks)} surah"
    )

    if args.skip_embedding:
        logger.info("\n[3/4] Skipping embedding (--skip-embedding)")
        logger.info("[4/4] Skipping vector storage")
    else:
        # Step 3: Initialize vector store
        logger.info(f"\n[3/4] Initializing Qdrant ({args.qdrant_mode} mode)...")
        store = QdrantVectorStore(mode=args.qdrant_mode, path=QDRANT_DB_PATH)

        # Step 4: Embed and store each chunk level
        logger.info("\n[4/4] Embedding and storing chunks...")

        embed_and_store(
            verse_chunks,
            COLLECTION_VERSE_CHUNKS,
            embedder,
            store,
            args.batch_size,
            args.recreate_collections,
        )
        embed_and_store(
            thematic_chunks,
            COLLECTION_THEMATIC_CHUNKS,
            embedder,
            store,
            args.batch_size,
            args.recreate_collections,
        )
        embed_and_store(
            surah_chunks,
            COLLECTION_SURAH_CHUNKS,
            embedder,
            store,
            args.batch_size,
            args.recreate_collections,
        )

        logger.info("\n=== Vector store summary ===")
        for col in [COLLECTION_VERSE_CHUNKS, COLLECTION_THEMATIC_CHUNKS, COLLECTION_SURAH_CHUNKS]:
            if store.collection_exists(col):
                logger.info(f"  {col}: {store.count(col)} vectors")

        # Optional benchmark
        if args.benchmark:
            logger.info("")
            logger.info("=== Running embedding benchmark ===")
            run_benchmark(embedder, store)

    logger.info("\n=== Phase 2 complete ===")


if __name__ == "__main__":
    main()

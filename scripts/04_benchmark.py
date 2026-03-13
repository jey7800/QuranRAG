"""Comprehensive retrieval benchmark: 200 queries × 5 retrievers.

Usage:
    python scripts/04_benchmark.py                          # Full benchmark
    python scripts/04_benchmark.py --retriever bm25         # Single retriever
    python scripts/04_benchmark.py --quick                  # 50-query subset
    python scripts/04_benchmark.py --regenerate-queries     # Force re-generate queries
    python scripts/04_benchmark.py --k 5 10 20 50           # Custom K values
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from tqdm import tqdm

from src.config import (
    BENCHMARK_BOOTSTRAP_N,
    BENCHMARK_K_VALUES,
    BENCHMARK_QUERIES_PATH,
    BENCHMARK_RESULTS_DIR,
    BENCHMARK_SEED,
    DEFAULT_EMBEDDER,
    EMBEDDING_DEVICE,
    QDRANT_DB_PATH,
    QDRANT_MODE,
    VERSE_CHUNKS_JSONL,
)
from src.eval.metrics import (
    AggregateMetrics,
    QueryMetrics,
    aggregate_metrics,
    compute_query_metrics,
)
from src.eval.query_generator import (
    BenchmarkQuery,
    generate_queries,
    load_queries,
    save_queries,
)
from src.eval.report import generate_report, save_report


RETRIEVER_NAMES = ["bm25", "tfidf", "dense_verse", "dense_all", "hybrid"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QuranRAG Retrieval Benchmark")
    parser.add_argument(
        "--retriever",
        type=str,
        choices=RETRIEVER_NAMES + ["all"],
        default="all",
        help="Which retriever to benchmark (default: all)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run a 50-query stratified subset for fast iteration",
    )
    parser.add_argument(
        "--regenerate-queries", action="store_true",
        help="Force re-generation of the query set",
    )
    parser.add_argument(
        "--k", nargs="+", type=int, default=None,
        help=f"K values for metrics (default: {BENCHMARK_K_VALUES})",
    )
    parser.add_argument(
        "--embedder", type=str, default=DEFAULT_EMBEDDER,
        help=f"Embedder for dense retrievers (default: {DEFAULT_EMBEDDER})",
    )
    parser.add_argument(
        "--qdrant-path", type=str, default=None,
        help="Override Qdrant DB path (useful if main DB is locked by MCP server)",
    )
    return parser.parse_args()


def stratified_sample(queries: list[BenchmarkQuery], n: int, seed: int) -> list[BenchmarkQuery]:
    """Sample n queries maintaining approximate category proportions."""
    rng = Random(seed)
    by_cat: dict[str, list[BenchmarkQuery]] = {}
    for q in queries:
        by_cat.setdefault(q.category, []).append(q)

    total = len(queries)
    sampled: list[BenchmarkQuery] = []
    for cat, cat_queries in by_cat.items():
        # Proportional allocation, at least 1 per category
        cat_n = max(1, round(n * len(cat_queries) / total))
        sampled.extend(rng.sample(cat_queries, min(cat_n, len(cat_queries))))

    # Trim if over n
    if len(sampled) > n:
        sampled = rng.sample(sampled, n)
    return sampled


def extract_verse_ids(results: list) -> list[str]:
    """Extract verse IDs from search results, handling all chunk types."""
    verse_ids: list[str] = []
    seen: set[str] = set()
    for r in results:
        cid = r.chunk_id

        if cid.startswith("verse:"):
            vid = cid[len("verse:"):]
            if vid not in seen:
                verse_ids.append(vid)
                seen.add(vid)
        elif cid.startswith("group:") or cid.startswith("surah:"):
            # Thematic/surah chunks: get verse_ids from payload
            for vid in r.payload.get("verse_ids", []):
                if vid not in seen:
                    verse_ids.append(vid)
                    seen.add(vid)
        else:
            vid = r.payload.get("verse_id", "")
            if vid and vid not in seen:
                verse_ids.append(vid)
                seen.add(vid)
    return verse_ids


def build_retrievers(
    selected: str,
    embedder_name: str,
    qdrant_path: Path | None = None,
) -> dict[str, object]:
    """Initialize the requested retrievers."""
    retrievers: dict[str, object] = {}
    need_baselines = selected in ("all", "bm25", "tfidf")
    need_dense = selected in ("all", "dense_verse", "dense_all", "hybrid")

    # BM25 and TF-IDF baselines
    if need_baselines or selected == "bm25":
        from src.eval.baselines import BM25Retriever
        if selected in ("all", "bm25"):
            retrievers["bm25"] = BM25Retriever(VERSE_CHUNKS_JSONL)

    if need_baselines or selected == "tfidf":
        from src.eval.baselines import TfidfRetriever
        if selected in ("all", "tfidf"):
            retrievers["tfidf"] = TfidfRetriever(VERSE_CHUNKS_JSONL)

    # Dense retrievers (need embedder + Qdrant)
    if need_dense:
        from src.embedding.factory import get_embedder
        from src.eval.baselines import DenseAllRetriever, DenseVerseRetriever
        from src.retrieval.semantic_retriever import SemanticRetriever
        from src.vectorstore.qdrant_store import QdrantVectorStore

        db_path = qdrant_path or QDRANT_DB_PATH
        store = QdrantVectorStore(mode=QDRANT_MODE, path=db_path)
        embedder = get_embedder(embedder_name, device=EMBEDDING_DEVICE)
        semantic = SemanticRetriever(store, embedder)

        if selected in ("all", "dense_verse"):
            retrievers["dense_verse"] = DenseVerseRetriever(semantic)
        if selected in ("all", "dense_all"):
            retrievers["dense_all"] = DenseAllRetriever(semantic)
        if selected in ("all", "hybrid"):
            from src.eval.baselines import HybridRetrieverAdapter
            from src.retrieval.context_enricher import ContextEnricher
            from src.retrieval.data_store import DataStore
            from src.retrieval.graph_retriever import GraphRetriever
            from src.retrieval.hybrid_retriever import HybridRetriever

            data_store = DataStore()
            data_store.load()
            graph = GraphRetriever(data_store)
            enricher = ContextEnricher(data_store)
            hybrid = HybridRetriever(semantic, graph, enricher)
            retrievers["hybrid"] = HybridRetrieverAdapter(hybrid)

    return retrievers


def run_benchmark(
    queries: list[BenchmarkQuery],
    retrievers: dict[str, object],
    k_values: list[int],
) -> dict[str, tuple[AggregateMetrics, list[QueryMetrics]]]:
    """Run all queries against all retrievers and compute metrics."""
    max_k = max(k_values)
    results: dict[str, tuple[AggregateMetrics, list[QueryMetrics]]] = {}

    for ret_name, retriever in retrievers.items():
        logger.info(f"Running {ret_name} on {len(queries)} queries (top_k={max_k})...")
        per_query: list[QueryMetrics] = []
        t0 = time.perf_counter()

        for q in tqdm(queries, desc=ret_name, leave=False):
            search_results = retriever.search(q.query, top_k=max_k)
            retrieved_ids = extract_verse_ids(search_results)
            relevant = set(q.expected_verses)

            qm = compute_query_metrics(
                query_id=q.id,
                retrieved_ids=retrieved_ids,
                relevant_ids=relevant,
                k_values=k_values,
                category=q.category,
                language=q.language,
                difficulty=q.difficulty,
            )
            per_query.append(qm)

        elapsed = time.perf_counter() - t0
        logger.info(f"  {ret_name}: {len(queries)} queries in {elapsed:.1f}s")

        # Compute aggregate (exclude negative queries for main metrics)
        non_negative = [qm for qm in per_query if qm.category != "negative"]
        agg = aggregate_metrics(non_negative, k_values, BENCHMARK_BOOTSTRAP_N, BENCHMARK_SEED)

        # Log summary
        for k in k_values:
            logger.info(f"  Recall@{k}: {agg.recall[k]}")
        logger.info(f"  MRR: {agg.mrr}")
        logger.info(f"  MAP: {agg.map}")

        results[ret_name] = (agg, per_query)

    return results


def main() -> None:
    args = parse_args()
    k_values = args.k or BENCHMARK_K_VALUES

    # Step 1: Load or generate queries
    if args.regenerate_queries or not BENCHMARK_QUERIES_PATH.exists():
        logger.info("Generating benchmark queries...")
        queries = generate_queries()
        save_queries(queries, BENCHMARK_QUERIES_PATH)
    else:
        queries = load_queries(BENCHMARK_QUERIES_PATH)

    if args.quick:
        queries = stratified_sample(queries, n=50, seed=BENCHMARK_SEED)
        logger.info(f"Quick mode: sampled {len(queries)} queries")

    # Step 2: Build retrievers
    logger.info(f"Initializing retrievers: {args.retriever}")
    qdrant_path = Path(args.qdrant_path) if args.qdrant_path else None
    retrievers = build_retrievers(args.retriever, args.embedder, qdrant_path)

    if not retrievers:
        logger.error("No retrievers initialized.")
        sys.exit(1)

    # Step 3: Run benchmark
    results = run_benchmark(queries, retrievers, k_values)

    # Step 4: Generate report
    BENCHMARK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    report_md = generate_report(results, k_values, len(queries))
    report_path = BENCHMARK_RESULTS_DIR / f"benchmark_{timestamp}.md"
    save_report(report_md, report_path)

    # Save raw JSON results
    json_path = BENCHMARK_RESULTS_DIR / f"benchmark_{timestamp}.json"
    json_data = {
        "timestamp": timestamp,
        "n_queries": len(queries),
        "k_values": k_values,
        "quick_mode": args.quick,
        "retrievers": {},
    }
    for ret_name, (agg, per_query) in results.items():
        json_data["retrievers"][ret_name] = {
            "n_queries": agg.n_queries,
            "recall": {str(k): {"mean": v.mean, "ci_lower": v.lower, "ci_upper": v.upper}
                       for k, v in agg.recall.items()},
            "precision": {str(k): {"mean": v.mean, "ci_lower": v.lower, "ci_upper": v.upper}
                          for k, v in agg.precision.items()},
            "ndcg": {str(k): {"mean": v.mean, "ci_lower": v.lower, "ci_upper": v.upper}
                     for k, v in agg.ndcg.items()},
            "mrr": {"mean": agg.mrr.mean, "ci_lower": agg.mrr.lower, "ci_upper": agg.mrr.upper}
                    if agg.mrr else None,
            "map": {"mean": agg.map.mean, "ci_lower": agg.map.lower, "ci_upper": agg.map.upper}
                    if agg.map else None,
            "per_query": [
                {
                    "id": qm.query_id,
                    "category": qm.category,
                    "language": qm.language,
                    "mrr": qm.mrr,
                    "ap": qm.ap,
                    "recall": {str(k): v for k, v in qm.recall.items()},
                }
                for qm in per_query
            ],
        }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Report saved to {report_path}")
    logger.info(f"Raw data saved to {json_path}")

    # Print summary to console
    print("\n" + "=" * 70)
    print(f"BENCHMARK RESULTS ({len(queries)} queries)")
    print("=" * 70)
    for ret_name, (agg, _) in results.items():
        print(f"\n  {ret_name}:")
        for k in k_values:
            print(f"    Recall@{k}: {agg.recall[k].mean:.3f}")
        print(f"    MRR:       {agg.mrr.mean:.3f}" if agg.mrr else "")
        print(f"    MAP:       {agg.map.mean:.3f}" if agg.map else "")
        for k in k_values:
            print(f"    NDCG@{k}:  {agg.ndcg[k].mean:.3f}")
    print()


if __name__ == "__main__":
    main()

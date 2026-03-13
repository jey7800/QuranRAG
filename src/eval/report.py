"""Generate markdown benchmark reports."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.eval.metrics import AggregateMetrics, ConfidenceInterval, QueryMetrics


def _ci_str(ci: ConfidenceInterval) -> str:
    return f"{ci.mean:.3f}"


def _ci_full(ci: ConfidenceInterval) -> str:
    return f"{ci.mean:.3f} [{ci.lower:.3f}, {ci.upper:.3f}]"


def generate_report(
    results: dict[str, tuple[AggregateMetrics, list[QueryMetrics]]],
    k_values: list[int],
    n_queries: int,
) -> str:
    """Generate a full markdown benchmark report.

    Args:
        results: {retriever_name: (aggregate_metrics, per_query_metrics)}
        k_values: K values used (e.g. [5, 10, 20])
        n_queries: Total number of queries
    """
    lines: list[str] = []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines.append(f"# QuranRAG Benchmark Report\n")
    lines.append(f"Generated: {ts}  ")
    lines.append(f"Queries: {n_queries}\n")

    # ── Overall results table ─────────────────────────────────────────────────
    lines.append("## Overall Results\n")
    headers = ["Retriever"]
    for k in k_values:
        headers.append(f"Recall@{k}")
    headers.extend(["MRR", "MAP"])
    for k in k_values:
        headers.append(f"NDCG@{k}")

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for name, (agg, _) in results.items():
        row = [f"**{name}**"]
        for k in k_values:
            row.append(_ci_str(agg.recall[k]))
        row.append(_ci_str(agg.mrr) if agg.mrr else "—")
        row.append(_ci_str(agg.map) if agg.map else "—")
        for k in k_values:
            row.append(_ci_str(agg.ndcg[k]))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ── Confidence intervals ──────────────────────────────────────────────────
    lines.append("## Confidence Intervals (95% bootstrap)\n")
    ci_headers = ["Retriever", "Recall@10", "MRR", "MAP", "NDCG@10"]
    lines.append("| " + " | ".join(ci_headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(ci_headers)) + " |")

    k_mid = k_values[1] if len(k_values) > 1 else k_values[0]
    for name, (agg, _) in results.items():
        row = [
            f"**{name}**",
            _ci_full(agg.recall[k_mid]),
            _ci_full(agg.mrr) if agg.mrr else "—",
            _ci_full(agg.map) if agg.map else "—",
            _ci_full(agg.ndcg[k_mid]),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ── Per-category breakdown ────────────────────────────────────────────────
    lines.append("## By Category\n")
    # Collect categories from first retriever's per-query metrics
    first_pq = list(results.values())[0][1]
    categories = sorted({qm.category for qm in first_pq})

    for name, (_, per_query) in results.items():
        lines.append(f"### {name}\n")
        cat_headers = ["Category", "N"]
        for k in k_values:
            cat_headers.append(f"R@{k}")
        cat_headers.extend(["MRR", "MAP"])
        lines.append("| " + " | ".join(cat_headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(cat_headers)) + " |")

        for cat in categories:
            cat_qms = [qm for qm in per_query if qm.category == cat]
            if not cat_qms:
                continue
            n = len(cat_qms)
            row = [cat, str(n)]
            for k in k_values:
                mean = sum(qm.recall[k] for qm in cat_qms) / n
                row.append(f"{mean:.3f}")
            mrr_mean = sum(qm.mrr for qm in cat_qms) / n
            map_mean = sum(qm.ap for qm in cat_qms) / n
            row.extend([f"{mrr_mean:.3f}", f"{map_mean:.3f}"])
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # ── Per-language breakdown ────────────────────────────────────────────────
    lines.append("## By Language\n")
    languages = sorted({qm.language for qm in first_pq})

    lang_headers = ["Retriever"]
    for lang in languages:
        for k in k_values:
            lang_headers.append(f"{lang.upper()} R@{k}")
    lines.append("| " + " | ".join(lang_headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(lang_headers)) + " |")

    for name, (_, per_query) in results.items():
        row = [f"**{name}**"]
        for lang in languages:
            lang_qms = [qm for qm in per_query if qm.language == lang]
            for k in k_values:
                if lang_qms:
                    mean = sum(qm.recall[k] for qm in lang_qms) / len(lang_qms)
                    row.append(f"{mean:.3f}")
                else:
                    row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ── Worst queries (error analysis) ────────────────────────────────────────
    lines.append("## Error Analysis: Worst Queries (by MRR)\n")
    # Show worst 10 from the best retriever (last one, presumably hybrid)
    best_name = list(results.keys())[-1]
    _, best_pq = results[best_name]
    # Filter out negative queries (expected MRR=0)
    non_neg = [qm for qm in best_pq if qm.category != "negative"]
    worst = sorted(non_neg, key=lambda qm: qm.mrr)[:10]
    lines.append(f"Retriever: **{best_name}**\n")
    lines.append("| Query ID | Category | Lang | MRR | R@10 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for qm in worst:
        k_mid_val = qm.recall.get(k_mid, 0)
        lines.append(f"| {qm.query_id} | {qm.category} | {qm.language} | {qm.mrr:.3f} | {k_mid_val:.3f} |")
    lines.append("")

    return "\n".join(lines)


def save_report(report: str, path: Path) -> None:
    """Save markdown report to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)

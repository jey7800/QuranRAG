"""Format enriched results as structured text for MCP tool output."""

from src.api.schemas import (
    CompareResponse,
    ContextResponse,
    EnrichedVerse,
    ThemeResponse,
)


def format_verse(v: EnrichedVerse) -> str:
    """Format a single enriched verse for LLM consumption."""
    lines = []
    lines.append(f"## [{v.verse_id}] {v.surah_name_en} ({v.surah_name_ar})")
    lines.append("")
    lines.append(f"**Arabic:** {v.text_arabic}")
    lines.append(f"**English (Asad):** {v.text_en_asad}")
    if v.text_fr_hamidullah:
        lines.append(f"**French (Hamidullah):** {v.text_fr_hamidullah}")
    lines.append("")
    lines.append(
        f"Period: {v.revelation_period} | Juz: {v.juz} | Hizb: {v.hizb}"
    )

    if v.topic_tags:
        lines.append(f"Topics: {', '.join(v.topic_tags)}")

    if v.asbab_al_nuzul:
        lines.append(f"\n**Revelation context:** {v.asbab_al_nuzul}")
    elif v.asbab_status == "not_documented":
        lines.append("\n*No documented revelation context for this verse.*")

    if v.polysemy_info:
        lines.append("\n**Polysemy alerts:**")
        for p in v.polysemy_info:
            senses_str = "; ".join(
                f"{s.get('meaning_en', '')}" for s in p.senses
            )
            lines.append(f"  - {p.word_arabic} ({p.root}): {senses_str}")
            if p.scholarly_note:
                lines.append(f"    Note: {p.scholarly_note}")

    if v.abrogation_info:
        a = v.abrogation_info
        if a.abrogated_by:
            lines.append(f"\n**Abrogation:** This verse is abrogated by [{a.abrogated_by}]")
        elif a.abrogates:
            lines.append(f"\n**Abrogation:** This verse abrogates [{a.abrogates}]")
        lines.append(f"  Topic: {a.topic} | Consensus: {a.scholarly_consensus}")
        if a.note:
            lines.append(f"  {a.note}")

    if v.related_verses:
        lines.append(f"\nRelated verses: {', '.join(v.related_verses[:10])}")

    return "\n".join(lines)


def format_search_results(results: list[EnrichedVerse], query: str) -> str:
    """Format search results for MCP output."""
    if not results:
        return f"No verses found for query: '{query}'"

    lines = [f"# Search results for: '{query}'\n"]
    lines.append(f"Found {len(results)} relevant verse(s).\n")

    for i, v in enumerate(results, 1):
        lines.append(f"---\n### Result {i}" + (f" (score: {v.score:.3f})" if v.score else ""))
        lines.append(format_verse(v))

    return "\n".join(lines)


def format_theme(theme: ThemeResponse) -> str:
    """Format theme exploration results."""
    lines = [f"# Theme: {theme.name_en}"]
    if theme.description:
        lines.append(f"\n{theme.description}")
    if theme.related_concepts:
        lines.append(f"\nRelated concepts: {', '.join(theme.related_concepts)}")
    lines.append(f"\n{len(theme.verses)} verse(s) found:\n")

    for v in theme.verses:
        lines.append("---")
        lines.append(format_verse(v))

    return "\n".join(lines)


def format_comparison(comp: CompareResponse) -> str:
    """Format translation comparison."""
    lines = [f"# Translation comparison for [{comp.verse_id}]\n"]
    lines.append(f"**Arabic:** {comp.text_arabic}\n")
    for name, text in comp.translations.items():
        label = name.replace("_", " ").title()
        lines.append(f"**{label}:** {text}")
    return "\n".join(lines)


def format_context(ctx: ContextResponse) -> str:
    """Format surrounding context."""
    center = ctx.center_verse
    lines = [f"# Context for [{center.verse_id}] (range: +-{ctx.range})\n"]

    for n in ctx.neighbors:
        marker = " **>>**" if n.verse_id == center.verse_id else ""
        lines.append(f"[{n.verse_id}]{marker} {n.text_en}")

    lines.append(f"\n---\n### Center verse details:")
    lines.append(format_verse(center))

    return "\n".join(lines)

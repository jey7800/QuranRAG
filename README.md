# QuranRAG

Open-source Retrieval-Augmented Generation system for answering questions grounded in the Quran with verse citations, historical context, and polysemy awareness.

## Principles

- **Fidelity** — every answer cites verse references (surah:ayah)
- **Neutrality** — no school of thought is privileged
- **Transparency** — polysemous words, missing context, and debated interpretations are flagged
- **Context** — each verse is enriched with revelation circumstances, period, and thematic links

## Features

- **6,236 verses** enriched with 4 layers of metadata: text (Arabic + English + French), morphology & roots, historical context (Asbab al-Nuzul), and thematic links
- **3-level chunking**: verse-level (6,236), thematic groups (1,637), surah summaries (114) — 7,987 total chunks
- **Hybrid retrieval**: semantic vector search + ontology graph expansion + context enrichment
- **Polysemy awareness**: flags Arabic words with multiple scholarly interpretations
- **Abrogation tracking**: notes verses with abrogation relationships and scholarly consensus
- **Multilingual**: queries in English, Arabic, and French
- **Two interfaces**: REST API and Claude MCP integration

## Quick Start

### Prerequisites

- Python 3.11+
- NVIDIA GPU recommended (CPU works but is slower)

### Installation

```bash
git clone https://github.com/your-repo/QuranRAG.git
cd QuranRAG
python -m venv .venv
.venv/Scripts/activate   # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -e .
```

### Build the Pipeline

```bash
# Phase 1: Fetch and build the enriched dataset
python scripts/01_fetch_data.py
python scripts/02_build_dataset.py

# Phase 2: Chunk, embed, and index into Qdrant
python scripts/03_embed.py

# Phase 3: Launch the server
python scripts/04_serve.py
```

### Launch Options

```bash
# REST API on port 8000
python scripts/04_serve.py

# MCP stdio mode (Claude Desktop)
python scripts/04_serve.py --mcp-stdio

# MCP HTTP mode (Claude.ai web)
python scripts/04_serve.py --mcp-http --port 8001
```

## Claude Desktop Integration

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "quran-rag": {
      "command": "/path/to/QuranRAG/.venv/Scripts/python.exe",
      "args": ["/path/to/QuranRAG/scripts/04_serve.py", "--mcp-stdio"],
      "cwd": "/path/to/QuranRAG"
    }
  }
}
```

Use absolute paths. The server lazy-loads the embedding model on first tool call for instant startup.

### MCP Tools

| Tool | Description |
|------|-------------|
| `search_verses` | Semantic search for verses relevant to a query |
| `get_verse` | Full details for a specific verse (e.g. `2:255`) |
| `explore_theme` | All verses related to a concept (e.g. `justice`, `moses`) |
| `compare_translations` | Side-by-side English/French translations with polysemy alerts |
| `get_context` | Surrounding verses and revelation circumstances |

## REST API

Base URL: `http://localhost:8000/api`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/search?q=...&top_k=10` | GET | Search verses by query |
| `/api/verse/{surah}/{ayah}` | GET | Get enriched verse |
| `/api/surah/{number}` | GET | Get all verses in a surah |
| `/api/theme/{concept_id}` | GET | Explore a thematic concept |
| `/api/compare/{surah}/{ayah}` | GET | Compare translations |
| `/api/context/{surah}/{ayah}?range=5` | GET | Get surrounding context |
| `/api/stats` | GET | Dataset statistics |

Optional filters on `/api/search`: `surah` (1-114), `period` (`meccan`/`medinan`).

## Architecture

```
src/
├── api/           REST API (FastAPI)
├── mcp/           Claude MCP server (5 tools)
├── retrieval/     Hybrid retrieval pipeline
│   ├── semantic_retriever.py   Vector similarity (Qdrant)
│   ├── graph_retriever.py      Ontology expansion (NetworkX)
│   ├── context_enricher.py     Polysemy, abrogation, neighbors
│   ├── hybrid_retriever.py     Combines all retrievers
│   └── data_store.py           In-memory verse + concept store
├── embedding/     4 embedder implementations
├── vectorstore/   Qdrant (memory/disk/remote modes)
├── chunking/      3-level chunking pipeline
└── data/          Dataset construction & schemas
```

### Retrieval Pipeline

1. **Semantic search** — embeds the query with BGE-M3, searches 3 Qdrant collections (verse, thematic, surah chunks)
2. **Graph expansion** — extracts topic tags from top results, traverses the ontology graph (285 concepts) via BFS
3. **Deduplication** — merges results by verse ID, keeps highest score
4. **Enrichment** — attaches polysemy alerts, abrogation info, neighboring verses, morphological roots

### Dataset Layers

| Layer | Content | Source |
|-------|---------|--------|
| Text | Arabic (Uthmani), English (Asad), French (Hamidullah), transliteration | alquran.cloud API, HuggingFace |
| Linguistics | Word-level morphology, roots, lemmas, POS tags, polysemy detection | Quranic Arabic Corpus |
| Historical | Asbab al-Nuzul, revelation period (Meccan/Medinan), chronological order | Al-Wahidi |
| Thematic | Ontology concepts, related verses, abrogation relationships | Quranic Ontology |

## Embedder Options

| Name | Model | Dimensions | Notes |
|------|-------|-----------|-------|
| `bge-m3` (default) | BAAI/bge-m3 | 1024 | Multilingual, best balance |
| `swan-large` | OALL/swan-large-embedding | 1024 | Arabic-optimized |
| `openai-3-large` | text-embedding-3-large | 3072 | API-based, requires key |
| `gemini-004` | text-embedding-004 | 768 | API-based, requires key |

Switch embedder: `python scripts/03_embed.py --embedder swan-large`

## Benchmark

20 hand-curated queries across theology, narrative, jurisprudence, ethics, cosmology, and eschatology.

| Metric | Score |
|--------|-------|
| Recall@5 | 0.243 |
| Recall@10 | 0.299 |
| Recall@20 | 0.384 |
| MRR | 0.545 |

Run benchmark: `python scripts/03_embed.py --benchmark-only`

## Development

```bash
pip install -e .[dev]
pytest
ruff check src/
```

## License

MIT

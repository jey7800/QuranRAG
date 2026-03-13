"""Central configuration for the Quran RAG project."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "eval"

RAW_QURAN_API_DIR = RAW_DIR / "quran_api"
RAW_QURAN_MD_DIR = RAW_DIR / "quran_md"
RAW_CORPUS_DIR = RAW_DIR / "quranic_corpus"
RAW_ONTOLOGY_DIR = RAW_DIR / "ontology"
RAW_ASBAB_DIR = RAW_DIR / "asbab_al_nuzul"

VERSES_JSONL = PROCESSED_DIR / "verses.jsonl"
CHAPTERS_JSON = PROCESSED_DIR / "chapters.json"
CONCEPTS_JSON = PROCESSED_DIR / "concepts.json"
STATS_JSON = PROCESSED_DIR / "stats.json"

POLYSEMY_CATALOG = RAW_DIR / "polysemy_catalog.json"
ABROGATION_FILE = RAW_DIR / "abrogation.json"

# ── Quran constants ────────────────────────────────────────────────────────────
TOTAL_SURAHS = 114
TOTAL_VERSES = 6236

# ── API settings ───────────────────────────────────────────────────────────────
ALQURAN_CLOUD_BASE_URL = "https://api.alquran.cloud/v1"
ALQURAN_EDITIONS = "quran-uthmani,en.asad,fr.hamidullah"
API_SLEEP_SECONDS = 0.2  # polite rate limit

# ── HuggingFace dataset ───────────────────────────────────────────────────────
QURAN_MD_DATASET = "Buraaq/quran-md-ayahs"

# ── Morphology corpus ─────────────────────────────────────────────────────────
MORPHOLOGY_FILE = RAW_CORPUS_DIR / "quranic-corpus-morphology-0.4.txt"

# ── Asbab al-Nuzul ────────────────────────────────────────────────────────────
ASBAB_PDF = RAW_ASBAB_DIR / "Asbab Al-Nuzul by Al-Wahidi.pdf"
ASBAB_PARSED = RAW_ASBAB_DIR / "asbab_parsed.json"

# ── Ontology ───────────────────────────────────────────────────────────────────
ONTOLOGY_GITHUB_BASE = (
    "https://raw.githubusercontent.com/seelenbrecher/islamic-agent/master/data/quran_data"
)

# ── Standard scholarly chronological order of the 114 surahs ───────────────────
# Index = chronological position (0-based), value = surah number
REVELATION_ORDER = [
    96, 68, 73, 74, 1, 111, 81, 87, 92, 89,   # 1-10
    93, 94, 103, 100, 108, 102, 107, 109, 105, 113,  # 11-20
    114, 112, 53, 80, 97, 91, 85, 95, 106, 101,  # 21-30
    75, 104, 77, 50, 90, 86, 54, 38, 7, 72,   # 31-40
    36, 25, 35, 19, 20, 56, 26, 27, 28, 17,   # 41-50
    10, 11, 12, 15, 6, 37, 31, 34, 39, 40,   # 51-60
    41, 42, 43, 44, 45, 46, 51, 88, 18, 16,   # 61-70
    71, 14, 21, 23, 32, 52, 67, 69, 70, 78,   # 71-80
    79, 82, 84, 30, 29, 83, 2, 8, 3, 33,    # 81-90
    60, 4, 99, 57, 47, 13, 55, 76, 65, 98,   # 91-100
    59, 24, 22, 63, 58, 49, 66, 64, 61, 62,   # 101-110
    48, 5, 9, 110,                              # 111-114
]

# Reverse map: surah_number → chronological position (1-based)
SURAH_TO_REVELATION_ORDER = {
    surah: pos + 1 for pos, surah in enumerate(REVELATION_ORDER)
}

# ── Chunking settings ─────────────────────────────────────────────────────────
DEFAULT_CHUNK_SIMILARITY_THRESHOLD = 0.65
MIN_GROUP_SIZE = 3
MAX_GROUP_SIZE = 7

CHUNKS_DIR = PROCESSED_DIR / "chunks"
VERSE_CHUNKS_JSONL = CHUNKS_DIR / "verse_chunks.jsonl"
THEMATIC_CHUNKS_JSONL = CHUNKS_DIR / "thematic_chunks.jsonl"
SURAH_CHUNKS_JSONL = CHUNKS_DIR / "surah_chunks.jsonl"

# ── Embedding settings ─────────────────────────────────────────────────────────
DEFAULT_EMBEDDER = "bge-m3"
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_DEVICE = None  # None = auto-detect, "cpu", "cuda"

# ── Vector store settings ─────────────────────────────────────────────────────
QDRANT_MODE = "disk"  # "memory", "disk", or "remote"
QDRANT_DB_PATH = DATA_DIR / "qdrant_db"

COLLECTION_VERSE_CHUNKS = "verse_chunks"
COLLECTION_THEMATIC_CHUNKS = "thematic_chunks"
COLLECTION_SURAH_CHUNKS = "surah_chunks"

# ── Retrieval settings ─────────────────────────────────────────────────────────
DEFAULT_TOP_K = 10
SEMANTIC_RETRIEVAL_K = 20
GRAPH_TOP_CONCEPTS = 3
GRAPH_HOPS = 1

# ── Benchmark ─────────────────────────────────────────────────────────────────
EVAL_QUERIES_PATH = EVAL_DIR / "embedding_benchmark.json"

# ── API settings ──────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
NEIGHBOR_RANGE = 2

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"

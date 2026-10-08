"""
Central configuration for the RAG Evaluation & Optimization Framework.
Every tunable setting lives here so you can defend every design choice
in one place, and so the rest of the code stays clean.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root, if present

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
GROUND_TRUTH_PATH = PROJECT_ROOT / "data" / "ground_truth.json"
CHROMA_DB_DIR = PROJECT_ROOT / "data" / "chroma_db"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# LLM (Groq — free tier, no credit card required)
# https://console.groq.com
# ---------------------------------------------------------------------------
# >>> YOU MUST FILL THIS IN <<<
# Put your key in a .env file (see .env.example) — never hardcode it here.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-20b"

# ---------------------------------------------------------------------------
# Embedding models (run locally, free, no API calls)
# ---------------------------------------------------------------------------
EMBEDDING_MODELS = [
    "all-MiniLM-L6-v2",   # small, fast — good default
    "all-mpnet-base-v2",  # larger, generally higher quality, slower
]

# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------
CHUNK_SIZE_FIXED = 500
CHUNK_OVERLAP_FIXED = 50
SENTENCES_PER_CHUNK = 4  # for the sentence-based strategy

CHUNKING_STRATEGIES = ["fixed", "sentence"]

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
RETRIEVAL_METHODS = ["vector", "bm25", "hybrid"]
TOP_K_VALUES = [3, 5]

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------------------------------------------------------------------------
# Prompt variants
# ---------------------------------------------------------------------------
PROMPT_TEMPLATES = {
    "strict": (
        "You are answering questions using ONLY the provided context. "
        "If the answer is not contained in the context, say "
        "\"I cannot find this in the provided documents.\" "
        "Do not use any outside knowledge.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    ),
    "loose": (
        "Use the following context to help answer the question as helpfully "
        "as possible. You may use general knowledge to clarify if needed.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    ),
}

# ---------------------------------------------------------------------------
# Curated experiment configurations
# ---------------------------------------------------------------------------
# A FULL factorial grid (2 chunking x 2 embeddings x 3 retrieval x 2 rerank
# x 2 top_k x 2 prompts) would be 96 configurations -- too many LLM calls for
# Groq's free tier rate limits and more than a 12-15 question ground-truth
# set can meaningfully support. Instead, this is a CURATED set that isolates
# one variable at a time against a fixed baseline, which is both more
# rate-limit-friendly and easier to interpret and defend than a giant grid.
#
# Each dict fully specifies one pipeline "recipe".
CURATED_CONFIGS = [
    {
        "name": "baseline",
        "chunking": "fixed",
        "embedding_model": "all-MiniLM-L6-v2",
        "retrieval_method": "vector",
        "top_k": 5,
        "rerank": False,
        "prompt": "loose",
    },
    {
        "name": "sentence_chunking",
        "chunking": "sentence",
        "embedding_model": "all-MiniLM-L6-v2",
        "retrieval_method": "vector",
        "top_k": 5,
        "rerank": False,
        "prompt": "loose",
    },
    {
        "name": "larger_embedding_model",
        "chunking": "fixed",
        "embedding_model": "all-mpnet-base-v2",
        "retrieval_method": "vector",
        "top_k": 5,
        "rerank": False,
        "prompt": "loose",
    },
    {
        "name": "bm25_only",
        "chunking": "fixed",
        "embedding_model": "all-MiniLM-L6-v2",
        "retrieval_method": "bm25",
        "top_k": 5,
        "rerank": False,
        "prompt": "loose",
    },
    {
        "name": "hybrid_retrieval",
        "chunking": "fixed",
        "embedding_model": "all-MiniLM-L6-v2",
        "retrieval_method": "hybrid",
        "top_k": 5,
        "rerank": False,
        "prompt": "loose",
    },
    {
        "name": "hybrid_plus_rerank",
        "chunking": "sentence",
        "embedding_model": "all-MiniLM-L6-v2",
        "retrieval_method": "hybrid",
        "top_k": 5,
        "rerank": True,
        "prompt": "loose",
    },
    {
        "name": "hybrid_rerank_strict_prompt",
        "chunking": "sentence",
        "embedding_model": "all-MiniLM-L6-v2",
        "retrieval_method": "hybrid",
        "top_k": 5,
        "rerank": True,
        "prompt": "strict",
    },
    {
        "name": "hybrid_rerank_strict_topk3",
        "chunking": "sentence",
        "embedding_model": "all-MiniLM-L6-v2",
        "retrieval_method": "hybrid",
        "top_k": 3,
        "rerank": True,
        "prompt": "strict",
    },
]

"""
Ties ingestion, embedding, retrieval, reranking, and generation together into
one callable pipeline for a single (question, configuration) pair.
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.ingest import get_chunks, load_pdf_texts
from src.embed_store import build_or_get_collection
from src.retrieval import retrieve
from src.reranker import rerank
from src.generate import generate_answer

_chunks_cache = {}
_collection_cache = {}


def _get_chunks_for_strategy(strategy: str, documents):
    if strategy not in _chunks_cache:
        _chunks_cache[strategy] = get_chunks(strategy, documents)
    return _chunks_cache[strategy]


def _get_collection(strategy: str, embedding_model: str, chunks):
    key = (strategy, embedding_model)
    if key not in _collection_cache:
        _collection_cache[key] = build_or_get_collection(chunks, strategy, embedding_model)
    return _collection_cache[key]


def run_pipeline(question: str, config: dict, documents=None):
    """
    Runs one full RAG pass for a single question under a single configuration.
    Returns a dict with the answer, the retrieved context texts, and timing
    info -- everything run_experiments.py and evaluate_ragas.py need.
    """
    if documents is None:
        documents = load_pdf_texts()

    chunking = config["chunking"]
    embedding_model = config["embedding_model"]
    retrieval_method = config["retrieval_method"]
    top_k = config["top_k"]
    do_rerank = config["rerank"]
    prompt_style = config["prompt"]

    chunks = _get_chunks_for_strategy(chunking, documents)
    collection = _get_collection(chunking, embedding_model, chunks)
    cache_key = f"{chunking}__{embedding_model}"

    t0 = time.time()
    # Retrieve more candidates than top_k when we're going to rerank, so
    # reranking has a meaningful pool to re-score from.
    retrieve_k = top_k * 3 if do_rerank else top_k
    hits = retrieve(retrieval_method, question, collection, chunks,
                     embedding_model, cache_key, k=retrieve_k)

    if do_rerank:
        hits = rerank(question, hits, top_n=top_k)
    t1 = time.time()

    context_texts = [h["text"] for h in hits]
    answer = generate_answer(question, context_texts, prompt_style)
    t2 = time.time()

    return {
        "question": question,
        "answer": answer,
        "contexts": context_texts,
        "retrieval_time_sec": round(t1 - t0, 3),
        "generation_time_sec": round(t2 - t1, 3),
        "total_time_sec": round(t2 - t0, 3),
    }

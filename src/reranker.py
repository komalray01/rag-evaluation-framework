"""
Reranking: given a query and a list of candidate chunks from the initial
retrieval pass, re-scores each (query, chunk) pair with a cross-encoder --
a slower but more precise model than the initial retriever, applied only to
the small candidate set. Pushes the truly best-matching chunks to the top.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import RERANKER_MODEL

_reranker_cache = {}


def get_reranker():
    from sentence_transformers import CrossEncoder

    if "model" not in _reranker_cache:
        print(f"Loading reranker model: {RERANKER_MODEL} (first load downloads it)")
        _reranker_cache["model"] = CrossEncoder(RERANKER_MODEL)
    return _reranker_cache["model"]


def rerank(query: str, hits: list, top_n: int = None):
    """
    hits: list of {"chunk_id", "text", "metadata"} dicts from retrieval.
    Returns the same hits, re-sorted by cross-encoder relevance score,
    optionally truncated to top_n.
    """
    if not hits:
        return hits

    reranker = get_reranker()
    pairs = [(query, h["text"]) for h in hits]
    scores = reranker.predict(pairs)

    scored_hits = list(zip(hits, scores))
    scored_hits.sort(key=lambda x: x[1], reverse=True)

    reranked = [h for h, _ in scored_hits]
    if top_n is not None:
        reranked = reranked[:top_n]
    return reranked

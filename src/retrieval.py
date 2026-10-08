"""
Three retrieval methods:
  - vector: semantic similarity search via ChromaDB
  - bm25: classic keyword-based search
  - hybrid: combines both using Reciprocal Rank Fusion (RRF), a simple,
    well-established way to merge two ranked lists without needing to
    calibrate their scores against each other
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.embed_store import get_embedding_model

_bm25_cache = {}


def _tokenize(text: str):
    return text.lower().split()


def get_bm25_index(chunks, cache_key: str):
    """Builds (or reuses a cached) BM25 index over a list of chunk dicts."""
    from rank_bm25 import BM25Okapi

    if cache_key not in _bm25_cache:
        tokenized = [_tokenize(c["text"]) for c in chunks]
        bm25 = BM25Okapi(tokenized)
        _bm25_cache[cache_key] = (bm25, chunks)
    return _bm25_cache[cache_key]


def vector_search(query: str, collection, embedding_model_name: str, k: int):
    model = get_embedding_model(embedding_model_name)
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=k)

    hits = []
    for doc_text, meta, chunk_id in zip(
        results["documents"][0], results["metadatas"][0], results["ids"][0]
    ):
        hits.append({"chunk_id": chunk_id, "text": doc_text, "metadata": meta})
    return hits


def bm25_search(query: str, chunks, cache_key: str, k: int):
    bm25, indexed_chunks = get_bm25_index(chunks, cache_key)
    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    hits = []
    for idx in ranked_indices:
        c = indexed_chunks[idx]
        hits.append({"chunk_id": c["chunk_id"], "text": c["text"],
                      "metadata": {"doc_id": c["doc_id"], "strategy": c["strategy"]}})
    return hits


def hybrid_search(query: str, collection, chunks, embedding_model_name: str,
                   cache_key: str, k: int):
    """
    Reciprocal Rank Fusion: each chunk's final score is the sum of
    1 / (60 + rank) across both retrieval methods' rankings. A chunk that
    ranks well in EITHER method gets a boost; a chunk that ranks well in
    BOTH gets the biggest boost. 60 is RRF's standard constant from the
    original paper (Cormack et al.) that prevents very top-ranked items
    from dominating disproportionately.
    """
    # Pull more candidates than k from each method so fusion has room to work
    vector_hits = vector_search(query, collection, embedding_model_name, k=k * 3)
    bm25_hits = bm25_search(query, chunks, cache_key, k=k * 3)

    scores = {}
    all_hits_by_id = {}

    for rank, hit in enumerate(vector_hits):
        scores[hit["chunk_id"]] = scores.get(hit["chunk_id"], 0) + 1 / (60 + rank)
        all_hits_by_id[hit["chunk_id"]] = hit

    for rank, hit in enumerate(bm25_hits):
        scores[hit["chunk_id"]] = scores.get(hit["chunk_id"], 0) + 1 / (60 + rank)
        all_hits_by_id[hit["chunk_id"]] = hit

    ranked_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)[:k]
    return [all_hits_by_id[cid] for cid in ranked_ids]


def retrieve(method: str, query: str, collection, chunks, embedding_model_name: str,
             cache_key: str, k: int):
    if method == "vector":
        return vector_search(query, collection, embedding_model_name, k)
    elif method == "bm25":
        return bm25_search(query, chunks, cache_key, k)
    elif method == "hybrid":
        return hybrid_search(query, collection, chunks, embedding_model_name, cache_key, k)
    else:
        raise ValueError(f"Unknown retrieval method: {method}")

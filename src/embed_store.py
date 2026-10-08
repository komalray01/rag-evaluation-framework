"""
Embeds chunks with a chosen sentence-transformers model and stores them in a
ChromaDB collection. One collection per (chunking_strategy, embedding_model)
combination, so different configurations don't collide.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import CHROMA_DB_DIR

_embedding_model_cache = {}


def get_embedding_model(model_name: str):
    """Cached loader so we don't reload the same model repeatedly."""
    from sentence_transformers import SentenceTransformer

    if model_name not in _embedding_model_cache:
        print(f"Loading embedding model: {model_name} (first load downloads it)")
        _embedding_model_cache[model_name] = SentenceTransformer(model_name)
    return _embedding_model_cache[model_name]


def collection_name(chunking_strategy: str, embedding_model: str) -> str:
    # Chroma collection names can't contain slashes/dots freely; sanitize.
    safe_model = embedding_model.replace("/", "_").replace(".", "_")
    return f"{chunking_strategy}__{safe_model}"


def build_or_get_collection(chunks, chunking_strategy: str, embedding_model_name: str):
    """
    Builds a ChromaDB collection for this (chunking, embedding) pair if it
    doesn't already exist, embedding all chunks. If it already exists
    (from a previous run), reuses it rather than re-embedding everything.
    """
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    name = collection_name(chunking_strategy, embedding_model_name)

    existing = [c.name for c in client.list_collections()]
    if name in existing:
        print(f"Reusing existing collection: {name}")
        return client.get_collection(name)

    print(f"Building new collection: {name} ({len(chunks)} chunks)")
    collection = client.create_collection(name)
    model = get_embedding_model(embedding_model_name)

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [{"doc_id": c["doc_id"], "strategy": c["strategy"]} for c in chunks]

    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    return collection

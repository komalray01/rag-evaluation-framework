"""
Loads PDFs from data/documents/ and splits them using two different
chunking strategies. Every chunk keeps metadata about which document and
which chunking strategy produced it, so results can be traced back.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    DOCUMENTS_DIR, CHUNK_SIZE_FIXED, CHUNK_OVERLAP_FIXED, SENTENCES_PER_CHUNK,
)


def load_pdf_texts():
    """
    Loads every PDF in data/documents/ and returns a list of
    {"doc_id": str, "text": str} dicts, one per PDF.
    """
    from langchain_community.document_loaders import PyPDFLoader

    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in {DOCUMENTS_DIR}. Add your document set there "
            f"first — see README.md for a suggested free, real document set."
        )

    documents = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        full_text = "\n".join(p.page_content for p in pages)
        documents.append({"doc_id": pdf_path.stem, "text": full_text})
        print(f"Loaded {pdf_path.name}: {len(pages)} pages, "
              f"{len(full_text):,} characters")

    return documents


def chunk_fixed(documents):
    """Fixed-size character chunking with overlap, via LangChain's splitter."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_FIXED,
        chunk_overlap=CHUNK_OVERLAP_FIXED,
    )

    chunks = []
    for doc in documents:
        pieces = splitter.split_text(doc["text"])
        for i, piece in enumerate(pieces):
            chunks.append({
                "chunk_id": f"{doc['doc_id']}_fixed_{i}",
                "doc_id": doc["doc_id"],
                "text": piece,
                "strategy": "fixed",
            })
    return chunks


def chunk_sentence(documents):
    """
    Sentence-aware chunking: splits on sentence boundaries, then groups
    SENTENCES_PER_CHUNK sentences together. Avoids cutting a sentence in
    half, unlike pure fixed-size chunking.
    """
    import re

    # Protect common abbreviations from being mistaken for sentence endings
    # (a plain period-based split incorrectly breaks on "Mr.", "e.g.", "U.S.",
    # etc. — all realistic in technical whitepapers). We temporarily swap the
    # period in known abbreviations for a placeholder, split normally, then
    # restore it.
    ABBREVIATIONS = ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.",
                      "vs.", "etc.", "e.g.", "i.e.", "U.S.", "U.K.", "approx."]

    chunks = []
    for doc in documents:
        protected_text = doc["text"]
        for abbr in ABBREVIATIONS:
            protected_text = protected_text.replace(abbr, abbr.replace(".", "<PERIOD>"))

        sentences = re.split(r"(?<=[.?!])\s+(?=[A-Z])", protected_text)
        sentences = [s.replace("<PERIOD>", ".").strip() for s in sentences if s.strip()]

        for i in range(0, len(sentences), SENTENCES_PER_CHUNK):
            group = sentences[i: i + SENTENCES_PER_CHUNK]
            text = " ".join(group)
            chunks.append({
                "chunk_id": f"{doc['doc_id']}_sentence_{i // SENTENCES_PER_CHUNK}",
                "doc_id": doc["doc_id"],
                "text": text,
                "strategy": "sentence",
            })
    return chunks


def get_chunks(strategy: str, documents=None):
    """Entry point: get_chunks('fixed') or get_chunks('sentence')."""
    if documents is None:
        documents = load_pdf_texts()

    if strategy == "fixed":
        return chunk_fixed(documents)
    elif strategy == "sentence":
        return chunk_sentence(documents)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")


if __name__ == "__main__":
    docs = load_pdf_texts()
    for strategy in ["fixed", "sentence"]:
        chunks = get_chunks(strategy, docs)
        print(f"\n{strategy}: {len(chunks)} chunks")
        print(f"  Example: {chunks[0]['text'][:150]}...")

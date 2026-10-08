# RAG Evaluation & Optimization Framework

Instead of building one fixed RAG chatbot, this project builds a **testing harness**
that systematically compares chunking strategies, embedding models, retrieval methods,
reranking, and prompt styles — then measures each combination on retrieval quality,
faithfulness, answer relevance, and latency. The goal is a results table that shows
*why* a RAG configuration succeeds or fails, not just a single "it works" demo.

**100% free stack:** Groq (free-tier LLM API, no credit card), sentence-transformers
(local, free embeddings), ChromaDB (local vector store), rank_bm25 (keyword search),
a local cross-encoder reranker, and RAGAS for evaluation.

## What you need to fill in before this runs

This code is complete and structured, but three things are deliberately left for you:

1. **A Groq API key** — free, no credit card, at https://console.groq.com. Copy
   `.env.example` to `.env` and paste your key in.
2. **Your document set** — put 5 or so PDFs into `data/documents/`. Suggested free,
   real, coherent set: the **AWS Well-Architected Framework whitepapers** (free,
   official, download from AWS's site — search "AWS Well-Architected Framework
   whitepaper pdf"). Any focused PDF set works; this one ties in well if you're also
   using AWS elsewhere on your resume.
3. **Your ground-truth question/answer pairs** — `data/ground_truth.json` currently has
   2 placeholder examples clearly marked `EXAMPLE - REPLACE ME`. Read your actual
   downloaded PDFs and write 12-15 real question/answer pairs grounded in that real
   text. **Do not skip this or leave the placeholders in** — this file is your entire
   evaluation methodology; questions not genuinely checked against your own source
   documents will produce scores you can't defend in an interview.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # then edit .env and paste your real Groq key
```

## Running the pipeline

```bash
# 1. (Optional) sanity-check chunking works on your PDFs
python src/ingest.py

# 2. Run every curated configuration against every ground-truth question
python src/run_experiments.py

# 3. Score everything with RAGAS and build the final comparison table
python src/evaluate_ragas.py

# 4. View the results
streamlit run dashboard/app.py
```

`run_experiments.py` will take a while and make many Groq API calls — one per
(configuration × question). With 8 curated configurations and, say, 15 questions,
that's 120 LLM calls. Groq's free tier is rate-limited, not billed, but if you hit a
rate limit mid-run, just re-run — configurations use cached embeddings/collections so
it won't redo everything from scratch, though it will redo the LLM calls.

## Why a curated set of 8 configurations, not a full grid

A full factorial grid (2 chunking × 2 embeddings × 3 retrieval methods × 2 rerank
on/off × 2 top_k × 2 prompts) is 96 combinations — too many free-tier API calls, and
more than a 12-15 question ground-truth set can meaningfully distinguish between. The
8 configurations in `src/config.py` (`CURATED_CONFIGS`) instead isolate one variable at
a time against a fixed baseline, moving toward a best-guess "everything combined"
configuration at the end. This is both more interpretable and much more
rate-limit-friendly — and it's a genuinely defensible experimental design choice to
explain in an interview, not just a shortcut.

## Project structure

```
rag-evaluation-framework/
├── README.md
├── requirements.txt
├── .env.example              # copy to .env and fill in your Groq key
├── .gitignore
├── data/
│   ├── documents/             # put your PDFs here
│   ├── ground_truth.json      # YOU write real Q&A pairs here
│   └── chroma_db/              # auto-created vector store
├── src/
│   ├── config.py               # all settings + the 8 curated configurations
│   ├── ingest.py                # PDF loading + two chunking strategies
│   ├── embed_store.py           # embedding + ChromaDB storage
│   ├── retrieval.py             # vector / BM25 / hybrid (Reciprocal Rank Fusion)
│   ├── reranker.py              # cross-encoder reranking
│   ├── generate.py              # Groq LLM calls + prompt templates
│   ├── pipeline.py              # ties one config + one question together end to end
│   ├── run_experiments.py       # outer loop: every config x every question
│   └── evaluate_ragas.py        # RAGAS scoring + final comparison table
├── dashboard/
│   └── app.py                    # Streamlit results dashboard
└── results/                       # raw_results.csv, scored_results.csv, comparison_table.csv
```

## Key design decisions 

- **Why retrieval and generation are evaluated separately:** bad retrieval can't be
  fixed by a better LLM, and vice versa — faithfulness, answer relevance,
  context_precision, and context_recall are kept as four separate numbers on purpose,
  never combined into one score, so you can tell *where* a configuration is weak.
- **Why hybrid retrieval uses Reciprocal Rank Fusion (RRF):** it combines two ranked
  lists (vector + BM25) without needing to calibrate their raw scores against each
  other, which is a real problem since vector similarity and BM25 scores live on
  completely different scales.
- **Why a curated config set instead of a full grid:** explained above — rate limits
  and interpretability, not laziness.
- **Why RAGAS uses Groq as the judge LLM instead of OpenAI (its default):** to keep the
  entire project free — this required wrapping Groq through LangChain's `ChatGroq` and
  RAGAS's `LangchainLLMWrapper`.
- **Why a hand-written ground truth set matters even though RAGAS's metrics are mostly
  LLM-judged:** LLM-as-judge scoring can be inconsistent — having your own real answer
  key lets you spot-check whether RAGAS's judgments actually line up with a human's.

## A note on library versions

RAGAS's API changes fairly often. `requirements.txt` pins `ragas==0.1.9` specifically
to reduce the risk of `evaluate_ragas.py` breaking against a newer, incompatible API.
If you intentionally upgrade ragas later, check their current docs for how to pass a
custom LLM/embeddings and build the evaluation dataset — the underlying concepts (four
metrics, question/answer/contexts/ground_truth) will still be the same even if some
function or class names have moved.

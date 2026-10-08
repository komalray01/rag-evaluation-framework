"""
Scores results/raw_results.csv using RAGAS metrics:
  - faithfulness: is the answer grounded in the retrieved context (not hallucinated)?
  - answer_relevancy: does the answer actually address the question asked?
  - context_precision: were the retrieved chunks actually relevant?
  - context_recall: did retrieval find everything needed to answer correctly?

RAGAS's default judge LLM is OpenAI. Since this project is 100% free, we
instead wrap Groq as the judge LLM via LangChain's ChatGroq integration.

NOTE ON LIBRARY VERSIONS: RAGAS's API has changed across versions. This
script targets the ragas==0.1.x API pinned in requirements.txt. If you've
installed a newer ragas and this breaks, check the RAGAS docs for the
current way to (a) pass a custom LLM and (b) build the evaluation dataset --
the core ideas (four metrics, question/answer/contexts/ground_truth) will
be the same even if some function names have moved.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import GROQ_API_KEY, GROQ_MODEL, RESULTS_DIR

RAW_RESULTS_PATH = RESULTS_DIR / "raw_results.csv"
SCORED_RESULTS_PATH = RESULTS_DIR / "scored_results.csv"
COMPARISON_TABLE_PATH = RESULTS_DIR / "comparison_table.csv"


def build_ragas_judge_llm():
    """Wraps Groq as the LLM RAGAS uses to judge faithfulness/relevance."""
    from langchain_groq import ChatGroq
    from ragas.llms import LangchainLLMWrapper

    if not GROQ_API_KEY:
        raise EnvironmentError("GROQ_API_KEY is not set — see .env.example.")

    chat = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=GROQ_MODEL, temperature=0.0)
    return LangchainLLMWrapper(chat)


def build_ragas_embeddings():
    """Local, free embeddings for RAGAS's context_precision/recall metrics."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return LangchainEmbeddingsWrapper(hf_embeddings)


def main():
    if not RAW_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"{RAW_RESULTS_PATH} not found — run src/run_experiments.py first."
        )

    raw_df = pd.read_csv(RAW_RESULTS_PATH)
    raw_df["contexts"] = raw_df["contexts"].apply(json.loads)  # back to list of str

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness, answer_relevancy, context_precision, context_recall,
    )

    judge_llm = build_ragas_judge_llm()
    embeddings = build_ragas_embeddings()

    all_scored = []

    # Score per-configuration so one bad row doesn't block every other config,
    # and so progress is visible as it goes rather than one long silent wait.
    for config_name, group in raw_df.groupby("config_name"):
        print(f"\nScoring configuration: {config_name} ({len(group)} questions)...")

        eval_dataset = Dataset.from_dict({
            "question": group["question"].tolist(),
            "answer": group["generated_answer"].tolist(),
            "contexts": group["contexts"].tolist(),
            "ground_truth": group["ground_truth_answer"].tolist(),
        })

        try:
            result = evaluate(
                eval_dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                llm=judge_llm,
                embeddings=embeddings,
            )
            scored_df = result.to_pandas()
        except Exception as e:
            print(f"  ERROR scoring {config_name}: {e}")
            print("  See the NOTE ON LIBRARY VERSIONS at the top of this file.")
            continue

        scored_df["config_name"] = config_name
        scored_df["retrieval_time_sec"] = group["retrieval_time_sec"].values
        scored_df["generation_time_sec"] = group["generation_time_sec"].values
        scored_df["total_time_sec"] = group["total_time_sec"].values
        all_scored.append(scored_df)

    if not all_scored:
        print("\nNo configurations scored successfully. See errors above.")
        return

    full_scored = pd.concat(all_scored, ignore_index=True)
    full_scored.to_csv(SCORED_RESULTS_PATH, index=False)
    print(f"\nSaved per-question scores to {SCORED_RESULTS_PATH}")

    # Build the final comparison table: one row per configuration, averaged.
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    time_cols = ["retrieval_time_sec", "generation_time_sec", "total_time_sec"]
    available_metric_cols = [c for c in metric_cols if c in full_scored.columns]

    comparison = (
        full_scored.groupby("config_name")[available_metric_cols + time_cols]
        .mean()
        .round(3)
        .sort_values("faithfulness", ascending=False)
    )
    comparison.to_csv(COMPARISON_TABLE_PATH)
    print(f"Saved final comparison table to {COMPARISON_TABLE_PATH}\n")
    print(comparison)


if __name__ == "__main__":
    main()

"""
Runs every configuration in config.CURATED_CONFIGS against every question in
data/ground_truth.json, and saves the raw per-question results to
results/raw_results.csv. evaluate_ragas.py then scores this file.

This is the "outer loop = configurations, inner loop = questions" structure:
for each configuration, every question gets run through that exact pipeline,
so every configuration is tested against the identical, fixed question set.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import CURATED_CONFIGS, GROUND_TRUTH_PATH, RESULTS_DIR
from src.ingest import load_pdf_texts
from src.pipeline import run_pipeline

RAW_RESULTS_PATH = RESULTS_DIR / "raw_results.csv"


def load_ground_truth():
    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(
            f"{GROUND_TRUTH_PATH} not found. Fill in data/ground_truth.json "
            f"with your own hand-written question/answer pairs first -- "
            f"see README.md and the template already in that file."
        )
    with open(GROUND_TRUTH_PATH) as f:
        return json.load(f)


def main():
    ground_truth = load_ground_truth()
    print(f"Loaded {len(ground_truth)} ground-truth question/answer pairs.")

    documents = load_pdf_texts()

    all_rows = []
    total_runs = len(CURATED_CONFIGS) * len(ground_truth)
    run_count = 0

    for config in CURATED_CONFIGS:
        print(f"\n{'='*60}")
        print(f"Running configuration: {config['name']}")
        print(f"{'='*60}")

        for item in ground_truth:
            run_count += 1
            question = item["question"]
            print(f"[{run_count}/{total_runs}] {config['name']} | {question[:60]}...")

            try:
                result = run_pipeline(question, config, documents=documents)
            except Exception as e:
                print(f"  ERROR on this question/config: {e}")
                print("  Skipping and continuing with the rest of the run.")
                continue

            all_rows.append({
                "config_name": config["name"],
                "question": question,
                "ground_truth_answer": item["ground_truth_answer"],
                "generated_answer": result["answer"],
                "contexts": json.dumps(result["contexts"]),  # store as JSON string in CSV
                "retrieval_time_sec": result["retrieval_time_sec"],
                "generation_time_sec": result["generation_time_sec"],
                "total_time_sec": result["total_time_sec"],
            })

    df = pd.DataFrame(all_rows)
    df.to_csv(RAW_RESULTS_PATH, index=False)
    print(f"\nSaved {len(df)} raw results to {RAW_RESULTS_PATH}")
    print("Next step: python src/evaluate_ragas.py")


if __name__ == "__main__":
    main()

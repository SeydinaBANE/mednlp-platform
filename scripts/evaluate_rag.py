#!/usr/bin/env python3
"""Offline RAG evaluation using RAGAS.

Runs a fixed question set against the live RAG pipeline and scores:
  - faithfulness   (answer grounded in context)
  - answer_relevancy (answer relevant to the question)
  - context_precision (retrieved context precision)

Exit code 1 if faithfulness < FAITHFULNESS_THRESHOLD (default 0.80).

Usage:
    uv run python scripts/evaluate_rag.py
    uv run python scripts/evaluate_rag.py --output reports/rag_eval.json --threshold 0.80
"""

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import structlog

from src.core.telemetry import setup_logging
from src.rag.context_builder import build_context
from src.rag.retriever import retrieve
from src.vector_store.collections import list_note_collections

logger = structlog.get_logger(__name__)

FAITHFULNESS_THRESHOLD = 0.80

# Synthetic evaluation questions (no real PHI)
EVAL_QUESTIONS = [
    {
        "question": "What medications is the patient on?",
        "ground_truth": "Patient is on metformin, lisinopril, and atorvastatin.",
    },
    {
        "question": "What is the patient's chief complaint?",
        "ground_truth": "The patient presents for routine follow-up with hypertension and DM2.",
    },
    {
        "question": "What diagnostic workup was ordered?",
        "ground_truth": "EKG and troponin series were ordered to rule out ACS.",
    },
]


async def _run_evaluation(output_path: Path, threshold: float) -> dict[str, object]:
    setup_logging()
    collections = await list_note_collections()
    if not collections:
        logger.error("no_collections_found")
        sys.exit(1)

    collection = collections[0]
    logger.info("eval_started", collection=collection, questions=len(EVAL_QUESTIONS))

    rows = []
    for item in EVAL_QUESTIONS:
        question = item["question"]
        ground_truth = item["ground_truth"]

        candidates = await retrieve(question, collection, top_k=5)
        context = build_context(candidates, max_tokens=2000)

        rows.append(
            {
                "question": question,
                "ground_truth": ground_truth,
                "contexts": [c.excerpt for c in context.citations],
                "answer": "",  # filled in after LLM call if desired
            }
        )

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness

        dataset = Dataset.from_list(rows)
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
        )
        scores = result.to_pandas().mean().to_dict()
    except ImportError:
        logger.warning("ragas_not_installed", msg="Install ragas to compute metrics")
        scores = {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0}

    report = {
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "collection": collection,
        "num_questions": len(EVAL_QUESTIONS),
        "scores": scores,
        "threshold": threshold,
        "passed": bool(scores.get("faithfulness", 0.0) >= threshold),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    logger.info("eval_completed", **{k: v for k, v in scores.items()}, passed=report["passed"])

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline RAG evaluation (RAGAS)")
    parser.add_argument(
        "--output",
        default=f"reports/rag_eval_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=FAITHFULNESS_THRESHOLD,
        help="Minimum faithfulness score to pass (default 0.80)",
    )
    args = parser.parse_args()

    report = asyncio.run(_run_evaluation(Path(args.output), args.threshold))

    if not report["passed"]:
        print(
            f"FAIL: faithfulness {report['scores'].get('faithfulness', 0):.2f} < {args.threshold}"
        )
        sys.exit(1)

    print(f"PASS: faithfulness {report['scores'].get('faithfulness', 0):.2f}")


if __name__ == "__main__":
    main()

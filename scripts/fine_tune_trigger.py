#!/usr/bin/env python3
"""Trigger a LoRA fine-tuning job on Vertex AI.

Usage:
    uv run python scripts/fine_tune_trigger.py --task icd10
    uv run python scripts/fine_tune_trigger.py --task triage --sync
"""

import argparse

import structlog

from src.core.telemetry import setup_logging
from src.fine_tuning.vertex_job import VertexJobConfig, submit_fine_tune_job

logger = structlog.get_logger(__name__)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Trigger fine-tuning on Vertex AI")
    parser.add_argument(
        "--task",
        required=True,
        choices=["icd10", "triage"],
        help="Fine-tuning task",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Wait for the job to complete (blocks until done)",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="GCS output directory (defaults to gcs_bucket_artifacts/fine_tuning/<task>)",
    )
    args = parser.parse_args()

    config = VertexJobConfig(
        task=args.task,
        sync=args.sync,
        base_output_dir=args.output_dir,
    )

    logger.info("submitting_fine_tune_job", task=args.task, sync=args.sync)
    job_name = submit_fine_tune_job(config)
    logger.info("fine_tune_job_submitted", job_name=job_name)
    print(f"Job submitted: {job_name}")


if __name__ == "__main__":
    main()

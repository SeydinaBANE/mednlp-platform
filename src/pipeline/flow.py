"""Prefect flow — NLP processing pipeline for a single clinical note."""

import structlog
from prefect import flow

from src.core.exceptions import PipelineError
from src.core.telemetry import PIPELINE_NOTES_TOTAL
from src.ingestion.schemas import NoteRecord
from src.pipeline.schemas import PipelineContext
from src.pipeline.stages.deidentifier import deidentify
from src.pipeline.stages.ner import extract_entities
from src.pipeline.stages.quality_gate import quality_gate
from src.pipeline.stages.segmenter import segment
from src.pipeline.stages.vectorizer import vectorize

logger = structlog.get_logger(__name__)


@flow(name="mednlp-note-pipeline", log_prints=False)
async def process_note(note: NoteRecord, skip_vectorizer: bool = False) -> PipelineContext:
    """Full NLP pipeline: segment → deidentify → NER → quality gate → vectorize.

    Args:
        note: the ingested clinical note
        skip_vectorizer: set True in dry-run / evaluation mode
    """
    ctx = PipelineContext(note=note)
    log = logger.bind(note_id=note.note_id, patient_id=note.patient_id)
    log.info("pipeline_started")

    try:
        ctx = await segment(ctx)
        ctx = await deidentify(ctx)
        ctx = await extract_entities(ctx)
        ctx = await quality_gate(ctx)
        if not skip_vectorizer:
            ctx = await vectorize(ctx)

        PIPELINE_NOTES_TOTAL.labels(status="success").inc()
        log.info("pipeline_completed", vector_indexed=ctx.vector_indexed)

    except PipelineError as exc:
        ctx.errors.append(str(exc))
        PIPELINE_NOTES_TOTAL.labels(status="failure").inc()
        log.error("pipeline_failed", stage=exc.stage, reason=exc.reason)
        raise

    return ctx


async def run_batch(notes: list[NoteRecord]) -> dict[str, int]:
    """Process a batch of notes concurrently. Returns counts by status."""
    import asyncio

    results = await asyncio.gather(
        *[process_note(note) for note in notes],
        return_exceptions=True,
    )

    counts = {"success": 0, "failure": 0}
    for r in results:
        if isinstance(r, Exception):
            counts["failure"] += 1
        else:
            counts["success"] += 1

    logger.info("batch_pipeline_done", **counts, total=len(notes))
    return counts

"""Vectorization stage — embed processed note and upsert to Qdrant."""

import time

import structlog
from prefect import task

from src.core.exceptions import PipelineError
from src.core.telemetry import PIPELINE_STAGE_LATENCY
from src.embeddings.biomedbert_embedder import BiomedBertEmbedder
from src.pipeline.schemas import PipelineContext
from src.vector_store.indexer import upsert_notes

logger = structlog.get_logger(__name__)

_embedder = BiomedBertEmbedder()


@task(name="vectorizer", retries=2, retry_delay_seconds=10)
async def vectorize(ctx: PipelineContext) -> PipelineContext:
    """Embed the processed note text and upsert to Qdrant."""
    start = time.perf_counter()
    text = ctx.processed_text

    try:
        vector = await _embedder.embed_one(text)
        payload = {
            "patient_id": ctx.note.patient_id,
            "note_type": ctx.note.note_type,
            "authored_at": ctx.note.authored_at.isoformat(),
            "source": ctx.note.source,
            "raw_text": text[:2000],  # store excerpt for context building
            "entity_types": list({e.label for e in ctx.entities}),
        }
        await upsert_notes(
            note_ids=[ctx.note.note_id],
            vectors=[vector],
            payloads=[payload],
            model_name=_embedder.model_name,
            model_version=_embedder.model_version,
        )
        ctx.vector_indexed = True
        elapsed = time.perf_counter() - start
        PIPELINE_STAGE_LATENCY.labels(stage="vectorizer", status="success").observe(elapsed)
        logger.info("vectorizer_done", note_id=ctx.note.note_id)

    except Exception as exc:
        PIPELINE_STAGE_LATENCY.labels(stage="vectorizer", status="failure").observe(0)
        raise PipelineError("vectorizer", str(exc)) from exc

    return ctx

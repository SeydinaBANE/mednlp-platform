"""Celery tasks — pipeline, indexing, fine-tune trigger."""

import asyncio
from typing import Any

import structlog
from celery import Celery

from src.core.config import get_settings

logger = structlog.get_logger(__name__)


def _make_celery() -> Celery:
    settings = get_settings()
    app = Celery(
        "mednlp",
        broker=settings.redis_url,
        backend=settings.redis_url,
    )
    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
    )
    return app


celery_app = _make_celery()


@celery_app.task(name="tasks.process_note", bind=True, max_retries=3)
def process_note_task(self: Any, note_dict: dict[str, Any]) -> dict[str, Any]:
    """Run the full NLP pipeline for a single note."""
    from src.ingestion.schemas import NoteRecord
    from src.pipeline.flow import process_note

    try:
        note = NoteRecord(**note_dict)
        ctx = asyncio.run(process_note(note))
        return {
            "note_id": ctx.note.note_id,
            "vector_indexed": ctx.vector_indexed,
            "n_entities": len(ctx.entities),
            "n_segments": len(ctx.segments),
            "quality_passed": ctx.quality.passed if ctx.quality else False,
        }
    except Exception as exc:
        logger.error("process_note_task_failed", note_id=note_dict.get("note_id"), error=str(exc))
        raise self.retry(exc=exc, countdown=2**self.request.retries * 10) from exc


@celery_app.task(name="tasks.backfill_index", bind=True, max_retries=2)
def backfill_index_task(self: Any, model_name: str, model_version: str) -> dict[str, Any]:
    """Re-vectorise notes with a new embedding model (backfill)."""
    logger.info("backfill_started", model=model_name, version=model_version)
    # Actual implementation deferred — requires pagination over all notes in DB
    return {"status": "started", "model": model_name, "version": model_version}


@celery_app.task(name="tasks.trigger_fine_tune", bind=True, max_retries=1)
def trigger_fine_tune_task(self: Any, task_name: str) -> dict[str, Any]:
    """Submit a fine-tuning job to Vertex AI."""
    from src.fine_tuning.vertex_job import VertexJobConfig, submit_fine_tune_job

    try:
        config = VertexJobConfig(task=task_name, sync=False)
        job_name = submit_fine_tune_job(config)
        logger.info("fine_tune_submitted", task=task_name, job=job_name)
        return {"status": "submitted", "task": task_name, "job_name": job_name}
    except Exception as exc:
        logger.error("fine_tune_trigger_failed", task=task_name, error=str(exc))
        raise self.retry(exc=exc, countdown=30) from exc

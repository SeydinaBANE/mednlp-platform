"""Batch async upsert into Qdrant with backpressure and Prometheus instrumentation."""

import asyncio
import uuid
from typing import Any

import structlog
from qdrant_client.models import PointStruct

from src.core.exceptions import VectorStoreError
from src.core.telemetry import EMBEDDING_INFERENCE_LATENCY, PIPELINE_NOTES_TOTAL
from src.vector_store.client import get_qdrant_client
from src.vector_store.collections import ensure_collection

logger = structlog.get_logger(__name__)

_DEFAULT_BATCH_SIZE = 64
_MAX_CONCURRENT_BATCHES = 4


async def upsert_notes(
    *,
    note_ids: list[str],
    vectors: list[list[float]],
    payloads: list[dict[str, Any]],
    model_name: str,
    model_version: str,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> int:
    """Upsert note vectors into Qdrant.

    Returns the total number of points successfully upserted.
    Each note gets a deterministic point ID: UUID5(note_id).
    """
    if not (len(note_ids) == len(vectors) == len(payloads)):
        raise ValueError("note_ids, vectors, and payloads must have the same length")

    collection = await ensure_collection(model_name, model_version)
    client = get_qdrant_client()

    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_BATCHES)
    total_upserted = 0
    tasks = []

    batches = list(_make_batches(note_ids, vectors, payloads, batch_size))

    async def _upsert_batch(points: list[PointStruct]) -> int:
        async with semaphore:
            try:
                await client.upsert(collection_name=collection, points=points, wait=True)
                return len(points)
            except Exception as exc:
                logger.error(
                    "batch_upsert_failed",
                    collection=collection,
                    batch_size=len(points),
                    error=str(exc),
                )
                raise VectorStoreError(f"Batch upsert failed: {exc}") from exc

    for batch_ids, batch_vecs, batch_payloads in batches:
        points = [
            PointStruct(
                id=_stable_point_id(nid),
                vector=vec,
                payload={**payload, "note_id": nid},
            )
            for nid, vec, payload in zip(batch_ids, batch_vecs, batch_payloads, strict=False)
        ]
        tasks.append(_upsert_batch(points))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            raise result
        assert isinstance(result, int)
        total_upserted += result

    PIPELINE_NOTES_TOTAL.labels(status="success").inc(total_upserted)
    EMBEDDING_INFERENCE_LATENCY.labels(model=model_name).observe(0)
    logger.info(
        "upsert_completed",
        collection=collection,
        total=total_upserted,
        model=model_name,
        version=model_version,
    )
    return total_upserted


def _stable_point_id(note_id: str) -> str:
    """Derive a deterministic UUID from a note_id for idempotent upserts."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"note:{note_id}"))


def _make_batches(
    note_ids: list[str],
    vectors: list[list[float]],
    payloads: list[dict[str, Any]],
    batch_size: int,
) -> list[tuple[list[str], list[list[float]], list[dict[str, Any]]]]:
    batches = []
    for i in range(0, len(note_ids), batch_size):
        batches.append(
            (
                note_ids[i : i + batch_size],
                vectors[i : i + batch_size],
                payloads[i : i + batch_size],
            )
        )
    return batches

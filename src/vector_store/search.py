"""ANN vector search with metadata filtering."""

from dataclasses import dataclass
from typing import Any

import structlog
from qdrant_client.models import FieldCondition, Filter, MatchValue, SearchParams

from src.core.exceptions import CollectionNotFoundError, VectorStoreError
from src.core.telemetry import RAG_QUERY_LATENCY
from src.vector_store.client import get_qdrant_client

logger = structlog.get_logger(__name__)

_DEFAULT_EF = 128
_DEFAULT_TOP_K = 5


@dataclass
class SearchResult:
    note_id: str
    score: float
    payload: dict[str, Any]


async def search_similar(
    query_vector: list[float],
    collection: str,
    *,
    top_k: int = _DEFAULT_TOP_K,
    patient_id: str | None = None,
    note_type: str | None = None,
    score_threshold: float | None = None,
) -> list[SearchResult]:
    """Search for the most similar note vectors.

    Optionally filter by patient_id and/or note_type.
    """
    client = get_qdrant_client()

    query_filter = _build_filter(patient_id=patient_id, note_type=note_type)

    try:
        results = await client.search(  # type: ignore[attr-defined]
            collection_name=collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
            score_threshold=score_threshold,
            search_params=SearchParams(hnsw_ef=_DEFAULT_EF),
            with_payload=True,
        )
    except Exception as exc:
        msg = str(exc).lower()
        if "not found" in msg or "doesn't exist" in msg:
            raise CollectionNotFoundError(collection) from exc
        raise VectorStoreError(f"Search failed on {collection!r}: {exc}") from exc

    hits = [
        SearchResult(
            note_id=str(r.payload.get("note_id", r.id)),
            score=r.score,
            payload=r.payload or {},
        )
        for r in results
    ]

    RAG_QUERY_LATENCY.observe(0)
    logger.debug("search_completed", collection=collection, top_k=top_k, hits=len(hits))
    return hits


async def search_multi_collection(
    query_vector: list[float],
    collections: list[str],
    *,
    top_k: int = _DEFAULT_TOP_K,
    patient_id: str | None = None,
) -> list[SearchResult]:
    """Search across multiple collections and merge results by score."""
    import asyncio

    all_results: list[SearchResult] = []
    tasks = [
        search_similar(query_vector, col, top_k=top_k, patient_id=patient_id) for col in collections
    ]
    per_collection = await asyncio.gather(*tasks, return_exceptions=True)

    for result in per_collection:
        if isinstance(result, Exception):
            logger.warning("collection_search_failed", error=str(result))
            continue
        all_results.extend(result)  # type: ignore[arg-type]

    all_results.sort(key=lambda r: r.score, reverse=True)
    return all_results[:top_k]


def _build_filter(
    patient_id: str | None,
    note_type: str | None,
) -> Filter | None:
    conditions = []
    if patient_id:
        conditions.append(FieldCondition(key="patient_id", match=MatchValue(value=patient_id)))
    if note_type:
        conditions.append(FieldCondition(key="note_type", match=MatchValue(value=note_type)))
    return Filter(must=conditions) if conditions else None

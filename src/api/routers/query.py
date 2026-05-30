"""FastAPI router — POST /query (batch) and GET /query/stream (SSE)."""

import json
import time

import structlog
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from src.core.exceptions import CollectionNotFoundError, GuardrailViolationError, RAGError
from src.core.schemas import QueryRequest, QueryResponse
from src.rag.answer_generator import generate, stream_generate
from src.rag.context_builder import build_context
from src.rag.retriever import retrieve
from src.vector_store.collections import list_note_collections

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/query", tags=["query"])

_DEFAULT_COLLECTION_MODEL = "biomedbert"
_DEFAULT_COLLECTION_VERSION = "v1"


async def _get_collection() -> str:
    """Return the default active collection or raise 503."""
    collections = await list_note_collections()
    if not collections:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No indexed collections available. Run the ingestion pipeline first.",
        )
    # Prefer the default biomedbert v1 collection; fall back to first available.
    default = f"notes_{_DEFAULT_COLLECTION_MODEL}_v1"
    return default if default in collections else collections[0]


@router.post("", response_model=QueryResponse, summary="RAG query (synchronous)")
async def query(body: QueryRequest) -> QueryResponse:
    """Full RAG pipeline: retrieve → rerank → build context → generate answer."""
    collection = await _get_collection()

    try:
        candidates = await retrieve(
            body.query,
            collection,
            top_k=body.top_k * 4,  # over-retrieve before context packing
            patient_id=body.patient_id,
        )
    except CollectionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    context = build_context(candidates, max_tokens=3500)

    if not context.citations:
        return QueryResponse(
            answer="Information not available in the provided notes.",
            sources=[],
            model="none",
            latency_ms=0,
        )

    try:
        response = await generate(
            body.query,
            context,
            model=body.model,
        )
    except GuardrailViolationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except RAGError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"RAG generation failed: {exc}",
        ) from exc

    return response


@router.get("/stream", summary="RAG query (SSE streaming)")
async def query_stream(
    q: str = Query(min_length=5, max_length=2000, description="Clinical question"),
    patient_id: str | None = Query(default=None),
    model: str | None = Query(default=None),
) -> StreamingResponse:
    """Stream the RAG answer as Server-Sent Events."""
    collection = await _get_collection()
    candidates = await retrieve(q, collection, patient_id=patient_id)
    context = build_context(candidates)

    from collections.abc import AsyncGenerator

    async def _event_stream() -> AsyncGenerator[str, None]:
        start = time.perf_counter()
        try:
            async for chunk in stream_generate(q, context, model=model):
                data = json.dumps({"token": chunk})
                yield f"data: {data}\n\n"
        except Exception as exc:  # noqa: BLE001
            logger.error("sse_stream_error", error=str(exc))
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        finally:
            latency_ms = int((time.perf_counter() - start) * 1000)
            yield f"data: {json.dumps({'done': True, 'latency_ms': latency_ms})}\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

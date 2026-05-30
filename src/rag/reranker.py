"""MedCPT cross-encoder reranker with Redis 10-minute result cache."""

import asyncio
import hashlib
import json
from functools import lru_cache
from typing import Any

import structlog

from src.vector_store.search import SearchResult

logger = structlog.get_logger(__name__)

_MEDCPT_MODEL = "ncats/MedCPT-Cross-Encoder"
_CACHE_TTL = 600  # 10 minutes


@lru_cache(maxsize=1)
def _get_cross_encoder() -> Any:  # noqa: ANN401
    from sentence_transformers import CrossEncoder  # lazy — heavy model load

    logger.info("loading_medcpt_reranker", model=_MEDCPT_MODEL)
    model = CrossEncoder(_MEDCPT_MODEL)
    logger.info("medcpt_reranker_loaded")
    return model


async def rerank(
    query: str,
    candidates: list[SearchResult],
    *,
    top_k: int = 5,
    redis_client: Any | None = None,  # noqa: ANN401
) -> list[SearchResult]:
    """Rerank candidates using the MedCPT cross-encoder.

    Results are cached in Redis for 10 minutes keyed by (query, candidate IDs).
    """
    if not candidates:
        return []

    cache_key = _cache_key(query, [c.note_id for c in candidates])

    if redis_client is not None:
        cached = await redis_client.get(cache_key)
        if cached:
            order: list[str] = json.loads(cached)
            id_map = {c.note_id: c for c in candidates}
            reranked = [id_map[nid] for nid in order if nid in id_map]
            logger.debug("rerank_cache_hit", key=cache_key)
            return reranked[:top_k]

    reranked = await asyncio.get_running_loop().run_in_executor(
        None, _score_and_sort, query, candidates
    )
    reranked = reranked[:top_k]

    if redis_client is not None:
        order = [c.note_id for c in reranked]
        await redis_client.set(cache_key, json.dumps(order), ex=_CACHE_TTL)
        logger.debug("rerank_cache_set", key=cache_key)

    logger.info("rerank_done", candidates=len(candidates), returned=len(reranked))
    return reranked


def _score_and_sort(query: str, candidates: list[SearchResult]) -> list[SearchResult]:
    model = _get_cross_encoder()
    texts = [c.payload.get("raw_text", c.payload.get("processed_text", "")) for c in candidates]
    pairs = [(query, t) for t in texts]
    scores: list[float] = model.predict(pairs).tolist()
    ranked = sorted(zip(candidates, scores, strict=False), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked]


def _cache_key(query: str, note_ids: list[str]) -> str:
    content = query + "|" + ",".join(sorted(note_ids))
    digest = hashlib.sha256(content.encode()).hexdigest()[:16]
    return f"rerank:{digest}"

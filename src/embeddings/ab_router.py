"""Deterministic A/B routing for embedding models.

Routing is based on MD5(note_id) % 100, which guarantees:
- The same note always gets the same model (no drift between re-indexing runs).
- Traffic split is controlled by ABTest.traffic_b_pct (0.0–1.0).
- Dual-write mode: index with both models simultaneously for safe rollout.
"""

import asyncio
import hashlib
from typing import Any

import structlog

from src.core.models import ABTest
from src.embeddings.base_embedder import BaseEmbedder
from src.vector_store.indexer import upsert_notes

logger = structlog.get_logger(__name__)


def route(
    note_id: str, ab_test: ABTest, embedder_a: BaseEmbedder, embedder_b: BaseEmbedder
) -> BaseEmbedder:
    """Return the embedder assigned to this note_id under the given A/B test.

    Bucket is deterministic: MD5(note_id) % 100.
    If bucket < traffic_b_pct * 100, use model B; otherwise use model A.
    """
    bucket = _bucket(note_id)
    threshold = int(ab_test.traffic_b_pct * 100)
    chosen = embedder_b if bucket < threshold else embedder_a
    logger.debug(
        "ab_route",
        note_id=note_id,
        bucket=bucket,
        threshold=threshold,
        chosen=chosen.model_name,
    )
    return chosen


def _bucket(note_id: str) -> int:
    """Deterministic integer bucket 0–99 for a note_id."""
    digest = hashlib.md5(note_id.encode(), usedforsecurity=False).hexdigest()
    return int(digest, 16) % 100


async def embed_and_index(
    *,
    note_ids: list[str],
    texts: list[str],
    payloads: list[dict[str, Any]],
    ab_test: ABTest,
    embedder_a: BaseEmbedder,
    embedder_b: BaseEmbedder,
) -> dict[str, int]:
    """Embed notes with the routed model and upsert into Qdrant.

    Returns a dict with upsert counts per model.
    """
    # Partition by assigned model
    ids_a, texts_a, payloads_a = [], [], []
    ids_b, texts_b, payloads_b = [], [], []

    for nid, text, payload in zip(note_ids, texts, payloads, strict=False):
        chosen = route(nid, ab_test, embedder_a, embedder_b)
        if chosen is embedder_b:
            ids_b.append(nid)
            texts_b.append(text)
            payloads_b.append(payload)
        else:
            ids_a.append(nid)
            texts_a.append(text)
            payloads_a.append(payload)

    counts: dict[str, int] = {}

    async def _process(
        embedder: BaseEmbedder,
        nids: list[str],
        txts: list[str],
        plds: list[dict[str, Any]],
    ) -> None:
        if not nids:
            return
        vectors = await embedder.embed(txts)
        n = await upsert_notes(
            note_ids=nids,
            vectors=vectors,
            payloads=plds,
            model_name=embedder.model_name,
            model_version=embedder.model_version,
        )
        counts[embedder.model_name] = counts.get(embedder.model_name, 0) + n

    await asyncio.gather(
        _process(embedder_a, ids_a, texts_a, payloads_a),
        _process(embedder_b, ids_b, texts_b, payloads_b),
    )

    logger.info("ab_embed_and_index_done", counts=counts)
    return counts


async def dual_write(
    *,
    note_ids: list[str],
    texts: list[str],
    payloads: list[dict[str, Any]],
    embedder_a: BaseEmbedder,
    embedder_b: BaseEmbedder,
) -> dict[str, int]:
    """Index ALL notes with BOTH models simultaneously (safe rollout / backfill)."""
    counts: dict[str, int] = {}

    async def _process(embedder: BaseEmbedder) -> None:
        vectors = await embedder.embed(texts)
        n = await upsert_notes(
            note_ids=note_ids,
            vectors=vectors,
            payloads=payloads,
            model_name=embedder.model_name,
            model_version=embedder.model_version,
        )
        counts[embedder.model_name] = n

    await asyncio.gather(_process(embedder_a), _process(embedder_b))
    logger.info("dual_write_done", counts=counts)
    return counts

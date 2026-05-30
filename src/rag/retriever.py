"""RAG retriever — embed query and fetch candidate notes from Qdrant."""

import structlog

from src.embeddings.base_embedder import BaseEmbedder
from src.embeddings.biomedbert_embedder import BiomedBertEmbedder
from src.vector_store.search import SearchResult, search_multi_collection, search_similar

logger = structlog.get_logger(__name__)

_DEFAULT_RETRIEVAL_K = 20  # over-retrieve before reranking


async def retrieve(
    query: str,
    collection: str,
    *,
    top_k: int = _DEFAULT_RETRIEVAL_K,
    patient_id: str | None = None,
    note_type: str | None = None,
    score_threshold: float | None = None,
    embedder: BaseEmbedder | None = None,
) -> list[SearchResult]:
    """Embed the query and retrieve the most similar notes from a single collection."""
    if embedder is None:
        embedder = BiomedBertEmbedder()

    query_vector = await embedder.embed_one(query)

    results = await search_similar(
        query_vector,
        collection,
        top_k=top_k,
        patient_id=patient_id,
        note_type=note_type,
        score_threshold=score_threshold,
    )

    logger.info("retrieval_done", collection=collection, hits=len(results), top_k=top_k)
    return results


async def retrieve_multi(
    query: str,
    collections: list[str],
    *,
    top_k: int = _DEFAULT_RETRIEVAL_K,
    patient_id: str | None = None,
    embedder: BaseEmbedder | None = None,
) -> list[SearchResult]:
    """Embed the query and retrieve across multiple collections (A/B model scenario)."""
    if not collections:
        return []

    if embedder is None:
        embedder = BiomedBertEmbedder()

    query_vector = await embedder.embed_one(query)

    results = await search_multi_collection(
        query_vector,
        collections,
        top_k=top_k,
        patient_id=patient_id,
    )

    logger.info("multi_retrieval_done", collections=collections, hits=len(results))
    return results

"""Qdrant collection management — create and validate collections per model version."""

import structlog
from qdrant_client.models import Distance, HnswConfigDiff, VectorParams

from src.core.exceptions import CollectionNotFoundError, VectorStoreError
from src.vector_store.client import get_qdrant_client

logger = structlog.get_logger(__name__)

# Canonical vector sizes per model family
VECTOR_SIZES: dict[str, int] = {
    "biomedbert": 768,
    "lora-mistral": 4096,
}

_DISTANCE = Distance.COSINE
_HNSW_M = 16
_HNSW_EF_CONSTRUCT = 100


def collection_name(model_name: str, model_version: str) -> str:
    """Return the canonical collection name: notes_{model}_{version}."""
    safe_model = model_name.replace("/", "_").replace("-", "_").lower()
    safe_version = model_version.replace(".", "_").lower()
    return f"notes_{safe_model}_{safe_version}"


async def ensure_collection(model_name: str, model_version: str) -> str:
    """Create the collection if it does not exist. Returns the collection name."""
    name = collection_name(model_name, model_version)
    client = get_qdrant_client()

    try:
        exists = await client.collection_exists(name)
        if exists:
            logger.debug("collection_exists", collection=name)
            return name

        vector_size = _resolve_vector_size(model_name)
        await client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=_DISTANCE,
            ),
            hnsw_config=HnswConfigDiff(m=_HNSW_M, ef_construct=_HNSW_EF_CONSTRUCT),
        )
        logger.info("collection_created", collection=name, vector_size=vector_size)
        return name

    except Exception as exc:
        raise VectorStoreError(f"Failed to ensure collection {name!r}: {exc}") from exc


async def get_collection_info(collection: str) -> dict[str, object]:
    """Return basic info about a collection. Raises CollectionNotFoundError if missing."""
    client = get_qdrant_client()
    try:
        exists = await client.collection_exists(collection)
        if not exists:
            raise CollectionNotFoundError(collection)
        info = await client.get_collection(collection)
        return {
            "name": collection,
            "points_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": str(info.status),
        }
    except CollectionNotFoundError:
        raise
    except Exception as exc:
        raise VectorStoreError(f"Failed to get collection info for {collection!r}: {exc}") from exc


async def list_note_collections() -> list[str]:
    """Return all collections whose name starts with 'notes_'."""
    client = get_qdrant_client()
    try:
        result = await client.get_collections()
        return [c.name for c in result.collections if c.name.startswith("notes_")]
    except Exception as exc:
        raise VectorStoreError(f"Failed to list collections: {exc}") from exc


def _resolve_vector_size(model_name: str) -> int:
    for prefix, size in VECTOR_SIZES.items():
        if prefix in model_name.lower():
            return size
    raise VectorStoreError(
        f"Unknown vector size for model {model_name!r}. "
        f"Add it to VECTOR_SIZES or pass vector_size explicitly."
    )

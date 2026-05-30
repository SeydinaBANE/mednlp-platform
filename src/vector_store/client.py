"""Async Qdrant client singleton."""

from functools import lru_cache

from qdrant_client import AsyncQdrantClient

from src.core.config import get_settings


@lru_cache
def get_qdrant_client() -> AsyncQdrantClient:
    settings = get_settings()
    return AsyncQdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        api_key=settings.qdrant_api_key or None,
        timeout=30,
    )

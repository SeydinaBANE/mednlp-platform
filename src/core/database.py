"""Async SQLAlchemy engine and session factory."""

from collections.abc import AsyncGenerator
from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.core.config import get_settings


@lru_cache(maxsize=1)
def _get_engine() -> object:
    settings = get_settings()
    return create_async_engine(
        settings.database_url,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        echo=False,
    )


@lru_cache(maxsize=1)
def _get_session_factory() -> async_sessionmaker[AsyncSession]:
    from sqlalchemy.ext.asyncio import AsyncEngine

    engine = _get_engine()
    assert isinstance(engine, AsyncEngine)
    return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — yields a DB session per request."""
    factory = _get_session_factory()
    async with factory() as session:
        yield session

#!/usr/bin/env python3
"""Initialize the database schema and seed required configuration.

Usage:
    uv run python scripts/bootstrap_db.py
    uv run python scripts/bootstrap_db.py --seed-ab-tests
"""

import argparse
import asyncio
from datetime import UTC, datetime

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.core.config import get_settings
from src.core.models import ABTest, Base
from src.core.telemetry import setup_logging

logger = structlog.get_logger(__name__)


async def create_tables(engine: object) -> None:
    from sqlalchemy.ext.asyncio import AsyncEngine

    assert isinstance(engine, AsyncEngine)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("tables_created")


async def seed_ab_tests(factory: async_sessionmaker[AsyncSession]) -> None:
    async with factory() as session:
        existing = await session.execute(
            __import__("sqlalchemy", fromlist=["select"]).select(ABTest)
        )
        if existing.scalars().first():
            logger.info("ab_tests_already_seeded")
            return

        session.add(
            ABTest(
                name="biomedbert-vs-lora-mistral",
                model_a="biomedbert",
                model_b="lora-mistral",
                traffic_b_pct=0.0,  # start with 0% — enable manually
                is_active=False,
                created_at=datetime.now(tz=UTC),
            )
        )
        await session.commit()
        logger.info("ab_tests_seeded")


async def main(seed_ab: bool) -> None:
    setup_logging()
    settings = get_settings()
    engine = create_async_engine(settings.database_url)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    logger.info("bootstrap_started", db=settings.database_url_sync)
    await create_tables(engine)

    if seed_ab:
        await seed_ab_tests(factory)

    await engine.dispose()
    logger.info("bootstrap_complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap the MedNLP database")
    parser.add_argument("--seed-ab-tests", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(seed_ab=args.seed_ab_tests))

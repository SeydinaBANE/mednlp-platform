#!/usr/bin/env python3
"""Re-vectorize all indexed notes with a new embedding model.

Usage:
    uv run python scripts/backfill_vectors.py --model v2
    uv run python scripts/backfill_vectors.py --model v2 --batch-size 32 --dry-run
"""

import argparse
import asyncio

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.core.config import get_settings
from src.core.models import Note
from src.core.telemetry import setup_logging
from src.embeddings.biomedbert_embedder import BiomedBertEmbedder
from src.vector_store.indexer import upsert_notes

logger = structlog.get_logger(__name__)

_DEFAULT_BATCH_SIZE = 32
_DEFAULT_PAGE_SIZE = 500


async def backfill(model_version: str, batch_size: int, dry_run: bool) -> None:
    setup_logging()
    settings = get_settings()
    engine = create_async_engine(settings.database_url)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    embedder = BiomedBertEmbedder()

    logger.info(
        "backfill_started",
        model=embedder.model_name,
        version=model_version,
        dry_run=dry_run,
    )

    offset = 0
    total_processed = 0

    while True:
        async with factory() as session:
            result = await session.execute(
                select(Note).order_by(Note.created_at).offset(offset).limit(_DEFAULT_PAGE_SIZE)
            )
            notes = result.scalars().all()

        if not notes:
            break

        # Process in sub-batches for embedding efficiency
        for i in range(0, len(notes), batch_size):
            batch = notes[i : i + batch_size]
            texts = [n.processed_text or n.raw_text for n in batch]
            note_ids = [n.id for n in batch]
            payloads = [
                {
                    "patient_id": n.patient_id,
                    "note_type": n.note_type,
                    "authored_at": n.authored_at.isoformat(),
                    "source": n.source,
                    "raw_text": (n.processed_text or n.raw_text)[:2000],
                }
                for n in batch
            ]

            if not dry_run:
                vectors = await embedder.embed(texts)
                await upsert_notes(
                    note_ids=note_ids,
                    vectors=vectors,
                    payloads=payloads,
                    model_name=embedder.model_name,
                    model_version=model_version,
                )
                total_processed += len(batch)
            else:
                total_processed += len(batch)
                logger.info("dry_run_batch", n=len(batch))

        offset += _DEFAULT_PAGE_SIZE
        logger.info("backfill_progress", processed=total_processed)

    await engine.dispose()
    logger.info("backfill_complete", total=total_processed, dry_run=dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill Qdrant vectors with a new model")
    parser.add_argument("--model", required=True, help="New model version tag (e.g. v2)")
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH_SIZE)
    parser.add_argument("--dry-run", action="store_true", help="Skip actual upserts")
    args = parser.parse_args()
    asyncio.run(backfill(args.model, args.batch_size, args.dry_run))

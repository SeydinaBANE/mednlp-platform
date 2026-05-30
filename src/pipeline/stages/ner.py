"""Named Entity Recognition stage using spaCy + scispaCy (biomedical entities)."""

import asyncio
import time
from functools import lru_cache
from typing import Any

import structlog
from prefect import task

from src.core.exceptions import PipelineError
from src.core.telemetry import PIPELINE_STAGE_LATENCY
from src.pipeline.schemas import Entity, PipelineContext

logger = structlog.get_logger(__name__)

_NER_MODEL = "en_core_sci_md"
_FALLBACK_MODEL = "en_core_web_sm"


@lru_cache(maxsize=1)
def _load_ner_nlp() -> Any:
    import spacy

    for model in (_NER_MODEL, _FALLBACK_MODEL):
        try:
            nlp = spacy.load(model)
            logger.info("ner_model_loaded", model=model)
            return nlp
        except OSError:
            logger.warning("ner_model_not_found", model=model)

    raise PipelineError("ner", "No spaCy NER model available")


@task(name="ner", retries=1, retry_delay_seconds=5)
async def extract_entities(ctx: PipelineContext) -> PipelineContext:
    """Extract biomedical named entities from de-identified text."""
    start = time.perf_counter()
    text = ctx.processed_text
    try:
        ctx.entities = await asyncio.get_running_loop().run_in_executor(None, _ner_sync, text)
        elapsed = time.perf_counter() - start
        PIPELINE_STAGE_LATENCY.labels(stage="ner", status="success").observe(elapsed)
        logger.info("ner_done", note_id=ctx.note.note_id, n_entities=len(ctx.entities))
    except PipelineError:
        raise
    except Exception as exc:
        PIPELINE_STAGE_LATENCY.labels(stage="ner", status="failure").observe(0)
        raise PipelineError("ner", str(exc)) from exc

    return ctx


def _ner_sync(text: str) -> list[Entity]:
    nlp = _load_ner_nlp()
    doc = nlp(text)
    entities = []
    seen: set[tuple[int, int]] = set()

    for ent in doc.ents:
        key = (ent.start_char, ent.end_char)
        if key in seen:
            continue
        seen.add(key)
        entities.append(
            Entity(
                text=ent.text,
                label=ent.label_,
                start_char=ent.start_char,
                end_char=ent.end_char,
                score=1.0,  # spaCy rule-based NER doesn't expose confidence
            )
        )

    return entities

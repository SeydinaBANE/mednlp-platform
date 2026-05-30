"""Sentence segmentation stage using spaCy."""

from functools import lru_cache
from typing import Any

import structlog
from prefect import task

from src.core.exceptions import PipelineError
from src.core.telemetry import PIPELINE_STAGE_LATENCY
from src.pipeline.schemas import PipelineContext, Segment

logger = structlog.get_logger(__name__)

_MODEL_NAME = "en_core_sci_md"
_FALLBACK_MODEL = "en_core_web_sm"


@lru_cache(maxsize=1)
def _load_nlp() -> Any:
    import spacy

    for model in (_MODEL_NAME, _FALLBACK_MODEL):
        try:
            nlp = spacy.load(model, disable=["ner", "textcat"])
            logger.info("spacy_model_loaded", model=model)
            return nlp
        except OSError:
            logger.warning("spacy_model_not_found", model=model)

    raise PipelineError("segmenter", f"Neither {_MODEL_NAME} nor {_FALLBACK_MODEL} is installed")


@task(name="segmenter", retries=1, retry_delay_seconds=5)
async def segment(ctx: PipelineContext) -> PipelineContext:
    """Split note text into sentences and populate ctx.segments."""
    import asyncio
    import time

    start = time.perf_counter()
    try:
        segments = await asyncio.get_running_loop().run_in_executor(
            None, _segment_sync, ctx.note.raw_text
        )
        ctx.segments = segments
        elapsed = time.perf_counter() - start
        PIPELINE_STAGE_LATENCY.labels(stage="segmenter", status="success").observe(elapsed)
        logger.info("segmenter_done", note_id=ctx.note.note_id, n_segments=len(segments))
    except PipelineError:
        raise
    except Exception as exc:
        PIPELINE_STAGE_LATENCY.labels(stage="segmenter", status="failure").observe(0)
        raise PipelineError("segmenter", str(exc)) from exc

    return ctx


def _segment_sync(text: str) -> list[Segment]:
    nlp = _load_nlp()
    doc = nlp(text)
    return [
        Segment(
            text=sent.text.strip(),
            start_char=sent.start_char,
            end_char=sent.end_char,
            sentence_index=i,
        )
        for i, sent in enumerate(doc.sents)
        if sent.text.strip()
    ]

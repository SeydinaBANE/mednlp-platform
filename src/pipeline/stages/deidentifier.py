"""PHI de-identification stage using Presidio Analyzer + Anonymizer."""

import asyncio
import time
from functools import lru_cache
from typing import Any

import structlog
from prefect import task

from src.core.exceptions import DeidentificationError
from src.core.telemetry import PIPELINE_STAGE_LATENCY
from src.pipeline.schemas import PipelineContext

logger = structlog.get_logger(__name__)

_PHI_ENTITIES = [
    "PERSON",
    "DATE_TIME",
    "PHONE_NUMBER",
    "US_SSN",
    "US_DRIVER_LICENSE",
    "MEDICAL_LICENSE",
    "EMAIL_ADDRESS",
    "LOCATION",
    "URL",
    "IP_ADDRESS",
]


@lru_cache(maxsize=1)
def _get_analyzer() -> Any:
    from presidio_analyzer import AnalyzerEngine

    logger.info("loading_presidio_analyzer")
    return AnalyzerEngine()


@lru_cache(maxsize=1)
def _get_anonymizer() -> Any:
    from presidio_anonymizer import AnonymizerEngine

    logger.info("loading_presidio_anonymizer")
    return AnonymizerEngine()


@task(name="deidentifier", retries=1, retry_delay_seconds=5)
async def deidentify(ctx: PipelineContext) -> PipelineContext:
    """Detect and anonymize PHI in the note text."""
    start = time.perf_counter()
    try:
        ctx.deidentified_text = await asyncio.get_running_loop().run_in_executor(
            None, _deidentify_sync, ctx.note.raw_text
        )
        elapsed = time.perf_counter() - start
        PIPELINE_STAGE_LATENCY.labels(stage="deidentifier", status="success").observe(elapsed)
        logger.info("deidentifier_done", note_id=ctx.note.note_id)
    except DeidentificationError:
        raise
    except Exception as exc:
        PIPELINE_STAGE_LATENCY.labels(stage="deidentifier", status="failure").observe(0)
        raise DeidentificationError("deidentifier", str(exc)) from exc

    return ctx


def _deidentify_sync(text: str) -> str:
    from presidio_anonymizer.entities import OperatorConfig

    analyzer = _get_analyzer()
    anonymizer = _get_anonymizer()

    results = analyzer.analyze(text=text, entities=_PHI_ENTITIES, language="en")

    if not results:
        return text

    operators = {
        entity: OperatorConfig("replace", {"new_value": f"<{entity}>"}) for entity in _PHI_ENTITIES
    }

    anonymized = anonymizer.anonymize(text=text, analyzer_results=results, operators=operators)
    return str(anonymized.text)

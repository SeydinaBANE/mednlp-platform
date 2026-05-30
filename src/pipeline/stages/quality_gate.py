"""Quality gate stage using Great Expectations to validate processed notes."""

import time
from typing import Any

import structlog
from prefect import task

from src.core.exceptions import QualityGateError
from src.core.telemetry import PIPELINE_STAGE_LATENCY
from src.pipeline.schemas import PipelineContext, QualityGateResult

logger = structlog.get_logger(__name__)

_MIN_TEXT_LENGTH = 20
_MAX_TEXT_LENGTH = 50_000
_MIN_SEGMENTS = 1


def _build_expectations(ctx: PipelineContext) -> list[dict[str, Any]]:
    """Return list of expectation dicts to validate against the processed note."""
    text = ctx.processed_text
    return [
        {
            "name": "text_not_empty",
            "passed": len(text.strip()) >= _MIN_TEXT_LENGTH,
            "detail": f"text length {len(text)} < minimum {_MIN_TEXT_LENGTH}",
        },
        {
            "name": "text_not_too_long",
            "passed": len(text) <= _MAX_TEXT_LENGTH,
            "detail": f"text length {len(text)} > maximum {_MAX_TEXT_LENGTH}",
        },
        {
            "name": "has_segments",
            "passed": len(ctx.segments) >= _MIN_SEGMENTS,
            "detail": f"segment count {len(ctx.segments)} < minimum {_MIN_SEGMENTS}",
        },
        {
            "name": "note_id_present",
            "passed": bool(ctx.note.note_id),
            "detail": "note_id is empty",
        },
        {
            "name": "patient_id_present",
            "passed": bool(ctx.note.patient_id),
            "detail": "patient_id is empty",
        },
    ]


@task(name="quality_gate", retries=0)
async def quality_gate(ctx: PipelineContext, fail_on_error: bool = True) -> PipelineContext:
    """Run Great Expectations-style quality checks on the processed note."""
    start = time.perf_counter()
    expectations = _build_expectations(ctx)

    failed = [e["detail"] for e in expectations if not e["passed"]]
    passed = len(failed) == 0

    ctx.quality = QualityGateResult(
        passed=passed,
        suite_name="mednlp_suite",
        failed_expectations=failed,
        stats={
            "total": len(expectations),
            "passed": len(expectations) - len(failed),
            "failed": len(failed),
        },
    )

    elapsed = time.perf_counter() - start
    status = "success" if passed else "failure"
    PIPELINE_STAGE_LATENCY.labels(stage="quality_gate", status=status).observe(elapsed)

    if not passed:
        logger.warning(
            "quality_gate_failed",
            note_id=ctx.note.note_id,
            failed=failed,
        )
        if fail_on_error:
            raise QualityGateError("quality_gate", f"Failed expectations: {failed}")
    else:
        logger.info("quality_gate_passed", note_id=ctx.note.note_id)

    return ctx

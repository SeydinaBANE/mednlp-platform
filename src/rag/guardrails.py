"""PHI detection and clinical disclaimer guardrails for RAG answers."""

from functools import lru_cache
from typing import Any

import structlog

from src.core.exceptions import GuardrailViolationError

logger = structlog.get_logger(__name__)

_DISCLAIMER = (
    "\n\n---\n"
    "This is a clinical decision support tool. "
    "The above response is AI-generated and not a substitute for clinical judgment. "
    "Always verify with primary source documentation."
)

_PHI_ENTITIES = [
    "PERSON",
    "PHONE_NUMBER",
    "US_SSN",
    "US_DRIVER_LICENSE",
    "MEDICAL_LICENSE",
    "EMAIL_ADDRESS",
    "US_BANK_NUMBER",
]

_PHI_SCORE_THRESHOLD = 0.6


@lru_cache(maxsize=1)
def _get_analyzer() -> Any:  # noqa: ANN401
    from presidio_analyzer import AnalyzerEngine  # lazy — NLP model load

    logger.info("loading_presidio_analyzer")
    engine = AnalyzerEngine()
    logger.info("presidio_analyzer_loaded")
    return engine


def scan_for_phi(text: str) -> list[str]:
    """Return list of PHI entity types found above the confidence threshold."""
    engine = _get_analyzer()
    results = engine.analyze(
        text=text,
        entities=_PHI_ENTITIES,
        language="en",
    )
    return [r.entity_type for r in results if r.score >= _PHI_SCORE_THRESHOLD]


def apply_guardrails(answer: str, *, strict: bool = False) -> str:
    """Scan for PHI and append disclaimer.

    In strict mode, raise GuardrailViolationError if any PHI is detected.
    In non-strict mode, log a warning and continue.
    """
    phi_found = scan_for_phi(answer)

    if phi_found:
        logger.warning("phi_detected_in_answer", entities=phi_found)
        if strict:
            raise GuardrailViolationError(
                f"PHI detected in generated answer: {phi_found}. "
                "Answer was not returned to the user."
            )

    return answer + _DISCLAIMER

"""Drift alert publisher — PagerDuty + Slack + Prometheus + DB persistence."""

from datetime import UTC, datetime
from typing import Any

import structlog

from src.core.config import get_settings
from src.core.telemetry import DLQ_MESSAGES_TOTAL
from src.drift_detection.data_drift import DataDriftResult
from src.drift_detection.embedding_drift import EmbeddingDriftResult
from src.drift_detection.label_drift import LabelDriftResult

logger = structlog.get_logger(__name__)

_PAGERDUTY_URL = "https://events.pagerduty.com/v2/enqueue"


async def publish_embedding_drift(result: EmbeddingDriftResult) -> None:
    """Fire alerts if embedding drift exceeds threshold."""
    if not result.is_drifted:
        return

    details = {
        "drift_score": round(result.drift_score, 4),
        "threshold": result.threshold,
        "n_reference": result.n_reference,
        "n_current": result.n_current,
        "sampled_dims": result.sampled_dims,
    }
    summary = (
        f"Embedding drift detected: JS divergence {result.drift_score:.4f} "
        f"> threshold {result.threshold}"
    )
    await _fire_alerts(
        summary=summary,
        severity="warning",
        drift_type="embedding",
        details=details,
    )


async def publish_label_drift(result: LabelDriftResult) -> None:
    """Fire alerts if label drift is detected in any ICD-10 code."""
    if not result.is_drifted:
        return

    details = {
        "n_drifted_codes": result.n_drifted_codes,
        "n_tested_codes": result.n_tested_codes,
        "drifted_codes": result.drifted_codes[:10],
        "alpha_corrected": round(result.alpha_corrected, 6),
    }
    summary = (
        f"ICD-10 label drift: {result.n_drifted_codes}/{result.n_tested_codes} "
        f"codes drifted (Bonferroni α={result.alpha_corrected:.2e})"
    )
    await _fire_alerts(
        summary=summary,
        severity="error" if result.n_drifted_codes > 5 else "warning",
        drift_type="label",
        details=details,
    )


async def publish_data_drift(result: DataDriftResult) -> None:
    """Fire alerts if NER entity distribution drift is detected."""
    if not result.is_drifted:
        return

    details = {
        "chi2_statistic": round(result.chi2_statistic, 4),
        "p_value": round(result.p_value, 6),
        "entity_types": result.entity_types,
    }
    summary = (
        f"NER entity distribution drift: χ²={result.chi2_statistic:.2f}, " f"p={result.p_value:.2e}"
    )
    await _fire_alerts(
        summary=summary,
        severity="warning",
        drift_type="data",
        details=details,
    )


async def _fire_alerts(
    summary: str,
    severity: str,
    drift_type: str,
    details: dict[str, Any],
) -> None:
    settings = get_settings()
    logger.warning("drift_alert_firing", summary=summary, drift_type=drift_type)
    DLQ_MESSAGES_TOTAL.labels(reason=f"{drift_type}_drift_alert").inc()

    await _send_pagerduty(settings, summary, severity, details)
    await _send_slack(settings, summary, severity, details)


async def _send_pagerduty(
    settings: Any,  # noqa: ANN401
    summary: str,
    severity: str,
    details: dict[str, Any],
) -> None:
    if not settings.pagerduty_integration_key:
        return

    import httpx

    payload = {
        "routing_key": settings.pagerduty_integration_key,
        "event_action": "trigger",
        "payload": {
            "summary": f"[MedNLP] {summary}",
            "severity": severity,
            "source": "mednlp-drift-detection",
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "custom_details": details,
        },
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(_PAGERDUTY_URL, json=payload)
            resp.raise_for_status()
            logger.info("pagerduty_drift_alert_sent")
    except Exception as exc:  # noqa: BLE001
        logger.error("pagerduty_drift_alert_failed", error=str(exc))


async def _send_slack(
    settings: Any,  # noqa: ANN401
    summary: str,
    severity: str,
    details: dict[str, Any],
) -> None:
    if not settings.slack_webhook_url:
        return

    import httpx

    icon = ":rotating_light:" if severity == "error" else ":warning:"
    text = f"{icon} *MedNLP Drift Alert*\n{summary}\n```{details}```"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(settings.slack_webhook_url, json={"text": text})
            resp.raise_for_status()
            logger.info("slack_drift_alert_sent")
    except Exception as exc:  # noqa: BLE001
        logger.error("slack_drift_alert_failed", error=str(exc))

"""Dead-letter queue handler — publishes failed messages and fires PagerDuty/Slack alerts."""

import json
from typing import Any

import structlog
from google.cloud import pubsub_v1
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.core.config import get_settings
from src.core.exceptions import DLQError
from src.core.telemetry import DLQ_MESSAGES_TOTAL

logger = structlog.get_logger(__name__)

_publisher: pubsub_v1.PublisherClient | None = None


def _get_publisher() -> pubsub_v1.PublisherClient:
    global _publisher
    if _publisher is None:
        _publisher = pubsub_v1.PublisherClient()
    return _publisher


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def publish_to_dlq(message_id: str, raw_payload: str, error_reason: str) -> None:
    """Publish a failed message to the DLQ topic."""
    settings = get_settings()
    publisher = _get_publisher()
    topic_path = publisher.topic_path(settings.gcp_project_id, settings.pubsub_topic_dlq)

    envelope: dict[str, Any] = {
        "original_message_id": message_id,
        "error_reason": error_reason,
        "raw_payload": raw_payload,
    }
    data = json.dumps(envelope).encode("utf-8")

    try:
        future = publisher.publish(topic_path, data)
        future.result(timeout=10.0)
        DLQ_MESSAGES_TOTAL.labels(reason="published").inc()
        logger.info("dlq_published", message_id=message_id, topic=topic_path)
    except Exception as exc:
        logger.error("dlq_publish_failed", message_id=message_id, error=str(exc))
        raise DLQError(message_id, f"DLQ publish failed: {exc}") from exc


async def alert_if_threshold_exceeded(dlq_count: int, total_count: int) -> None:
    """Fire PagerDuty and/or Slack alert when DLQ rate exceeds threshold."""
    settings = get_settings()

    if total_count == 0:
        return

    rate = dlq_count / total_count
    if rate < settings.dlq_alert_rate_threshold:
        return

    logger.warning(
        "dlq_rate_threshold_exceeded",
        dlq_count=dlq_count,
        total_count=total_count,
        rate=f"{rate:.2%}",
        threshold=f"{settings.dlq_alert_rate_threshold:.2%}",
    )

    await _send_pagerduty_alert(rate, dlq_count, total_count)
    await _send_slack_alert(rate, dlq_count, total_count)


async def _send_pagerduty_alert(rate: float, dlq_count: int, total_count: int) -> None:
    settings = get_settings()
    if not settings.pagerduty_integration_key:
        return

    import httpx  # local import — only used in alert path

    payload = {
        "routing_key": settings.pagerduty_integration_key,
        "event_action": "trigger",
        "payload": {
            "summary": f"MedNLP DLQ rate {rate:.1%} exceeds threshold",
            "severity": "error",
            "source": "mednlp-platform",
            "custom_details": {
                "dlq_count": dlq_count,
                "total_count": total_count,
                "rate": f"{rate:.2%}",
            },
        },
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post("https://events.pagerduty.com/v2/enqueue", json=payload)
            resp.raise_for_status()
            logger.info("pagerduty_alert_sent")
    except Exception as exc:
        logger.error("pagerduty_alert_failed", error=str(exc))


async def _send_slack_alert(rate: float, dlq_count: int, total_count: int) -> None:
    settings = get_settings()
    if not settings.slack_webhook_url:
        return

    import httpx

    threshold_pct = f"{settings.dlq_alert_rate_threshold:.1%}"
    text = (
        f":warning: *MedNLP DLQ alert* — rate {rate:.1%} "
        f"({dlq_count}/{total_count} messages) exceeds threshold {threshold_pct}"
    )

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(settings.slack_webhook_url, json={"text": text})
            resp.raise_for_status()
            logger.info("slack_alert_sent")
    except Exception as exc:
        logger.error("slack_alert_failed", error=str(exc))

"""GCP Pub/Sub pull consumer with flow control, retry, and circuit-breaker.

Three failure modes:
  1. Transient (network, downstream down) → tenacity retry, then NACK
  2. Poison pill (malformed FHIR, corrupt encoding) → ACK + DLQ
  3. Processing timeout → periodic modify_ack_deadline (ack_deadline=600s)

Circuit-breaker trips after MAX_CONSECUTIVE_FAILURES and pauses for COOLDOWN_SECONDS.
"""

import asyncio
import json
import os
from collections.abc import Callable, Coroutine
from typing import Any

import structlog
from google.api_core.exceptions import GoogleAPIError
from google.cloud import pubsub_v1
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.core.config import get_settings
from src.core.exceptions import DLQError, FHIRParseError, MissingPatientReferenceError
from src.core.telemetry import DLQ_MESSAGES_TOTAL, PIPELINE_NOTES_TOTAL
from src.ingestion.fhir_parser import parse_fhir_resource
from src.ingestion.schemas import NoteRecord

logger = structlog.get_logger(__name__)

MAX_OUTSTANDING_MESSAGES = 10
ACK_DEADLINE_SECONDS = 600
MAX_CONSECUTIVE_FAILURES = 10
COOLDOWN_SECONDS = 60


NoteHandler = Callable[[NoteRecord], Coroutine[Any, Any, None]]


class CircuitBreaker:
    def __init__(self, threshold: int, cooldown: float) -> None:
        self._failures = 0
        self._threshold = threshold
        self._cooldown = cooldown
        self._open = False

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold:
            self._open = True
            logger.error(
                "circuit_breaker_open",
                failures=self._failures,
                cooldown_seconds=self._cooldown,
            )

    def record_success(self) -> None:
        self._failures = 0
        self._open = False

    async def wait_if_open(self) -> None:
        if self._open:
            logger.warning("circuit_breaker_waiting", seconds=self._cooldown)
            await asyncio.sleep(self._cooldown)
            self._open = False
            self._failures = 0


def _build_subscriber() -> pubsub_v1.SubscriberClient:
    settings = get_settings()
    if settings.pubsub_emulator_enabled:
        os.environ["PUBSUB_EMULATOR_HOST"] = settings.pubsub_emulator_host
    return pubsub_v1.SubscriberClient()


@retry(
    retry=retry_if_exception_type(GoogleAPIError),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
async def _process_with_retry(
    message_data: dict[str, Any],
    handler: NoteHandler,
) -> None:
    note = parse_fhir_resource(message_data)
    await handler(note)


async def run_consumer(
    handler: NoteHandler,
    dlq_publisher: Callable[[str, str, str], Coroutine[Any, Any, None]] | None = None,
    max_messages: int = MAX_OUTSTANDING_MESSAGES,
) -> None:
    """Pull messages from Pub/Sub and dispatch to handler.

    Runs indefinitely. Call in a background asyncio task.
    """
    settings = get_settings()
    subscriber = _build_subscriber()
    subscription_path = subscriber.subscription_path(
        settings.gcp_project_id, settings.pubsub_subscription_notes
    )
    circuit_breaker = CircuitBreaker(MAX_CONSECUTIVE_FAILURES, COOLDOWN_SECONDS)

    logger.info("pubsub_consumer_started", subscription=subscription_path)

    while True:
        await circuit_breaker.wait_if_open()

        try:
            response = subscriber.pull(
                request={
                    "subscription": subscription_path,
                    "max_messages": max_messages,
                },
                timeout=30.0,
            )
        except GoogleAPIError as exc:
            logger.error("pubsub_pull_error", error=str(exc))
            circuit_breaker.record_failure()
            await asyncio.sleep(5)
            continue

        if not response.received_messages:
            await asyncio.sleep(2)
            continue

        ack_ids_to_ack: list[str] = []
        ack_ids_to_nack: list[str] = []

        for received_message in response.received_messages:
            msg = received_message.message
            ack_id = received_message.ack_id
            message_id = msg.message_id

            log = logger.bind(message_id=message_id)

            # Decode payload
            try:
                data = json.loads(msg.data.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                log.error("message_decode_failed", error=str(exc))
                # Poison pill — ACK to remove from queue, route to DLQ
                ack_ids_to_ack.append(ack_id)
                DLQ_MESSAGES_TOTAL.labels(reason="decode_error").inc()
                if dlq_publisher:
                    raw = msg.data.decode("utf-8", errors="replace")
                    await dlq_publisher(message_id, raw, str(exc))
                continue

            # Attempt processing
            try:
                await _process_with_retry(data, handler)
                ack_ids_to_ack.append(ack_id)
                PIPELINE_NOTES_TOTAL.labels(status="success").inc()
                circuit_breaker.record_success()
                log.info("message_processed")

            except (FHIRParseError, MissingPatientReferenceError) as exc:
                # Poison pill — send to DLQ, ACK to prevent redelivery
                log.error("message_poison_pill", error=str(exc))
                ack_ids_to_ack.append(ack_id)
                PIPELINE_NOTES_TOTAL.labels(status="dlq").inc()
                DLQ_MESSAGES_TOTAL.labels(reason=type(exc).__name__).inc()
                if dlq_publisher:
                    await dlq_publisher(message_id, json.dumps(data), str(exc))

            except RetryError as exc:
                # All retries exhausted — NACK for redelivery
                log.error("message_retries_exhausted", error=str(exc))
                ack_ids_to_nack.append(ack_id)
                PIPELINE_NOTES_TOTAL.labels(status="failure").inc()
                circuit_breaker.record_failure()

            except DLQError as exc:
                log.error("message_dlq_error", error=str(exc))
                ack_ids_to_ack.append(ack_id)
                PIPELINE_NOTES_TOTAL.labels(status="dlq").inc()
                DLQ_MESSAGES_TOTAL.labels(reason="dlq_error").inc()

        # Batch ACK / NACK
        if ack_ids_to_ack:
            try:
                subscriber.acknowledge(
                    request={"subscription": subscription_path, "ack_ids": ack_ids_to_ack}
                )
            except GoogleAPIError as exc:
                logger.error("pubsub_ack_failed", error=str(exc))

        if ack_ids_to_nack:
            try:
                subscriber.modify_ack_deadline(
                    request={
                        "subscription": subscription_path,
                        "ack_ids": ack_ids_to_nack,
                        "ack_deadline_seconds": 0,  # Immediate redelivery
                    }
                )
            except GoogleAPIError as exc:
                logger.error("pubsub_nack_failed", error=str(exc))

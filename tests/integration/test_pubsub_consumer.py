"""Integration test — Pub/Sub consumer with the local emulator.

Requires docker-compose.test.yml to be running with the pubsub-emulator service.
The emulator must be accessible at PUBSUB_EMULATOR_HOST (default localhost:8085).
"""

import asyncio
import json
import os

import pytest

PUBSUB_EMULATOR_HOST = os.getenv("PUBSUB_EMULATOR_HOST", "localhost:8085")
GCP_PROJECT = "test-project"
TOPIC_ID = "notes.incoming.test"
SUBSCRIPTION_ID = "notes.processor.test"


def _emulator_available() -> bool:
    import socket

    host, port = PUBSUB_EMULATOR_HOST.split(":")
    try:
        with socket.create_connection((host, int(port)), timeout=2):
            return True
    except OSError:
        return False


# Skip all tests in this module if the emulator is unreachable
pytestmark = pytest.mark.skipif(
    not _emulator_available(),
    reason="Pub/Sub emulator not available",
)


def _setup_emulator_env() -> None:
    os.environ["PUBSUB_EMULATOR_HOST"] = PUBSUB_EMULATOR_HOST


def _create_topic_and_subscription() -> None:
    from google.cloud import pubsub_v1

    _setup_emulator_env()
    publisher = pubsub_v1.PublisherClient()
    subscriber = pubsub_v1.SubscriberClient()

    topic_path = publisher.topic_path(GCP_PROJECT, TOPIC_ID)
    sub_path = subscriber.subscription_path(GCP_PROJECT, SUBSCRIPTION_ID)

    try:
        publisher.create_topic(request={"name": topic_path})
    except Exception:  # noqa: BLE001
        pass  # already exists

    try:
        subscriber.create_subscription(request={"name": sub_path, "topic": topic_path})
    except Exception:  # noqa: BLE001
        pass


def _publish_fhir_note(note_id: str) -> None:
    from google.cloud import pubsub_v1

    _setup_emulator_env()
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(GCP_PROJECT, TOPIC_ID)

    fhir_doc = {
        "resourceType": "DocumentReference",
        "id": note_id,
        "status": "current",
        "subject": {"reference": f"Patient/patient-{note_id}"},
        "date": "2024-03-15T10:30:00Z",
        "type": {"coding": [{"system": "http://loinc.org", "code": "11506-3"}]},
        "text": {
            "status": "generated",
            "div": f"<div>Integration test note {note_id}. Patient presents with fever.</div>",
        },
    }

    future = publisher.publish(topic_path, json.dumps(fhir_doc).encode("utf-8"))
    future.result(timeout=10)


class TestPubSubConsumerIntegration:
    def setup_method(self) -> None:
        _setup_emulator_env()
        _create_topic_and_subscription()

    async def test_consumer_processes_valid_fhir_note(self) -> None:
        """Publish a valid FHIR note and verify the consumer processes it."""
        from src.ingestion.pubsub_consumer import run_consumer
        from src.ingestion.schemas import NoteRecord

        processed: list[NoteRecord] = []

        async def handler(note: NoteRecord) -> None:
            processed.append(note)

        _publish_fhir_note("integration-note-001")

        # Patch settings to use emulator subscription
        import unittest.mock as mock

        from src.core.config import Settings

        with mock.patch("src.ingestion.pubsub_consumer.get_settings") as mock_settings:
            mock_settings.return_value = Settings(
                gcp_project_id=GCP_PROJECT,
                pubsub_subscription_notes=SUBSCRIPTION_ID,
                pubsub_emulator_host=PUBSUB_EMULATOR_HOST,
            )

            # Run consumer with a short timeout — it will process the published message
            try:
                await asyncio.wait_for(run_consumer(handler, max_messages=1), timeout=15.0)
            except TimeoutError:
                pass  # expected — consumer runs indefinitely

        assert len(processed) >= 1
        assert processed[0].note_id == "integration-note-001"
        assert processed[0].patient_id == "patient-integration-note-001"

    async def test_consumer_routes_invalid_fhir_to_dlq(self) -> None:
        """Publish a malformed FHIR note and verify it's routed to DLQ."""
        import unittest.mock as mock

        from google.cloud import pubsub_v1

        from src.core.config import Settings

        _setup_emulator_env()
        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(GCP_PROJECT, TOPIC_ID)

        # Note without subject (poison pill)
        bad_fhir = {
            "resourceType": "DocumentReference",
            "id": "bad-note",
            "status": "current",
            "text": {"status": "generated", "div": "<div>Some text.</div>"},
            # Missing subject → MissingPatientReferenceError
        }
        publisher.publish(topic_path, json.dumps(bad_fhir).encode()).result(timeout=10)

        dlq_calls: list[tuple[str, str, str]] = []

        async def handler(note: object) -> None:
            pass

        async def dlq_publisher(msg_id: str, payload: str, reason: str) -> None:
            dlq_calls.append((msg_id, payload, reason))

        from src.ingestion.pubsub_consumer import run_consumer

        with mock.patch("src.ingestion.pubsub_consumer.get_settings") as mock_settings:
            mock_settings.return_value = Settings(
                gcp_project_id=GCP_PROJECT,
                pubsub_subscription_notes=SUBSCRIPTION_ID,
                pubsub_emulator_host=PUBSUB_EMULATOR_HOST,
            )
            try:
                await asyncio.wait_for(
                    run_consumer(handler, dlq_publisher=dlq_publisher, max_messages=1),
                    timeout=15.0,
                )
            except TimeoutError:
                pass

        assert len(dlq_calls) >= 1

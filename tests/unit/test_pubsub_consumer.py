import json
import os
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tenacity import RetryError

from src.core.exceptions import DLQError, FHIRParseError
from src.ingestion.pubsub_consumer import (
    CircuitBreaker,
    _build_subscriber,
    run_consumer,
)
from src.ingestion.schemas import NoteRecord, NoteType

# ── Helpers ───────────────────────────────────────────────────────────────────


class _StopLoopError(Exception):  # noqa: N818
    """Breaks out of run_consumer's infinite loop during testing."""


def _make_message(message_id: str, data: dict | None = None, raw: bytes | None = None) -> MagicMock:
    msg = MagicMock()
    msg.message.message_id = message_id
    msg.message.data = raw if raw is not None else json.dumps(data or {}).encode()
    msg.ack_id = f"ack-{message_id}"
    return msg


def _make_pull_response(messages: list[MagicMock]) -> MagicMock:
    resp = MagicMock()
    resp.received_messages = messages
    return resp


def _empty_pull_response() -> MagicMock:
    resp = MagicMock()
    resp.received_messages = []
    return resp


def _sample_note() -> NoteRecord:
    return NoteRecord(
        note_id="note-1",
        patient_id="patient-1",
        note_type=NoteType.PROGRESS_NOTE,
        authored_at=datetime.now(tz=UTC),
        raw_text="Patient presents with chest pain.",
        source="fhir",
    )


# ── CircuitBreaker ────────────────────────────────────────────────────────────


class TestCircuitBreaker:
    def test_initial_state_is_closed(self) -> None:
        cb = CircuitBreaker(threshold=3, cooldown=5.0)
        assert cb._open is False
        assert cb._failures == 0

    def test_does_not_open_before_threshold(self) -> None:
        cb = CircuitBreaker(threshold=3, cooldown=5.0)
        cb.record_failure()
        cb.record_failure()
        assert cb._open is False

    def test_opens_at_threshold(self) -> None:
        cb = CircuitBreaker(threshold=3, cooldown=5.0)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb._open is True

    def test_record_success_resets_state(self) -> None:
        cb = CircuitBreaker(threshold=2, cooldown=5.0)
        cb.record_failure()
        cb.record_failure()
        assert cb._open is True
        cb.record_success()
        assert cb._open is False
        assert cb._failures == 0

    async def test_wait_if_open_sleeps_and_resets(self) -> None:
        cb = CircuitBreaker(threshold=1, cooldown=0.001)
        cb.record_failure()
        assert cb._open is True
        await cb.wait_if_open()
        assert cb._open is False

    async def test_wait_if_closed_is_instant(self) -> None:
        cb = CircuitBreaker(threshold=10, cooldown=60.0)
        await cb.wait_if_open()
        assert cb._open is False


# ── _build_subscriber ─────────────────────────────────────────────────────────


class TestBuildSubscriber:
    def test_sets_emulator_env_when_enabled(self) -> None:
        with (
            patch("src.ingestion.pubsub_consumer.get_settings") as mock_settings,
            patch("src.ingestion.pubsub_consumer.pubsub_v1.SubscriberClient"),
        ):
            mock_settings.return_value.pubsub_emulator_enabled = True
            mock_settings.return_value.pubsub_emulator_host = "localhost:8085"
            _build_subscriber()
            assert os.environ.get("PUBSUB_EMULATOR_HOST") == "localhost:8085"

    def test_no_emulator_env_when_disabled(self) -> None:
        os.environ.pop("PUBSUB_EMULATOR_HOST", None)
        with (
            patch("src.ingestion.pubsub_consumer.get_settings") as mock_settings,
            patch("src.ingestion.pubsub_consumer.pubsub_v1.SubscriberClient"),
        ):
            mock_settings.return_value.pubsub_emulator_enabled = False
            _build_subscriber()
            assert "PUBSUB_EMULATOR_HOST" not in os.environ


# ── run_consumer ──────────────────────────────────────────────────────────────


class TestRunConsumer:
    async def test_successful_message_is_acked(self) -> None:
        handler = AsyncMock()
        msg = _make_message("m1", {"resourceType": "DocumentReference"})

        mock_sub = MagicMock()
        mock_sub.subscription_path.return_value = "projects/p/subscriptions/s"
        mock_sub.pull.side_effect = [_make_pull_response([msg]), _StopLoopError()]

        with (
            patch("src.ingestion.pubsub_consumer._build_subscriber", return_value=mock_sub),
            patch(
                "src.ingestion.pubsub_consumer._process_with_retry", new_callable=AsyncMock
            ) as mock_proc,
            patch("src.ingestion.pubsub_consumer.asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_proc.return_value = None

            with pytest.raises(_StopLoopError):
                await run_consumer(handler)

        mock_sub.acknowledge.assert_called_once()
        acked = mock_sub.acknowledge.call_args[1]["request"]["ack_ids"]
        assert "ack-m1" in acked

    async def test_json_decode_error_acks_and_routes_to_dlq(self) -> None:
        handler = AsyncMock()
        dlq_publisher = AsyncMock()
        msg = _make_message("m2", raw=b"not json {{{")

        mock_sub = MagicMock()
        mock_sub.subscription_path.return_value = "projects/p/subscriptions/s"
        mock_sub.pull.side_effect = [_make_pull_response([msg]), _StopLoopError()]

        with (
            patch("src.ingestion.pubsub_consumer._build_subscriber", return_value=mock_sub),
            patch("src.ingestion.pubsub_consumer.asyncio.sleep", new_callable=AsyncMock),
        ):
            with pytest.raises(_StopLoopError):
                await run_consumer(handler, dlq_publisher=dlq_publisher)

        mock_sub.acknowledge.assert_called_once()
        dlq_publisher.assert_awaited_once()

    async def test_fhir_parse_error_acks_and_routes_to_dlq(self) -> None:
        handler = AsyncMock()
        dlq_publisher = AsyncMock()
        msg = _make_message("m3", {"resourceType": "DocumentReference"})

        mock_sub = MagicMock()
        mock_sub.subscription_path.return_value = "projects/p/subscriptions/s"
        mock_sub.pull.side_effect = [_make_pull_response([msg]), _StopLoopError()]

        with (
            patch("src.ingestion.pubsub_consumer._build_subscriber", return_value=mock_sub),
            patch(
                "src.ingestion.pubsub_consumer._process_with_retry",
                new_callable=AsyncMock,
                side_effect=FHIRParseError("m3", "bad FHIR"),
            ),
            patch("src.ingestion.pubsub_consumer.asyncio.sleep", new_callable=AsyncMock),
        ):
            with pytest.raises(_StopLoopError):
                await run_consumer(handler, dlq_publisher=dlq_publisher)

        acked = mock_sub.acknowledge.call_args[1]["request"]["ack_ids"]
        assert "ack-m3" in acked
        dlq_publisher.assert_awaited_once()

    async def test_retry_error_nacks_message(self) -> None:
        handler = AsyncMock()
        msg = _make_message("m4", {"resourceType": "DocumentReference"})

        mock_sub = MagicMock()
        mock_sub.subscription_path.return_value = "projects/p/subscriptions/s"
        mock_sub.pull.side_effect = [_make_pull_response([msg]), _StopLoopError()]

        retry_exc = RetryError(last_attempt=MagicMock())

        with (
            patch("src.ingestion.pubsub_consumer._build_subscriber", return_value=mock_sub),
            patch(
                "src.ingestion.pubsub_consumer._process_with_retry",
                new_callable=AsyncMock,
                side_effect=retry_exc,
            ),
            patch("src.ingestion.pubsub_consumer.asyncio.sleep", new_callable=AsyncMock),
        ):
            with pytest.raises(_StopLoopError):
                await run_consumer(handler)

        mock_sub.modify_ack_deadline.assert_called_once()
        nacked = mock_sub.modify_ack_deadline.call_args[1]["request"]["ack_ids"]
        assert "ack-m4" in nacked

    async def test_dlq_error_acks_message(self) -> None:
        handler = AsyncMock()
        msg = _make_message("m5", {"resourceType": "DocumentReference"})

        mock_sub = MagicMock()
        mock_sub.subscription_path.return_value = "projects/p/subscriptions/s"
        mock_sub.pull.side_effect = [_make_pull_response([msg]), _StopLoopError()]

        with (
            patch("src.ingestion.pubsub_consumer._build_subscriber", return_value=mock_sub),
            patch(
                "src.ingestion.pubsub_consumer._process_with_retry",
                new_callable=AsyncMock,
                side_effect=DLQError("m5", "dlq reason"),
            ),
            patch("src.ingestion.pubsub_consumer.asyncio.sleep", new_callable=AsyncMock),
        ):
            with pytest.raises(_StopLoopError):
                await run_consumer(handler)

        acked = mock_sub.acknowledge.call_args[1]["request"]["ack_ids"]
        assert "ack-m5" in acked

    async def test_empty_response_does_not_ack(self) -> None:
        handler = AsyncMock()

        mock_sub = MagicMock()
        mock_sub.subscription_path.return_value = "projects/p/subscriptions/s"
        mock_sub.pull.side_effect = [_empty_pull_response(), _StopLoopError()]

        with (
            patch("src.ingestion.pubsub_consumer._build_subscriber", return_value=mock_sub),
            patch("src.ingestion.pubsub_consumer.asyncio.sleep", new_callable=AsyncMock),
        ):
            with pytest.raises(_StopLoopError):
                await run_consumer(handler)

        mock_sub.acknowledge.assert_not_called()

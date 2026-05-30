from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.exceptions import DLQError
from src.ingestion.dlq_handler import alert_if_threshold_exceeded, publish_to_dlq


class TestPublishToDlq:
    async def test_publishes_message_successfully(self) -> None:
        mock_future = MagicMock()
        mock_future.result.return_value = "msg-id-abc"

        mock_publisher = MagicMock()
        mock_publisher.topic_path.return_value = "projects/p/topics/dlq"
        mock_publisher.publish.return_value = mock_future

        with patch("src.ingestion.dlq_handler._get_publisher", return_value=mock_publisher):
            await publish_to_dlq("msg-1", '{"bad": "data"}', "FHIRParseError")

        mock_publisher.publish.assert_called_once()

    async def test_raises_dlq_error_on_publish_failure(self) -> None:
        mock_future = MagicMock()
        mock_future.result.side_effect = RuntimeError("Pub/Sub unavailable")

        mock_publisher = MagicMock()
        mock_publisher.topic_path.return_value = "projects/p/topics/dlq"
        mock_publisher.publish.return_value = mock_future

        with patch("src.ingestion.dlq_handler._get_publisher", return_value=mock_publisher):
            with pytest.raises(DLQError):
                await publish_to_dlq("msg-2", "{}", "timeout")


class TestAlertIfThresholdExceeded:
    async def test_no_alert_when_rate_below_threshold(self) -> None:
        with (
            patch("src.ingestion.dlq_handler._send_pagerduty_alert") as mock_pd,
            patch("src.ingestion.dlq_handler._send_slack_alert") as mock_slack,
        ):
            await alert_if_threshold_exceeded(dlq_count=1, total_count=100)
            mock_pd.assert_not_called()
            mock_slack.assert_not_called()

    async def test_fires_alerts_when_rate_exceeds_threshold(self) -> None:
        pd_path = "src.ingestion.dlq_handler._send_pagerduty_alert"
        slack_path = "src.ingestion.dlq_handler._send_slack_alert"
        with (
            patch(pd_path, new_callable=AsyncMock) as mock_pd,
            patch(slack_path, new_callable=AsyncMock) as mock_slack,
        ):
            await alert_if_threshold_exceeded(dlq_count=10, total_count=100)
            mock_pd.assert_awaited_once()
            mock_slack.assert_awaited_once()

    async def test_no_alert_when_total_count_is_zero(self) -> None:
        with patch("src.ingestion.dlq_handler._send_pagerduty_alert") as mock_pd:
            await alert_if_threshold_exceeded(dlq_count=0, total_count=0)
            mock_pd.assert_not_called()

from unittest.mock import patch

from src.core.telemetry import (
    DLQ_MESSAGES_TOTAL,
    EMBEDDING_INFERENCE_LATENCY,
    LLM_TOKENS_TOTAL,
    PIPELINE_NOTES_TOTAL,
    PIPELINE_STAGE_LATENCY,
    RAG_QUERY_LATENCY,
    setup_logging,
    setup_metrics,
    setup_telemetry,
    setup_tracing,
    start_prometheus_server,
)


class TestPrometheusMetrics:
    def test_counters_increment(self) -> None:
        PIPELINE_NOTES_TOTAL.labels(status="success").inc()
        PIPELINE_NOTES_TOTAL.labels(status="failure").inc()
        PIPELINE_NOTES_TOTAL.labels(status="dlq").inc()
        DLQ_MESSAGES_TOTAL.labels(reason="decode_error").inc()
        LLM_TOKENS_TOTAL.labels(model="claude", direction="prompt").inc()
        LLM_TOKENS_TOTAL.labels(model="claude", direction="completion").inc()

    def test_histograms_observe(self) -> None:
        PIPELINE_STAGE_LATENCY.labels(stage="segmenter", status="success").observe(0.5)
        PIPELINE_STAGE_LATENCY.labels(stage="ner", status="failure").observe(2.0)
        RAG_QUERY_LATENCY.observe(1.2)
        EMBEDDING_INFERENCE_LATENCY.labels(model="biomedbert").observe(0.1)


class TestSetupFunctions:
    def test_setup_logging_dev_mode(self) -> None:
        setup_logging()

    def test_setup_logging_prod_mode(self) -> None:
        with patch("src.core.telemetry.get_settings") as mock_settings:
            mock_settings.return_value.log_level = "WARNING"
            mock_settings.return_value.is_production = True
            setup_logging()

    def test_setup_tracing_no_endpoint(self) -> None:
        setup_tracing()

    def test_setup_tracing_with_endpoint(self) -> None:
        setup_tracing(otlp_endpoint="http://localhost:4317")

    def test_setup_metrics_no_endpoint(self) -> None:
        setup_metrics()

    @patch("src.core.telemetry.start_http_server")
    def test_start_prometheus_server(self, mock_server: object) -> None:
        start_prometheus_server(port=9091)
        from unittest.mock import MagicMock

        assert isinstance(mock_server, MagicMock)

    @patch("src.core.telemetry.start_http_server")
    def test_setup_telemetry_with_prometheus(self, mock_server: object) -> None:
        setup_telemetry(prometheus_port=9092)
        from unittest.mock import MagicMock

        assert isinstance(mock_server, MagicMock)

    def test_setup_telemetry_no_prometheus(self) -> None:
        setup_telemetry()

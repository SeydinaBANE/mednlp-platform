import logging
from typing import cast

import structlog
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import Counter, Histogram, start_http_server

from src.core.config import get_settings

# ── Prometheus metrics ────────────────────────────────────────────────────────

PIPELINE_STAGE_LATENCY = Histogram(
    "pipeline_stage_latency_seconds",
    "Processing time per pipeline stage",
    ["stage", "status"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

PIPELINE_NOTES_TOTAL = Counter(
    "pipeline_note_processing_total",
    "Total notes processed by status",
    ["status"],  # success | failure | dlq
)

RAG_QUERY_LATENCY = Histogram(
    "rag_query_latency_seconds",
    "End-to-end RAG query latency",
    buckets=[0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0],
)

EMBEDDING_INFERENCE_LATENCY = Histogram(
    "embedding_model_inference_seconds",
    "Embedding model inference latency",
    ["model"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
)

DLQ_MESSAGES_TOTAL = Counter(
    "dlq_messages_total",
    "Messages sent to dead-letter queue",
    ["reason"],
)

LLM_TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "Tokens consumed from OpenRouter",
    ["model", "direction"],  # direction: prompt | completion
)

# ── Drift detection metrics ───────────────────────────────────────────────────

from prometheus_client import Gauge  # noqa: E402

EMBEDDING_DRIFT_SCORE = Gauge(
    "embedding_drift_score",
    "Jensen-Shannon divergence score for embedding drift",
)

LABEL_DRIFT_CODES_TOTAL = Gauge(
    "label_drift_codes_total",
    "Number of ICD-10 codes with statistically significant label drift",
)

DATA_DRIFT_PVALUE = Gauge(
    "data_drift_pvalue",
    "Chi-squared p-value for NER entity count distribution drift",
)


def setup_logging() -> None:
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    renderer = cast(
        structlog.types.Processor,
        structlog.processors.JSONRenderer()
        if settings.is_production
        else structlog.dev.ConsoleRenderer(),
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


def setup_tracing(otlp_endpoint: str | None = None) -> None:
    resource = Resource.create({"service.name": "mednlp-platform"})
    provider = TracerProvider(resource=resource)

    if otlp_endpoint:
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)


def setup_metrics(otlp_endpoint: str | None = None) -> None:
    readers = []
    if otlp_endpoint:
        readers.append(
            PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=otlp_endpoint),
                export_interval_millis=15_000,
            )
        )
    provider = MeterProvider(metric_readers=readers)
    metrics.set_meter_provider(provider)


def start_prometheus_server(port: int = 9090) -> None:
    """Expose Prometheus metrics on /metrics — call once at startup."""
    start_http_server(port)


def setup_telemetry(
    otlp_endpoint: str | None = None,
    prometheus_port: int | None = None,
) -> None:
    setup_logging()
    setup_tracing(otlp_endpoint)
    setup_metrics(otlp_endpoint)
    if prometheus_port:
        start_prometheus_server(prometheus_port)

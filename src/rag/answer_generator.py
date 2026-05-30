"""Generate RAG answers via OpenRouter, with streaming support."""

import time
from collections.abc import AsyncIterator

import structlog
from openai.types.chat import ChatCompletionMessageParam

from src.core.config import get_settings
from src.core.openrouter_client import complete, stream_complete
from src.core.schemas import QueryResponse
from src.core.telemetry import RAG_QUERY_LATENCY
from src.rag.context_builder import BuiltContext
from src.rag.guardrails import apply_guardrails
from src.rag.prompt_templates import get_template

logger = structlog.get_logger(__name__)


def _build_messages(
    query: str,
    context: BuiltContext,
    template_name: str,
    template_version: str,
) -> list[ChatCompletionMessageParam]:
    tmpl = get_template(template_name, template_version)
    user_content = tmpl["user"].format(
        context=context.context_text,
        question=query,
    )
    return [
        {"role": "system", "content": tmpl["system"]},
        {"role": "user", "content": user_content},
    ]


async def generate(
    query: str,
    context: BuiltContext,
    *,
    model: str | None = None,
    template_name: str = "clinical_qa",
    template_version: str = "v1",
    strict_guardrails: bool = False,
) -> QueryResponse:
    """Generate a complete (non-streaming) RAG answer."""
    settings = get_settings()
    model = model or settings.openrouter_model_heavy
    messages = _build_messages(query, context, template_name, template_version)

    start = time.perf_counter()
    answer = await complete(messages, model=model)
    latency_ms = int((time.perf_counter() - start) * 1000)

    answer = apply_guardrails(answer, strict=strict_guardrails)
    RAG_QUERY_LATENCY.observe(latency_ms / 1000)

    logger.info(
        "rag_answer_generated",
        model=model,
        latency_ms=latency_ms,
        citations=len(context.citations),
    )

    return QueryResponse(
        answer=answer,
        sources=context.citations,
        model=model,
        latency_ms=latency_ms,
    )


async def stream_generate(
    query: str,
    context: BuiltContext,
    *,
    model: str | None = None,
    template_name: str = "clinical_qa",
    template_version: str = "v1",
) -> AsyncIterator[str]:
    """Stream answer tokens. Yields raw text chunks (not SSE formatted)."""
    settings = get_settings()
    model = model or settings.openrouter_model_heavy
    messages = _build_messages(query, context, template_name, template_version)

    logger.info("rag_stream_started", model=model)

    async for chunk in stream_complete(messages, model=model):
        yield chunk

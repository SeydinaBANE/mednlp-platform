from collections.abc import AsyncIterator
from functools import lru_cache

import structlog
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from src.core.config import get_settings
from src.core.exceptions import LLMError

logger = structlog.get_logger(__name__)


@lru_cache
def get_openrouter_client() -> AsyncOpenAI:
    settings = get_settings()
    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
        default_headers={
            "HTTP-Referer": settings.app_url,
            "X-Title": "MedNLP Platform",
        },
    )


async def complete(
    messages: list[ChatCompletionMessageParam],
    model: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> str:
    """Single-turn completion via OpenRouter."""
    settings = get_settings()
    model = model or settings.openrouter_model_heavy
    client = get_openrouter_client()

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = response.choices[0].message.content or ""
        logger.debug(
            "openrouter_completion",
            model=model,
            prompt_tokens=response.usage.prompt_tokens if response.usage else None,
            completion_tokens=response.usage.completion_tokens if response.usage else None,
        )
        return content
    except Exception as exc:
        status = getattr(exc, "status_code", 0)
        raise LLMError(model=model, status_code=status, detail=str(exc)) from exc


async def stream_complete(
    messages: list[ChatCompletionMessageParam],
    model: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.1,
) -> AsyncIterator[str]:
    """Streaming completion via OpenRouter — yields token chunks."""
    settings = get_settings()
    model = model or settings.openrouter_model_heavy
    client = get_openrouter_client()

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as exc:
        status = getattr(exc, "status_code", 0)
        raise LLMError(model=model, status_code=status, detail=str(exc)) from exc

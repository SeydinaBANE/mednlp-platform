from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.exceptions import LLMError
from src.core.openrouter_client import complete, get_openrouter_client, stream_complete


class TestGetOpenRouterClient:
    def test_returns_async_openai_instance(self) -> None:
        get_openrouter_client.cache_clear()
        client = get_openrouter_client()
        assert client is not None
        get_openrouter_client.cache_clear()

    def test_is_cached(self) -> None:
        get_openrouter_client.cache_clear()
        c1 = get_openrouter_client()
        c2 = get_openrouter_client()
        assert c1 is c2
        get_openrouter_client.cache_clear()


class TestComplete:
    async def test_returns_content_on_success(self) -> None:
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "diagnosis: chest pain"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("src.core.openrouter_client.get_openrouter_client", return_value=mock_client):
            result = await complete([{"role": "user", "content": "What is the diagnosis?"}])

        assert result == "diagnosis: chest pain"

    async def test_uses_default_heavy_model_when_none(self) -> None:
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "ok"
        mock_response.usage = None

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("src.core.openrouter_client.get_openrouter_client", return_value=mock_client):
            await complete([{"role": "user", "content": "hi"}], model=None)

        call_kwargs = mock_client.chat.completions.create.call_args
        assert "anthropic/claude-3.5-sonnet" in str(call_kwargs)

    async def test_raises_llm_error_on_api_failure(self) -> None:
        exc = Exception("upstream error")
        exc.status_code = 503  # type: ignore[attr-defined]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=exc)

        with patch("src.core.openrouter_client.get_openrouter_client", return_value=mock_client):
            with pytest.raises(LLMError) as exc_info:
                await complete([{"role": "user", "content": "hello"}])

        assert exc_info.value.status_code == 503

    async def test_llm_error_status_code_zero_when_absent(self) -> None:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))

        with patch("src.core.openrouter_client.get_openrouter_client", return_value=mock_client):
            with pytest.raises(LLMError) as exc_info:
                await complete([{"role": "user", "content": "hello"}])

        assert exc_info.value.status_code == 0


class TestStreamComplete:
    async def test_yields_non_null_chunks(self) -> None:
        chunks = ["Hello", " world", None, "!"]

        async def _stream() -> object:
            for text in chunks:
                chunk = MagicMock()
                chunk.choices[0].delta.content = text
                yield chunk

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_stream())

        with patch("src.core.openrouter_client.get_openrouter_client", return_value=mock_client):
            result = []
            async for piece in stream_complete([{"role": "user", "content": "hi"}]):
                result.append(piece)

        assert result == ["Hello", " world", "!"]

    async def test_raises_llm_error_on_stream_failure(self) -> None:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("stream err"))

        with patch("src.core.openrouter_client.get_openrouter_client", return_value=mock_client):
            with pytest.raises(LLMError):
                async for _ in stream_complete([{"role": "user", "content": "hi"}]):
                    pass

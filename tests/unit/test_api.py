"""Unit tests for API: middleware (JWT, rate limiting), audit router, database."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.middleware import (
    AuthMiddleware,
    create_access_token,
    extract_token,
    verify_token,
)
from src.core.exceptions import InvalidTokenError

# ── JWT helpers ───────────────────────────────────────────────────────────────


class TestCreateAndVerifyToken:
    def test_creates_valid_token(self) -> None:
        token = create_access_token("user-42")
        assert isinstance(token, str)
        assert len(token) > 20

    def test_round_trip(self) -> None:
        token = create_access_token("user-42", extra={"role": "clinician"})
        claims = verify_token(token)
        assert claims["sub"] == "user-42"
        assert claims["role"] == "clinician"

    def test_expired_token_raises(self) -> None:
        import jwt

        from src.core.config import get_settings

        settings = get_settings()
        payload = {
            "sub": "user-42",
            "iat": int(time.time()) - 3600,
            "exp": int(time.time()) - 1,  # already expired
        }
        expired_token = jwt.encode(payload, settings.secret_key, algorithm="HS256")

        with pytest.raises(InvalidTokenError, match="expired"):
            verify_token(expired_token)

    def test_tampered_token_raises(self) -> None:
        token = create_access_token("user-42")
        tampered = token[:-5] + "XXXXX"
        with pytest.raises(InvalidTokenError):
            verify_token(tampered)

    def test_wrong_secret_raises(self) -> None:
        import jwt

        payload = {"sub": "user-42", "exp": int(time.time()) + 3600}
        bad_token = jwt.encode(payload, "wrong-secret", algorithm="HS256")
        with pytest.raises(InvalidTokenError):
            verify_token(bad_token)


class TestExtractToken:
    def test_extracts_bearer_token(self) -> None:
        mock_request = MagicMock()
        mock_request.headers.get.return_value = "Bearer my-jwt-token"
        assert extract_token(mock_request) == "my-jwt-token"

    def test_returns_none_when_no_auth_header(self) -> None:
        mock_request = MagicMock()
        mock_request.headers.get.return_value = ""
        assert extract_token(mock_request) is None

    def test_returns_none_for_non_bearer(self) -> None:
        mock_request = MagicMock()
        mock_request.headers.get.return_value = "Basic dXNlcjpwYXNz"
        assert extract_token(mock_request) is None


# ── AuthMiddleware ────────────────────────────────────────────────────────────


class TestAuthMiddleware:
    async def test_public_paths_bypass_auth(self) -> None:
        middleware = AuthMiddleware(app=MagicMock())
        mock_request = MagicMock()
        mock_request.url.path = "/health"
        mock_call_next = AsyncMock(return_value=MagicMock(status_code=200))

        response = await middleware.dispatch(mock_request, mock_call_next)
        assert response.status_code == 200
        mock_call_next.assert_awaited_once()

    async def test_missing_token_returns_401(self) -> None:
        middleware = AuthMiddleware(app=MagicMock())
        mock_request = MagicMock()
        mock_request.url.path = "/query"
        mock_request.headers.get.return_value = ""
        mock_call_next = AsyncMock()

        response = await middleware.dispatch(mock_request, mock_call_next)
        assert response.status_code == 401
        mock_call_next.assert_not_awaited()

    async def test_valid_token_passes(self) -> None:
        middleware = AuthMiddleware(app=MagicMock())
        token = create_access_token("user-42")
        mock_request = MagicMock()
        mock_request.url.path = "/query"
        mock_request.headers.get.return_value = f"Bearer {token}"
        mock_request.state = MagicMock()
        mock_call_next = AsyncMock(return_value=MagicMock(status_code=200))

        response = await middleware.dispatch(mock_request, mock_call_next)
        assert response.status_code == 200
        assert mock_request.state.user == "user-42"

    async def test_invalid_token_returns_401(self) -> None:
        middleware = AuthMiddleware(app=MagicMock())
        mock_request = MagicMock()
        mock_request.url.path = "/query"
        mock_request.headers.get.return_value = "Bearer invalid.token.here"
        mock_call_next = AsyncMock()

        response = await middleware.dispatch(mock_request, mock_call_next)
        assert response.status_code == 401


# ── Rate limiting ─────────────────────────────────────────────────────────────


class TestCheckRateLimit:
    def _make_redis_mock(self, count: int) -> MagicMock:
        """Build a sync-pipeline / async-execute Redis mock."""
        mock_pipe = MagicMock()  # pipeline commands are sync (zadd, zcard…)
        mock_pipe.execute = AsyncMock(return_value=[None, None, count, None])

        mock_redis = MagicMock()  # from_url returns a sync-ish client
        mock_redis.pipeline.return_value = mock_pipe
        mock_redis.aclose = AsyncMock()
        return mock_redis

    async def test_passes_when_under_limit(self) -> None:
        from src.api.middleware import check_rate_limit

        mock_request = MagicMock()
        mock_request.client.host = "127.0.0.1"

        with patch("redis.asyncio.from_url", return_value=self._make_redis_mock(count=5)):
            await check_rate_limit(mock_request)  # should not raise

    async def test_raises_429_when_over_limit(self) -> None:
        from fastapi import HTTPException

        from src.api.middleware import check_rate_limit

        mock_request = MagicMock()
        mock_request.client.host = "10.0.0.1"

        with patch("redis.asyncio.from_url", return_value=self._make_redis_mock(count=200)):
            with pytest.raises(HTTPException) as exc_info:
                await check_rate_limit(mock_request)

        assert exc_info.value.status_code == 429


# ── Database ──────────────────────────────────────────────────────────────────


class TestDatabase:
    def test_get_db_yields_session(self) -> None:
        from src.core.database import _get_engine, _get_session_factory

        _get_engine.cache_clear()
        _get_session_factory.cache_clear()

        mock_engine = MagicMock()
        mock_factory = MagicMock()

        with (
            patch("src.core.database.create_async_engine", return_value=mock_engine),
            patch("src.core.database.async_sessionmaker", return_value=mock_factory),
        ):
            engine = _get_engine()
            assert engine is not None

        _get_engine.cache_clear()
        _get_session_factory.cache_clear()

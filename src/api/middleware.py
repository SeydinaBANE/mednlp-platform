"""FastAPI middleware — JWT authentication, Redis rate limiting, OTel request tracing."""

import time
from collections.abc import Callable
from typing import Any

import jwt
import structlog
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.config import get_settings
from src.core.exceptions import AuthError, InvalidTokenError
from src.core.telemetry import PIPELINE_STAGE_LATENCY

logger = structlog.get_logger(__name__)

_BEARER_PREFIX = "Bearer "
_PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc", "/metrics"}


# ── JWT helpers ───────────────────────────────────────────────────────────────


def create_access_token(subject: str, extra: dict[str, Any] | None = None) -> str:
    """Create a signed HS256 JWT for `subject`."""
    settings = get_settings()
    payload: dict[str, Any] = {
        "sub": subject,
        "iat": int(time.time()),
        "exp": int(time.time()) + settings.jwt_expire_minutes * 60,
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


def verify_token(token: str) -> dict[str, Any]:
    """Decode and verify a JWT. Raises InvalidTokenError on failure."""
    settings = get_settings()
    try:
        return jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
    except jwt.ExpiredSignatureError as exc:
        raise InvalidTokenError("Token has expired") from exc
    except jwt.InvalidTokenError as exc:
        raise InvalidTokenError(f"Invalid token: {exc}") from exc


def extract_token(request: Request) -> str | None:
    """Return the Bearer token from the Authorization header, or None."""
    auth: str = request.headers.get("Authorization", "")
    if auth.startswith(_BEARER_PREFIX):
        return auth[len(_BEARER_PREFIX) :]
    return None


# ── Rate limiting ─────────────────────────────────────────────────────────────


async def check_rate_limit(request: Request) -> None:
    """Sliding-window rate limiter backed by Redis.

    Key: `rl:{client_ip}`, window: 60s, limit from settings.
    Raises HTTP 429 when the limit is exceeded.
    """
    settings = get_settings()
    client_ip = request.client.host if request.client else "unknown"
    redis_key = f"rl:{client_ip}"

    try:
        import redis.asyncio as aioredis

        redis = aioredis.from_url(settings.redis_url, decode_responses=True)
        pipe = redis.pipeline()
        now = int(time.time())
        window_start = now - 60

        pipe.zremrangebyscore(redis_key, 0, window_start)
        pipe.zadd(redis_key, {str(now): now})
        pipe.zcard(redis_key)
        pipe.expire(redis_key, 61)
        results = await pipe.execute()
        await redis.aclose()  # type: ignore[attr-defined]

        count = int(results[2])
        if count > settings.rate_limit_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {settings.rate_limit_per_minute}/min",
                headers={"Retry-After": "60"},
            )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("rate_limit_check_failed", error=str(exc))


# ── Starlette middleware ───────────────────────────────────────────────────────


class AuthMiddleware(BaseHTTPMiddleware):
    """Enforce JWT authentication on all non-public paths."""

    async def dispatch(self, request: Request, call_next: Callable[..., Any]) -> Any:  # noqa: ANN401
        if request.url.path in _PUBLIC_PATHS or request.url.path.startswith("/docs"):
            return await call_next(request)

        token = extract_token(request)
        if not token:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Authorization header missing or malformed"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        try:
            claims = verify_token(token)
            request.state.user = claims.get("sub", "unknown")
        except (AuthError, InvalidTokenError) as exc:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": str(exc)},
                headers={"WWW-Authenticate": "Bearer"},
            )

        return await call_next(request)


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Log request latency and attach trace context to structlog."""

    async def dispatch(self, request: Request, call_next: Callable[..., Any]) -> Any:  # noqa: ANN401
        start = time.perf_counter()
        path = request.url.path

        import structlog

        with structlog.contextvars.bound_contextvars(
            path=path,
            method=request.method,
            client=request.client.host if request.client else "unknown",
        ):
            response = await call_next(request)
            elapsed = time.perf_counter() - start
            status_code = str(response.status_code)
            PIPELINE_STAGE_LATENCY.labels(stage="http_request", status=status_code).observe(elapsed)
            latency_ms = int(elapsed * 1000)
            logger.info("http_request", status_code=response.status_code, latency_ms=latency_ms)
            return response

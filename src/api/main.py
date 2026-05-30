"""FastAPI application entry point — registers all routers, middleware, and lifecycle hooks."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.middleware import AuthMiddleware, RequestTracingMiddleware
from src.api.routers.audit import router as audit_router
from src.api.routers.query import router as query_router
from src.core.config import get_settings
from src.core.schemas import HealthResponse
from src.core.telemetry import setup_telemetry

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    setup_telemetry(prometheus_port=9090 if settings.is_production else None)
    logger.info("mednlp_api_started", env=settings.env)
    yield
    logger.info("mednlp_api_stopped")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="MedNLP Platform API",
        description="Clinical Intelligence Platform — RAG, NLP pipeline, fine-tuning",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501"],  # Streamlit portal
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

    # ── Tracing + Auth ────────────────────────────────────────────────────────
    app.add_middleware(RequestTracingMiddleware)
    app.add_middleware(AuthMiddleware)

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(query_router)
    app.include_router(audit_router)

    # ── Health ────────────────────────────────────────────────────────────────
    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health() -> HealthResponse:
        db_status = await _check_db()
        redis_status = await _check_redis(settings.redis_url)
        qdrant_status = await _check_qdrant(settings.qdrant_host, settings.qdrant_port)
        return HealthResponse(
            status="ok" if all(s == "ok" for s in [db_status, redis_status]) else "degraded",
            version="0.1.0",
            db=db_status,
            redis=redis_status,
            qdrant=qdrant_status,
        )

    return app


async def _check_db() -> str:
    try:
        from sqlalchemy import text

        from src.core.database import _get_session_factory

        factory = _get_session_factory()
        async with factory() as session:
            await session.execute(text("SELECT 1"))
        return "ok"
    except Exception as exc:  # noqa: BLE001
        logger.warning("health_db_failed", error=str(exc))
        return "error"


async def _check_redis(redis_url: str) -> str:
    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(redis_url)
        await client.ping()
        await client.aclose()  # type: ignore[attr-defined]
        return "ok"
    except Exception as exc:  # noqa: BLE001
        logger.warning("health_redis_failed", error=str(exc))
        return "error"


async def _check_qdrant(host: str, port: int) -> str:
    try:
        from src.vector_store.client import get_qdrant_client

        client = get_qdrant_client()
        await client.get_collections()
        return "ok"
    except Exception as exc:  # noqa: BLE001
        logger.warning("health_qdrant_failed", error=str(exc))
        return "error"


app = create_app()

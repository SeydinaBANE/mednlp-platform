"""Unit tests for FastAPI app: health endpoint, audit router, and main app assembly."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import ASGITransport, AsyncClient

from src.api.main import _check_db, _check_qdrant, _check_redis, create_app
from src.core.models import AuditLog

# ── App factory ───────────────────────────────────────────────────────────────


class TestCreateApp:
    def test_app_has_expected_routes(self) -> None:
        app = create_app()
        routes = {r.path for r in app.routes}  # type: ignore[attr-defined]
        assert "/health" in routes
        assert "/query" in routes
        assert "/audit/note/{note_id}" in routes

    def test_app_has_middleware(self) -> None:
        app = create_app()
        # Starlette middleware stack is a list
        assert len(app.user_middleware) >= 2


# ── Health checks ─────────────────────────────────────────────────────────────


class TestHealthChecks:
    async def test_check_db_ok(self) -> None:
        mock_factory = MagicMock()
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock()
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch("src.core.database._get_session_factory", return_value=mock_factory):
            status = await _check_db()
        assert status == "ok"

    async def test_check_db_error_on_exception(self) -> None:
        with patch("src.core.database._get_session_factory", side_effect=RuntimeError("no db")):
            status = await _check_db()
        assert status == "error"

    async def test_check_redis_ok(self) -> None:
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock()
        mock_client.aclose = AsyncMock()

        with patch("redis.asyncio.from_url", return_value=mock_client):
            status = await _check_redis("redis://localhost:6379")
        assert status == "ok"

    async def test_check_redis_error_on_exception(self) -> None:
        with patch("redis.asyncio.from_url", side_effect=RuntimeError("no redis")):
            status = await _check_redis("redis://localhost:6379")
        assert status == "error"

    async def test_check_qdrant_ok(self) -> None:
        mock_client = AsyncMock()
        mock_client.get_collections = AsyncMock()

        with patch("src.vector_store.client.get_qdrant_client", return_value=mock_client):
            status = await _check_qdrant("localhost", 6333)
        assert status == "ok"

    async def test_check_qdrant_error_on_exception(self) -> None:
        with patch(
            "src.vector_store.client.get_qdrant_client", side_effect=RuntimeError("qdrant down")
        ):
            status = await _check_qdrant("localhost", 6333)
        assert status == "error"


class TestHealthEndpoint:
    async def test_health_returns_200_when_all_ok(self) -> None:
        app = create_app()

        with (
            patch("src.api.main._check_db", return_value="ok"),
            patch("src.api.main._check_redis", return_value="ok"),
            patch("src.api.main._check_qdrant", return_value="ok"),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["db"] == "ok"
        assert data["redis"] == "ok"

    async def test_health_returns_degraded_when_db_down(self) -> None:
        app = create_app()

        with (
            patch("src.api.main._check_db", return_value="error"),
            patch("src.api.main._check_redis", return_value="ok"),
            patch("src.api.main._check_qdrant", return_value="ok"),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/health")

        assert resp.status_code == 200
        assert resp.json()["status"] == "degraded"


# ── Audit router ──────────────────────────────────────────────────────────────


class TestAuditRouter:
    def _make_audit_row(self, note_id: str = "note-1") -> AuditLog:
        row = MagicMock(spec=AuditLog)
        row.id = 1
        row.actor = "user-42"
        row.action = "read"
        row.resource_type = "note"
        row.resource_id = note_id
        row.ip_address = "127.0.0.1"
        row.details = {}
        row.created_at = datetime.now(tz=UTC)
        return row

    async def test_get_note_audit_returns_200(self) -> None:
        from fastapi import FastAPI

        from src.api.routers.audit import router

        app = FastAPI()
        app.include_router(router)

        mock_row = self._make_audit_row()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_row]

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        async def override_get_db() -> object:
            yield mock_session

        from src.core.database import get_db

        app.dependency_overrides[get_db] = override_get_db

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/audit/note/note-1")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 0

    async def test_record_audit_creates_entry(self) -> None:
        from fastapi import FastAPI

        from src.api.routers.audit import router

        app = FastAPI()
        app.include_router(router)

        mock_entry = MagicMock(spec=AuditLog)
        mock_entry.id = 42
        mock_entry.created_at = datetime.now(tz=UTC)

        mock_session = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()
        mock_session.add = MagicMock()

        async def side_effect_refresh(obj: object) -> None:
            obj.id = 42  # type: ignore[attr-defined]
            obj.created_at = datetime.now(tz=UTC)  # type: ignore[attr-defined]

        mock_session.refresh.side_effect = side_effect_refresh

        async def override_get_db() -> object:
            yield mock_session

        from src.core.database import get_db

        app.dependency_overrides[get_db] = override_get_db

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/audit",
                params={
                    "actor": "user-42",
                    "action": "read",
                    "resource_type": "note",
                    "resource_id": "note-1",
                },
            )

        assert resp.status_code == 201

"""Integration tests — verify connectivity to Postgres and Redis.

Requires docker-compose.test.yml to be running (make test-integration).
"""

import os

import asyncpg
import redis.asyncio as aioredis

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5433/mednlp_test",
)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6380/0")


class TestPostgresConnectivity:
    async def test_can_connect_and_query(self) -> None:
        dsn = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://").replace(
            "postgresql://", "postgresql://"
        )
        conn = await asyncpg.connect(dsn)
        try:
            result = await conn.fetchval("SELECT 1")
            assert result == 1
        finally:
            await conn.close()

    async def test_mednlp_test_db_exists(self) -> None:
        dsn = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
        conn = await asyncpg.connect(dsn)
        try:
            db_name = await conn.fetchval("SELECT current_database()")
            assert db_name == "mednlp_test"
        finally:
            await conn.close()


class TestRedisConnectivity:
    async def test_can_ping(self) -> None:
        client = aioredis.from_url(REDIS_URL)
        try:
            pong = await client.ping()
            assert pong is True
        finally:
            await client.aclose()

    async def test_set_and_get(self) -> None:
        client = aioredis.from_url(REDIS_URL)
        try:
            await client.set("mednlp_test_key", "ok", ex=10)
            value = await client.get("mednlp_test_key")
            assert value == b"ok"
        finally:
            await client.delete("mednlp_test_key")
            await client.aclose()

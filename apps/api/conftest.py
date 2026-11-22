import asyncio
import importlib
import os
from collections.abc import AsyncIterator, Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from app.core.config import get_settings
from app.core.database import (
    Base,
    create_db_engine,
    create_redis_client,
    create_session_factory,
)
from app.main import app

# Tests use their own database and Redis DB index so dev data is never touched.
DEFAULT_TEST_REDIS_URL = "redis://localhost:6379/15"


def _run_in_thread(coro: Coroutine[Any, Any, None]) -> None:
    """Run a coroutine on its own loop, leaving pytest-asyncio's loop untouched."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(asyncio.run, coro).result()


def _test_database_url() -> str:
    explicit = os.environ.get("TEST_DATABASE_URL")
    if explicit:
        return explicit
    base = make_url(get_settings().database_url)
    renamed = base.set(database=f"{base.database}_test")
    return renamed.render_as_string(hide_password=False)


async def _ensure_database(url: str) -> None:
    target = make_url(url)
    admin = create_async_engine(
        target.set(database="postgres"), isolation_level="AUTOCOMMIT"
    )
    try:
        async with admin.connect() as conn:
            exists = await conn.scalar(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": target.database},
            )
            if not exists:
                quoted = (target.database or "").replace('"', '""')
                await conn.execute(text(f'CREATE DATABASE "{quoted}"'))
    finally:
        await admin.dispose()


def _import_models() -> None:
    """Register ORM models on Base.metadata once Task 2.1 adds them."""
    try:
        importlib.import_module("app.models.domain")
    except ModuleNotFoundError as exc:
        if exc.name not in {"app.models", "app.models.domain"}:
            raise


@pytest.fixture(scope="session")
def test_database_url() -> str:
    url = _test_database_url()
    name = make_url(url).database or ""
    if not name.endswith("_test"):
        pytest.fail(f"Refusing to use database '{name}': name must end with '_test'")
    try:
        _run_in_thread(_ensure_database(url))
    except Exception as exc:
        pytest.skip(f"PostgreSQL not reachable: {type(exc).__name__}")
    return url


@pytest_asyncio.fixture
async def db_engine(test_database_url: str) -> AsyncIterator[AsyncEngine]:
    """Engine on a fresh schema for each test; tables are dropped afterwards."""
    _import_models()
    engine = create_db_engine(test_database_url)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield engine
    finally:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    factory = create_session_factory(db_engine)
    async with factory() as session:
        yield session


@pytest_asyncio.fixture
async def redis_client() -> AsyncIterator[Redis]:
    """Redis client on the test DB index, flushed before and after each test."""
    url = os.environ.get("TEST_REDIS_URL", DEFAULT_TEST_REDIS_URL)
    if url == get_settings().redis_url:
        pytest.fail("TEST_REDIS_URL must differ from REDIS_URL (FLUSHDB is used)")
    client = create_redis_client(url)
    try:
        await client.ping()
    except Exception as exc:
        await client.aclose()
        pytest.skip(f"Redis not reachable: {type(exc).__name__}")
    await client.flushdb()
    try:
        yield client
    finally:
        await client.flushdb()
        await client.aclose()


@pytest_asyncio.fixture
async def client(
    db_engine: AsyncEngine, redis_client: Redis
) -> AsyncIterator[AsyncClient]:
    """HTTP client bound to the app. ASGITransport skips the lifespan, so the
    shared resources are attached to app.state here, as main.py would do."""
    app.state.engine = db_engine
    app.state.session_factory = create_session_factory(db_engine)
    app.state.redis = redis_client
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        yield http

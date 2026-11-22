from collections.abc import AsyncIterator, Awaitable
from typing import cast

from fastapi import Request
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative base for all ORM models (defined in app/models/domain.py)."""


def create_db_engine(database_url: str) -> AsyncEngine:
    return create_async_engine(database_url, pool_pre_ping=True)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


def create_redis_client(redis_url: str) -> Redis:
    return cast(Redis, Redis.from_url(redis_url, decode_responses=True))


async def ensure_pgvector(engine: AsyncEngine) -> None:
    """Enable pgvector; the Postgres image ships the extension but leaves it off."""
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))


async def check_postgres(engine: AsyncEngine) -> None:
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))


async def check_redis(client: Redis) -> None:
    await cast("Awaitable[bool]", client.ping())


# FastAPI dependencies: resources are created once in the app lifespan (main.py).


def get_engine(request: Request) -> AsyncEngine:
    return cast(AsyncEngine, request.app.state.engine)


def get_redis(request: Request) -> Redis:
    return cast(Redis, request.app.state.redis)


async def get_db_session(request: Request) -> AsyncIterator[AsyncSession]:
    factory = cast(
        "async_sessionmaker[AsyncSession]", request.app.state.session_factory
    )
    async with factory() as session:
        yield session

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, FastAPI, Response, status
from pydantic import BaseModel
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.config import get_settings
from app.core.database import (
    check_postgres,
    check_redis,
    create_db_engine,
    create_redis_client,
    create_session_factory,
    ensure_pgvector,
    get_engine,
    get_redis,
)

logger = logging.getLogger("app")

ProbeStatus = Literal["ok", "error"]


class HealthResponse(BaseModel):
    """Mirrors shared-contracts HealthResponse (moves to contracts.py in Task 2.1)."""

    status: Literal["ok", "degraded"]
    postgres: ProbeStatus
    redis: ProbeStatus


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    engine = create_db_engine(settings.database_url)
    redis_client = create_redis_client(settings.redis_url)
    app.state.engine = engine
    app.state.session_factory = create_session_factory(engine)
    app.state.redis = redis_client

    # The server must start even if Postgres is down; /health then reports it.
    try:
        await ensure_pgvector(engine)
    except Exception as exc:
        logger.warning("Could not enable pgvector at startup: %s", type(exc).__name__)

    try:
        yield
    finally:
        await redis_client.aclose()
        await engine.dispose()


async def _probe(
    check: Awaitable[None], name: str, timeout_seconds: float
) -> ProbeStatus:
    try:
        async with asyncio.timeout(timeout_seconds):
            await check
    except Exception as exc:
        logger.warning("Health probe failed for %s: %s", name, type(exc).__name__)
        return "error"
    return "ok"


router = APIRouter(prefix="/api/v1")


@router.get("/health", response_model=HealthResponse)
async def health(
    response: Response,
    engine: Annotated[AsyncEngine, Depends(get_engine)],
    redis_client: Annotated[Redis, Depends(get_redis)],
) -> HealthResponse:
    timeout = get_settings().health_probe_timeout_seconds
    postgres, redis = await asyncio.gather(
        _probe(check_postgres(engine), "postgres", timeout),
        _probe(check_redis(redis_client), "redis", timeout),
    )
    healthy = postgres == "ok" and redis == "ok"
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="ok" if healthy else "degraded", postgres=postgres, redis=redis
    )


app = FastAPI(
    title="Virtual AI Shopkeeper API",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/api/v1/docs",
    openapi_url="/api/v1/openapi.json",
)
app.include_router(router)

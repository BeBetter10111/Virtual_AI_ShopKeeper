import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, FastAPI, Response, status
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine

from app.api.rest_routes import router as rest_router
from app.api.ws_handler import router as ws_router
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
from app.schemas.contracts import HealthResponse
from app.services.session_manager import create_default_session_manager

logger = logging.getLogger("app")

ProbeStatus = Literal["ok", "error"]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    engine = create_db_engine(settings.database_url)
    redis_client = create_redis_client(settings.redis_url)
    app.state.engine = engine
    app.state.session_factory = create_session_factory(engine)
    app.state.redis = redis_client

    manager = create_default_session_manager(redis_client, app.state.session_factory)
    app.state.session_manager = manager
    await manager.warm_up()

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


async def _probe(check: Awaitable[None], name: str, timeout: float) -> ProbeStatus:
    try:
        await asyncio.wait_for(check, timeout=timeout)
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
app.include_router(rest_router)
app.include_router(ws_router)

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# apps/api/app/core/config.py -> repository root
_REPO_ROOT = Path(__file__).resolve().parents[4]


class Settings(BaseSettings):
    """Application settings. Defaults match docker-compose.yml for local development."""

    model_config = SettingsConfigDict(
        env_file=_REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = (
        "postgresql+asyncpg://shopkeeper:shopkeeper_dev@localhost:5432/shopkeeper"
    )
    redis_url: str = "redis://localhost:6379/0"

    jwt_secret_key: str = "change-me"
    access_token_ttl_seconds: int = 3600

    health_probe_timeout_seconds: float = 2.0


@lru_cache
def get_settings() -> Settings:
    return Settings()

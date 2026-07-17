from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application configuration sourced from the environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(..., description="API key for the OpenAI-compatible backend.")
    openai_model: str = Field(default="gpt-4o-mini")
    embedding_model: str = Field(default="text-embedding-3-small")

    qdrant_url: str | None = Field(default=None)
    qdrant_api_key: str | None = Field(default=None)
    qdrant_collection: str = Field(default="documents")

    agent_max_iterations: int = Field(default=5, ge=1, le=25)
    request_timeout_seconds: float = Field(default=30.0, gt=0.0)


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance built from the environment."""
    return Settings()

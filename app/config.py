from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application configuration sourced from the environment / `.env`.

    Loads from environment variables and `.env` file automatically.
    All fields can be overridden with `export VAR=value`.

    Examples:
        ```bash
        export OPENAI_API_KEY=sk-...
        export AGENT_MAX_ITERATIONS=10
        export LOG_LEVEL=DEBUG
        ```
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- OpenAI Configuration ---
    openai_api_key: str = Field(
        ...,
        description="API key for the OpenAI-compatible backend.",
        min_length=10,
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model for chat completions",
        json_schema_extra={"examples": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]},
    )
    openai_base_url: str | None = Field(
        default=None,
        description="Custom OpenAI base URL (for proxies or Azure)",
        json_schema_extra={"examples": ["https://api.openai.com/v1", "https://your-proxy.com/v1"]},
    )

    # --- Embedding Configuration ---
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model",
        json_schema_extra={"examples": ["text-embedding-3-small", "text-embedding-3-large"]},
    )
    embedding_vector_size: int = Field(
        default=1536,
        ge=1,
        description="Vector dimension for embedding model",
    )

    # --- Qdrant Configuration ---
    qdrant_url: str | None = Field(
        default=None,
        description="Qdrant server URL (uses environment variable QDRANT_URL if not set)",
        json_schema_extra={"examples": ["http://localhost:6333", "https://qdrant.cloud"]},
    )
    qdrant_api_key: str | None = Field(
        default=None,
        description="Qdrant API key (uses environment variable QDRANT_API_KEY if not set)",
    )
    qdrant_collection: str = Field(
        default="documents",
        min_length=1,
        description="Qdrant collection name",
    )

    # --- Agent Configuration ---
    agent_max_iterations: int = Field(
        default=5,
        ge=1,
        le=25,
        description="Maximum tool-calling loop iterations per request",
    )
    agent_timeout_seconds: float = Field(
        default=30.0,
        gt=0.0,
        le=120.0,
        description="Timeout for agent execution in seconds",
    )

    # --- HTTP Configuration ---
    request_timeout_seconds: float = Field(
        default=30.0,
        gt=0.0,
        le=120.0,
        description="HTTP request timeout for OpenAI calls",
    )

    # --- CORS Configuration ---
    cors_origins: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins (use specific domains in production)",
    )
    cors_allow_credentials: bool = Field(
        default=True,
        description="Allow credentials in CORS requests",
    )
    cors_allow_methods: list[str] = Field(
        default=["*"],
        description="Allowed HTTP methods for CORS",
    )
    cors_allow_headers: list[str] = Field(
        default=["*"],
        description="Allowed HTTP headers for CORS",
    )

    # --- Logging Configuration ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string",
    )
    log_json: bool = Field(
        default=False,
        description="Enable JSON logging format (for production)",
    )

    # --- Retry Configuration ---
    retry_max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts for API calls",
    )
    retry_min_wait: float = Field(
        default=1.0,
        gt=0.0,
        description="Minimum retry wait time in seconds",
    )
    retry_max_wait: float = Field(
        default=10.0,
        gt=0.0,
        description="Maximum retry wait time in seconds",
    )

    # --- Rate Limiting Configuration ---
    rate_limit_requests_per_minute: int = Field(
        default=100,
        ge=1,
        description="Rate limiting: requests per minute per client",
    )

    # --- Validation ---
    @field_validator("openai_api_key", mode="after")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Ensure API key is present and has minimum length."""
        if not v or len(v) < 10:
            raise ValueError(
                "OPENAI_API_KEY must be set and at least 10 characters long. "
                "Did you create a .env file or set the environment variable?"
            )
        return v

    @field_validator("qdrant_url")
    @classmethod
    def validate_qdrant_url(cls, v: str | None) -> str | None:
        """Ensure Qdrant URL has correct format if provided."""
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("QDRANT_URL must start with http:// or https://")
        return v

    @field_validator("cors_origins")
    @classmethod
    def validate_cors_origins(cls, v: list[str]) -> list[str]:
        """Validate CORS origins metadata bounds."""
        if v == [""]:
            # Fallback safe assignment
            return ["*"]
        return v

    def __repr__(self) -> str:
        """Human-readable representation for debugging (hides sensitive data)."""
        return (
            f"<Settings openai_model={self.openai_model} "
            f"agent_max_iterations={self.agent_max_iterations} "
            f"log_level={self.log_level}>"
        )


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance built from the environment.

    Uses lru_cache to ensure settings are loaded only once.
    """
    return Settings()


# Singleton instance alias provider
settings = get_settings()

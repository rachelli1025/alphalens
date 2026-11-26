from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-wide settings, read from environment variables (and .env if present)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://alphalens:alphalens@localhost:5432/alphalens"
    redis_url: str = "redis://localhost:6379/0"

    anthropic_api_key: str = ""
    voyage_api_key: str = ""
    sec_user_agent: str = ""

    # Models (see docs/DESIGN.md §4.3)
    agent_model: str = "claude-opus-5-5"
    extraction_model: str = "claude-haiku-5-5"
    embedding_model: str = "voyage-finance-2"
    embedding_dim: int = 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[4]  # valid in a source checkout
REPO_ROOT_ENV = REPO_ROOT / ".env"


class Settings(BaseSettings):
    """Process-wide settings, read from environment variables (and .env if present)."""

    model_config = SettingsConfigDict(env_file=(REPO_ROOT_ENV, ".env"), extra="ignore")

    database_url: str = "postgresql+psycopg://alphalens:alphalens@localhost:5432/alphalens"
    redis_url: str = "redis://localhost:6379/0"

    anthropic_api_key: str = ""
    voyage_api_key: str = ""
    sec_user_agent: str = ""

    # API access control (see apps/api/app/routers/ask.py). Empty = no key required,
    # which is only acceptable locally: deployed instances must set ACCESS_KEY.
    access_key: str = ""
    # Comma-separated browser origins allowed to call the API.
    cors_origins: str = "http://localhost:3000"

    # Models (see docs/DESIGN.md §4.3)
    agent_model: str = "claude-opus-5-5"  # W3 research agent
    # /api/ask and `alphalens ask`. Haiku until the answer eval exists (DESIGN.md D7).
    answer_model: str = "claude-haiku-5-5"
    answer_effort: str = "medium"  # low | medium | high | xhigh | max
    # Query understanding: which companies a question is about (DESIGN.md §8, first step).
    query_model: str = "claude-haiku-5-5"
    # Second retrieval stage: rerank hybrid-search candidates (DESIGN.md §8).
    rerank_model: str = "rerank-2.5"
    # Retrieval eval v2 (35 cases): hybrid Recall@8 0.89 / MRR 0.66 -> rerank 1.00 / 0.90.
    rerank_enabled: bool = True
    extraction_model: str = "claude-haiku-5-5"
    embedding_model: str = "voyage-finance-2"
    embedding_dim: int = 1024

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip().rstrip("/") for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

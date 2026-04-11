from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    model: str = "gemma4:e4"
    timeout: int = 60
    temperature: float = 0.1


class EmbeddingsConfig(BaseModel):
    model: str = "all-MiniLM-L6-v2"


class DedupConfig(BaseModel):
    minhash_threshold: float = 0.7
    minhash_num_perm: int = 128
    cosine_definite_threshold: float = 0.88
    cosine_borderline_threshold: float = 0.80
    window_hours: int = 72


class ScrapeConfig(BaseModel):
    messages_per_channel: int = 100
    delay_between_channels_seconds: float = 1.0


class DigestConfig(BaseModel):
    default_top_k: int = 5
    daily_relevance_floor: float = 0.2
    weekly_relevance_floor: float = 0.4


class DatabaseConfig(BaseModel):
    url: str = "sqlite+aiosqlite:///data/digest.db"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Telegram secrets (from .env)
    telegram_api_id: int = 0
    telegram_api_hash: str = ""
    telegram_bot_token: str = ""
    webhook_url: str = ""

    # Database
    database_url: str = "sqlite+aiosqlite:///data/digest.db"

    # Nested configs (from config.yaml)
    ollama: OllamaConfig = OllamaConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig()
    dedup: DedupConfig = DedupConfig()
    scrape: ScrapeConfig = ScrapeConfig()
    digest: DigestConfig = DigestConfig()
    database: DatabaseConfig = DatabaseConfig()

    @classmethod
    def from_yaml(cls, yaml_path: Path | None = None, **overrides) -> Settings:
        """Load settings from .env (secrets) + config.yaml (app config)."""
        yaml_path = yaml_path or _PROJECT_ROOT / "config.yaml"
        yaml_data: dict = {}
        if yaml_path.exists():
            with open(yaml_path) as f:
                yaml_data = yaml.safe_load(f) or {}

        # Merge YAML data with any overrides
        merged = {**yaml_data, **overrides}
        return cls(**merged)


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings.from_yaml()

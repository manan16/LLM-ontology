from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


load_dotenv()


class Settings(BaseSettings):
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="qwen2.5:7b-instruct", alias="OLLAMA_MODEL")
    ollama_fallback_models: str = Field(default="gemma3:12b,qwen3:14b", alias="OLLAMA_FALLBACK_MODELS")
    ollama_temperature: float = Field(default=0.0, alias="OLLAMA_TEMPERATURE")
    ollama_timeout_seconds: int = Field(default=300, alias="OLLAMA_TIMEOUT_SECONDS")
    ollama_keep_alive: str = Field(default="15m", alias="OLLAMA_KEEP_ALIVE")

    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_username: str = Field(default="neo4j", alias="NEO4J_USERNAME")
    neo4j_password: str = Field(default="neo4j", alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", alias="NEO4J_DATABASE")

    chunk_max_chars: int = Field(default=1800, alias="CHUNK_MAX_CHARS")
    chunk_overlap_chars: int = Field(default=150, alias="CHUNK_OVERLAP_CHARS")
    log_level: str = Field(default="DEBUG", alias="LOG_LEVEL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        populate_by_name=True,
        extra="ignore",
    )

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[1]


def get_settings() -> Settings:
    return Settings()

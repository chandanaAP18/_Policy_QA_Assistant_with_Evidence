"""
Application-wide configuration.

All configuration is sourced from environment variables (via a `.env` file in
development, or real environment variables in production). No values are
hardcoded elsewhere in the codebase — every module that needs configuration
imports `get_settings()` from this file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = two levels up from this file (src/policy_qa_assistant/config -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Strongly-typed application settings, populated from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
                 
    )

    # --- Gemini (Google AI Studio) ---------------------------------------
    gemini_api_key: str = Field(default="", description="Google AI Studio API key")
    gemini_model: str = Field(default="gemini-2.0-flash", description="Gemini model name")
    enable_llm: bool = Field(default=False, description="Enable Gemini grounded drafting when an API key is configured")

    # --- Retrieval ---------------------------------------------------------
    top_k: int = Field(default=10, ge=1, le=50, description="Number of chunks to retrieve per query")
    retrieval_score_threshold: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum similarity score to keep a retrieved chunk"
    )

    # --- Chunking ------------------------------------------------------------
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=120, ge=0)

    # --- Vector store ---------------------------------------------------------
    chroma_db_path: str = Field(default="data/vector_store")
    chroma_collection_name: str = Field(default="policy_documents")

    # --- Embeddings ------------------------------------------------------------
    embedding_model: str = Field(default="all-MiniLM-L6-v2")

    # --- Storage paths -----------------------------------------------------
    policy_docs_path: str = Field(default="data/policies")
    questions_db_path: str = Field(default="data/questions.db")

    # --- Application ---------------------------------------------------------
    app_env: str = Field(default="development")
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000, gt=0, lt=65536)
    log_level: str = Field(default="INFO")
    log_dir: str = Field(default="logs")
    policy_admin_key: str = Field(
        default="", description="Secret required to edit an ingested policy document"
    )

    # --- Confidence / evaluation -------------------------------------------
    min_confidence_for_support: float = Field(default=0.15, ge=0.0, le=1.0)

    @field_validator("log_level")
    @classmethod
    def _uppercase_log_level(cls, value: str) -> str:
        return value.upper()

    # --- Resolved absolute paths (computed, not env-driven) ------------------
    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def chroma_db_abs_path(self) -> Path:
        return self._resolve(self.chroma_db_path)

    @property
    def policy_docs_abs_path(self) -> Path:
        return self._resolve(self.policy_docs_path)

    @property
    def questions_db_abs_path(self) -> Path:
        return self._resolve(self.questions_db_path)

    @property
    def log_dir_abs_path(self) -> Path:
        return self._resolve(self.log_dir)

    def _resolve(self, relative_or_absolute: str) -> Path:
        path = Path(relative_or_absolute)
        return path if path.is_absolute() else (self.project_root / path)

    def ensure_directories(self) -> None:
        """Create all directories this application writes to, if missing."""
        for path in (
            self.chroma_db_abs_path,
            self.policy_docs_abs_path,
            self.log_dir_abs_path,
            self.questions_db_abs_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached, process-wide Settings instance.

    Using `lru_cache` means the `.env` file and environment are only read
    once per process, and every module gets the same settings object.
    """
    settings = Settings()
    settings.ensure_directories()
    return settings

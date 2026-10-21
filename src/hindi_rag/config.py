"""Typed, validated application configuration.

All settings come from environment variables (case-insensitive) and an
optional ``.env`` file. Invalid combinations fail fast at startup rather
than at first request.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["development", "staging", "production"]
LLMBackend = Literal["groq", "ollama"]


class ConfigurationError(RuntimeError):
    """Raised when the service is missing a required credential/configuration."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -- application -----------------------------------------------------
    app_name: str = "hindi-rag-whatsapp"
    app_version: str = "2.0.0"
    environment: Environment = "development"
    log_level: str = "INFO"
    log_json: bool = True

    # -- LLM -------------------------------------------------------------
    llm_backend: LLMBackend = "groq"
    groq_api_key: SecretStr | None = None
    groq_model: str = "llama-3.1-8b-instant"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:4b"
    llm_temperature: float = Field(0.2, ge=0.0, le=1.0)
    llm_timeout_s: float = Field(30.0, gt=0)
    llm_max_retries: int = Field(3, ge=0, le=10)

    # -- embeddings / vector store --------------------------------------
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    chroma_persist_dir: Path = Path("./data/chroma")
    collection_name: str = "hindi_rag_v2"

    # -- retrieval -------------------------------------------------------
    retrieval_k: int = Field(4, ge=1, le=20)
    retrieval_fetch_k: int = Field(12, ge=1, le=100)
    mmr_lambda: float = Field(0.5, ge=0.0, le=1.0)
    chunk_size: int = Field(512, ge=64, le=4096)
    chunk_overlap: int = Field(64, ge=0, le=1024)
    corpus_dir: Path = Path("./data/corpus")

    # -- WhatsApp --------------------------------------------------------
    twilio_account_sid: str = ""
    twilio_auth_token: SecretStr | None = None
    twilio_whatsapp_number: str = "whatsapp:+14155238886"
    twilio_validate_signature: bool = True

    # -- safety limits ---------------------------------------------------
    max_question_chars: int = Field(1000, ge=1, le=8000)
    rate_limit_requests: int = Field(30, ge=1)
    rate_limit_window_s: int = Field(60, ge=1)

    # -- Hindi generation prompt ----------------------------------------
    system_prompt: str = (
        "आप एक सहायक AI हैं जो केवल हिंदी में उत्तर देते हैं। "
        "दिए गए संदर्भ के आधार पर सटीक और सरल उत्तर दें। "
        "यदि उत्तर संदर्भ में उपलब्ध नहीं है, तो ठीक यही लिखें: "
        "'मुझे इस बारे में जानकारी नहीं है।' "
        "कभी भी संदर्भ से बाहर की जानकारी न बनाएं।"
    )

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_must_be_smaller_than_chunk(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 512)
        if v >= chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return v

    @model_validator(mode="after")
    def _validate_cross_field(self) -> Settings:
        if self.retrieval_fetch_k < self.retrieval_k:
            raise ValueError("RETRIEVAL_FETCH_K must be >= RETRIEVAL_K")
        if self.llm_backend == "groq" and self.llm_max_retries < 1:
            raise ValueError("llm_max_retries must be >= 1 for hosted LLM backends")
        return self

    # -- helpers ---------------------------------------------------------
    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    def require_llm_credentials(self) -> None:
        """Fail fast with an actionable message instead of a 500 on first use."""
        if self.llm_backend == "groq":
            key = self.groq_api_key.get_secret_value() if self.groq_api_key else ""
            if not key.strip():
                raise ConfigurationError(
                    "LLM_BACKEND=groq requires GROQ_API_KEY. "
                    "Set it in the environment or switch LLM_BACKEND=ollama."
                )

    def require_twilio_credentials(self) -> None:
        if not self.twilio_account_sid or not self.twilio_auth_token:
            raise ConfigurationError(
                "WhatsApp webhook requires TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor (FastAPI dependency override friendly)."""
    return Settings()

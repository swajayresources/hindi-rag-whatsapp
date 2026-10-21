"""Configuration validation: fail fast, actionable messages."""

from __future__ import annotations

import pytest

from hindi_rag.config import ConfigurationError, Settings


def test_defaults_are_sane() -> None:
    s = Settings(_env_file=None)
    assert s.retrieval_k >= 1
    assert s.retrieval_fetch_k >= s.retrieval_k
    assert s.chunk_overlap < s.chunk_size
    assert s.app_version


def test_overlap_must_be_smaller_than_chunk() -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, chunk_size=256, chunk_overlap=256)


def test_fetch_k_must_be_at_least_k() -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, retrieval_k=10, retrieval_fetch_k=4)


def test_groq_requires_api_key() -> None:
    s = Settings(_env_file=None, llm_backend="groq", groq_api_key=None)
    with pytest.raises(ConfigurationError):
        s.require_llm_credentials()


def test_groq_with_key_passes() -> None:
    s = Settings(_env_file=None, llm_backend="groq", groq_api_key="test-key")
    s.require_llm_credentials()  # must not raise


def test_ollama_does_not_require_key() -> None:
    s = Settings(_env_file=None, llm_backend="ollama")
    s.require_llm_credentials()  # must not raise


def test_twilio_credentials_required_for_webhook() -> None:
    s = Settings(_env_file=None)
    with pytest.raises(ConfigurationError):
        s.require_twilio_credentials()


def test_production_flag() -> None:
    assert Settings(_env_file=None, environment="production").is_production
    assert not Settings(_env_file=None, environment="development").is_production

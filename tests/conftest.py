"""Shared fixtures: fake retriever/LLM/service so unit tests stay dependency-free."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hindi_rag.config import Settings  # noqa: E402
from hindi_rag.generation import RagService  # noqa: E402
from tests.fakes import FakeLLM, FakeRetriever, make_doc  # noqa: E402


@pytest.fixture()
def settings() -> Settings:
    return Settings(
        _env_file=None,
        llm_backend="ollama",
        twilio_validate_signature=False,
        chroma_persist_dir=Path("./data/chroma-test"),
        corpus_dir=ROOT / "data" / "corpus",
        rate_limit_requests=1000,
    )


@pytest.fixture()
def fake_service(settings: Settings) -> RagService:
    retriever = FakeRetriever([make_doc("भारत की राजधानी नई दिल्ली है।")])
    return RagService(settings, retriever=retriever, llm=FakeLLM())


@pytest.fixture()
def empty_service(settings: Settings) -> RagService:
    return RagService(settings, retriever=FakeRetriever([]), llm=FakeLLM())

"""Shared fixtures: fake retriever/LLM/service so unit tests stay dependency-free."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hindi_rag.config import Settings  # noqa: E402
from hindi_rag.generation import RagService  # noqa: E402


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


def make_doc(text: str, source: str = "भारत.txt") -> SimpleNamespace:
    return SimpleNamespace(page_content=text, metadata={"source": source})


class FakeRetriever:
    def __init__(self, docs: list[SimpleNamespace]) -> None:
        self.docs = docs
        self.calls: list[str] = []

    def invoke(self, question: str) -> list[SimpleNamespace]:
        self.calls.append(question)
        return self.docs


class FakeLLM:
    def __init__(self, content: str = "नई दिल्ली भारत की राजधानी है।", fail_times: int = 0) -> None:
        self.content = content
        self.fail_times = fail_times
        self.calls = 0

    def invoke(self, messages):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("transient provider error")
        return SimpleNamespace(content=self.content)

    def stream(self, messages):  # type: ignore[no-untyped-def]
        yield SimpleNamespace(content=self.content)


@pytest.fixture()
def fake_service(settings: Settings) -> RagService:
    retriever = FakeRetriever([make_doc("भारत की राजधानी नई दिल्ली है।")])
    return RagService(settings, retriever=retriever, llm=FakeLLM())


@pytest.fixture()
def empty_service(settings: Settings) -> RagService:
    return RagService(settings, retriever=FakeRetriever([]), llm=FakeLLM())

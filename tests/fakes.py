"""Test doubles shared across suites (no network, no vector DB, no LLM)."""

from __future__ import annotations

from types import SimpleNamespace


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

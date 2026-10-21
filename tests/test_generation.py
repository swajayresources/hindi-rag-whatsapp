"""RAG pipeline behaviour with fake retriever/LLM (no network, no vector DB)."""

from __future__ import annotations

import pytest

from hindi_rag.generation import REFUSAL, LLMError, RagService, invoke_with_retries
from tests.fakes import FakeLLM, FakeRetriever, make_doc


def test_no_context_refuses_without_calling_llm(settings) -> None:
    llm = FakeLLM()
    service = RagService(settings, retriever=FakeRetriever([]), llm=llm)

    answer = service.answer("भारत की राजधानी क्या है?")

    assert answer.text == REFUSAL
    assert answer.grounded is False
    assert answer.sources == []
    assert llm.calls == 0, "must not spend an LLM call when nothing was retrieved"


def test_grounded_answer_includes_sources(settings) -> None:
    docs = [make_doc("भारत की राजधानी नई दिल्ली है।", "भारत.txt")]
    service = RagService(settings, retriever=FakeRetriever(docs), llm=FakeLLM())

    answer = service.answer("भारत की राजधानी क्या है?", include_sources=True)

    assert answer.grounded is True
    assert "नई दिल्ली" in answer.text
    assert answer.sources and answer.sources[0].source == "भारत.txt"
    assert answer.latency_ms >= 0


def test_sources_can_be_omitted(settings) -> None:
    docs = [make_doc("कुछ संदर्भ", "दिल्ली.txt")]
    service = RagService(settings, retriever=FakeRetriever(docs), llm=FakeLLM("उत्तर"))
    answer = service.answer("प्रश्न", include_sources=False)
    assert answer.sources == []


def test_model_refusal_is_marked_ungrounded(settings) -> None:
    docs = [make_doc("असंबंधित संदर्भ")]
    service = RagService(settings, retriever=FakeRetriever(docs), llm=FakeLLM(REFUSAL))
    answer = service.answer("प्रश्न")
    assert answer.grounded is False


def test_empty_llm_content_falls_back_to_refusal(settings) -> None:
    docs = [make_doc("संदर्भ")]
    service = RagService(settings, retriever=FakeRetriever(docs), llm=FakeLLM(""))
    answer = service.answer("प्रश्न")
    assert answer.text == REFUSAL
    assert answer.grounded is False


def test_retries_then_succeeds(settings, monkeypatch) -> None:
    monkeypatch.setattr("hindi_rag.generation.time.sleep", lambda _s: None)
    docs = [make_doc("भारत की राजधानी नई दिल्ली है।")]
    llm = FakeLLM(fail_times=2)
    service = RagService(settings, retriever=FakeRetriever(docs), llm=llm)

    answer = service.answer("राजधानी?")

    assert llm.calls == 3
    assert answer.grounded is True


def test_retries_exhausted_raises(settings, monkeypatch) -> None:
    monkeypatch.setattr("hindi_rag.generation.time.sleep", lambda _s: None)
    docs = [make_doc("संदर्भ")]
    service = RagService(
        settings, retriever=FakeRetriever(docs), llm=FakeLLM(fail_times=99)
    )
    with pytest.raises(LLMError):
        service.answer("प्रश्न")


def test_invoke_with_retries_zero_retries_raises_immediately() -> None:
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise ValueError("nope")

    with pytest.raises(LLMError):
        invoke_with_retries(boom, retries=0)
    assert calls["n"] == 1


def test_context_includes_source_markers(settings) -> None:
    docs = [make_doc("सामग्री", "दिल्ली.txt")]
    service = RagService(settings, retriever=FakeRetriever(docs), llm=FakeLLM())
    context = service.build_context(docs)
    assert "स्रोत: दिल्ली.txt" in context
    assert "सामग्री" in context

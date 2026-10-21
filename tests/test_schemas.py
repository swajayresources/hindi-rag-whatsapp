"""API schema validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hindi_rag.schemas import ChatRequest, ChatResponse, Source


def test_question_is_stripped() -> None:
    assert ChatRequest(question="  प्रश्न  ").question == "प्रश्न"


def test_empty_question_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(question="   ")


def test_too_long_question_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(question="क" * 3000)


def test_include_sources_default_true() -> None:
    assert ChatRequest(question="प्रश्न").include_sources is True


def test_chat_response_shape() -> None:
    response = ChatResponse(
        question="प्रश्न",
        answer="उत्तर",
        sources=[Source(source="भारत.txt", snippet="...")],
        grounded=True,
        latency_ms=12.5,
        request_id="abc",
    )
    assert response.sources[0].source == "भारत.txt"
    assert response.sources[0].score is None

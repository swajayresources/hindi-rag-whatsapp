"""WhatsApp helper behaviour."""

from __future__ import annotations

import pytest

from hindi_rag.whatsapp import (
    EMPTY_REPLY,
    FALLBACK_REPLY,
    TwilioSignatureError,
    build_reply,
    truncate_for_whatsapp,
    twiml_response,
    validate_twilio_signature,
)


class _ExplodingService:
    def answer(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("provider down")


class _EchoService:
    def answer(self, question, include_sources=True):  # type: ignore[no-untyped-def]
        class A:
            text = "उत्तर: " + question
            grounded = True

        return A()


def test_truncate_short_text_unchanged() -> None:
    assert truncate_for_whatsapp("छोटा") == "छोटा"


def test_truncate_long_text_adds_ellipsis() -> None:
    out = truncate_for_whatsapp("क" * 2000, limit=1500)
    assert len(out) <= 1500
    assert out.endswith("...")


def test_build_reply_empty_body() -> None:
    assert build_reply(_EchoService(), "   ") == EMPTY_REPLY


def test_build_reply_uses_service() -> None:
    assert "उत्तर" in build_reply(_EchoService(), "प्रश्न")


def test_build_reply_falls_back_on_error() -> None:
    assert build_reply(_ExplodingService(), "प्रश्न") == FALLBACK_REPLY


def test_missing_signature_raises() -> None:
    with pytest.raises(TwilioSignatureError):
        validate_twilio_signature(auth_token="t", url="https://x/y", params={}, signature=None)


def test_invalid_signature_raises() -> None:
    with pytest.raises(TwilioSignatureError):
        validate_twilio_signature(
            auth_token="t", url="https://x/y", params={"Body": "hi"}, signature="bogus"
        )


def test_twiml_response_contains_message() -> None:
    xml = twiml_response("नमस्ते")
    assert "नमस्ते" in xml
    assert "Response" in xml

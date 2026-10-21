"""WhatsApp delivery via Twilio, with signature verification.

Security note: without signature validation the webhook accepts forged
requests from anyone who learns the URL. Verification is on by default and
only disabled explicitly for local testing.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

WHATSAPP_MAX_CHARS = 1500
FALLBACK_REPLY = "माफ करें, अभी तकनीकी समस्या है। कृपया थोड़ी देर बाद पुनः प्रयास करें।"
EMPTY_REPLY = "कृपया अपना प्रश्न लिखें।"


class TwilioSignatureError(PermissionError):
    """Raised when a webhook request fails Twilio signature validation."""


def validate_twilio_signature(
    *,
    auth_token: str,
    url: str,
    params: Mapping[str, Any],
    signature: str | None,
) -> None:
    """Validate ``X-Twilio-Signature`` for an incoming webhook request."""
    if not signature:
        raise TwilioSignatureError("missing X-Twilio-Signature header")

    from twilio.request_validator import RequestValidator

    validator = RequestValidator(auth_token)
    flat_params = {str(k): str(v) for k, v in params.items()}
    if not validator.validate(url, flat_params, signature):
        raise TwilioSignatureError("invalid Twilio signature")


def truncate_for_whatsapp(text: str, limit: int = WHATSAPP_MAX_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def build_reply(service: Any, body: str) -> str:
    """Produce the WhatsApp reply text for an inbound message."""
    question = (body or "").strip()
    if not question:
        return EMPTY_REPLY
    try:
        answer = service.answer(question, include_sources=False)
        return truncate_for_whatsapp(answer.text)
    except Exception as exc:
        logger.error("whatsapp answer failed", extra={"error": str(exc)}, exc_info=True)
        return FALLBACK_REPLY


def twiml_response(reply: str) -> str:
    from twilio.twiml.messaging_response import MessagingResponse

    response = MessagingResponse()
    response.message(reply)
    return str(response)

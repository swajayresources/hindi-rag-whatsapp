"""
WhatsApp webhook handler via Twilio.
Adapted from sahilpmehra/WhatsAppRAG + worldbank/WhatsApp-RAG-Example.

Setup:
  1. Get free Twilio account → WhatsApp Sandbox
  2. Set webhook URL to: https://<your-ngrok>.ngrok.io/whatsapp/webhook
  3. Fill TWILIO_* vars in .env
"""
import logging
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

from .config import get_settings
from .generation import answer

cfg = get_settings()
logger = logging.getLogger(__name__)

_client = None


def get_twilio_client() -> Client:
    global _client
    if _client is None:
        _client = Client(cfg.twilio_account_sid, cfg.twilio_auth_token)
    return _client


def send_message(to: str, body: str) -> None:
    get_twilio_client().messages.create(
        from_=cfg.twilio_whatsapp_number,
        to=to,
        body=body,
    )


def handle_incoming(from_number: str, body: str) -> str:
    """Process an incoming WhatsApp message and return reply text."""
    logger.info("Incoming from %s: %s", from_number, body[:80])

    if not body.strip():
        return "कृपया अपना प्रश्न लिखें।"

    try:
        reply = answer(body.strip())
    except Exception as exc:
        logger.error("RAG error: %s", exc)
        reply = "माफ करें, अभी तकनीकी समस्या है। कृपया थोड़ी देर बाद पुनः प्रयास करें।"

    # Twilio has a 1600-char limit per message
    if len(reply) > 1500:
        reply = reply[:1497] + "..."

    return reply


def twiml_response(reply: str) -> str:
    resp = MessagingResponse()
    resp.message(reply)
    return str(resp)

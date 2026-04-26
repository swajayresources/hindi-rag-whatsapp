"""
FastAPI app — exposes:
  POST /whatsapp/webhook   Twilio webhook
  POST /chat              Direct REST API (for testing without WhatsApp)
  GET  /health
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from .whatsapp import handle_incoming, twiml_response
from .generation import answer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Hindi RAG WhatsApp Bot (Ollama backend)...")
    logger.info("Ready.")
    yield


app = FastAPI(title="Hindi RAG WhatsApp Bot", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/whatsapp/webhook", response_class=PlainTextResponse)
async def whatsapp_webhook(
    From: str = Form(...),
    Body: str = Form(...),
):
    reply = handle_incoming(from_number=From, body=Body)
    return twiml_response(reply)


class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")
    reply = answer(req.question.strip())
    return {"question": req.question, "answer": reply}

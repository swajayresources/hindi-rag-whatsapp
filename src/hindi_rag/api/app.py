"""HTTP API: chat, health probes and the Twilio WhatsApp webhook."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import Depends, FastAPI, Form, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse

from ..config import ConfigurationError, Settings, get_settings
from ..generation import LLMError, RagService, get_rag_service
from ..ingestion import ingest_corpus
from ..logging_config import configure_logging, request_id_var
from ..retrieval import get_vector_store
from ..schemas import ChatRequest, ChatResponse, HealthResponse
from ..whatsapp import (
    TwilioSignatureError,
    build_reply,
    twiml_response,
    validate_twilio_signature,
)
from .middleware import RateLimitMiddleware, RequestContextMiddleware

logger = logging.getLogger(__name__)


def _bootstrap(settings: Settings) -> None:
    """Log configuration problems early but keep the service bootable."""
    try:
        settings.require_llm_credentials()
    except ConfigurationError as exc:
        if settings.is_production:
            raise
        logger.warning("llm credentials missing (non-production)", extra={"error": str(exc)})


def _ensure_index(settings: Settings, service: RagService) -> None:
    store = get_vector_store(settings)
    if store.count() == 0:
        logger.info(
            "vector store empty - ingesting corpus",
            extra={"corpus_dir": str(settings.corpus_dir)},
        )
        report = ingest_corpus(
            settings.corpus_dir,
            store,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        logger.info("auto-ingest finished", extra={"chunks": report.chunks})
    else:
        logger.info("vector store ready", extra={"chunks": store.count()})


def create_app(
    settings: Settings | None = None,
    service: RagService | None = None,
    *,
    bootstrap_index: bool = True,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _bootstrap(settings)
        rag = service or get_rag_service(settings)
        app.state.settings = settings
        app.state.rag = rag
        if bootstrap_index:
            _ensure_index(settings, rag)
        logger.info(
            "service started",
            extra={"version": settings.app_version, "environment": settings.environment},
        )
        yield
        logger.info("service stopped")

    app = FastAPI(
        title="Hindi RAG API",
        version=settings.app_version,
        description=(
            "Retrieval-augmented question answering in Hindi with grounded citations, "
            "MMR retrieval and WhatsApp delivery."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        RateLimitMiddleware,
        requests=settings.rate_limit_requests,
        window_s=settings.rate_limit_window_s,
    )
    app.add_middleware(RequestContextMiddleware)

    # -- dependencies -----------------------------------------------------
    def get_service(request: Request) -> RagService:
        return request.app.state.rag

    def get_cfg(request: Request) -> Settings:
        return request.app.state.settings

    # -- error handling ---------------------------------------------------
    def _error(status: int, code: str, detail: str) -> Response:
        return JSONResponse(
            status_code=status,
            content={"error": code, "detail": detail, "request_id": request_id_var.get()},
        )

    @app.exception_handler(LLMError)
    async def llm_error_handler(request: Request, exc: LLMError) -> Response:
        logger.error("llm failure", extra={"error": str(exc)})
        return _error(503, "llm_unavailable", "The language model is temporarily unavailable.")

    @app.exception_handler(TwilioSignatureError)
    async def twilio_error_handler(request: Request, exc: TwilioSignatureError) -> Response:
        logger.warning("rejected webhook", extra={"error": str(exc)})
        return _error(403, "invalid_signature", "Webhook signature validation failed.")

    @app.exception_handler(ConfigurationError)
    async def configuration_error_handler(request: Request, exc: ConfigurationError) -> Response:
        logger.error("misconfiguration", extra={"error": str(exc)})
        return _error(503, "misconfigured", str(exc))

    # -- routes -----------------------------------------------------------
    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health(cfg: Settings = Depends(get_cfg)) -> HealthResponse:
        checks: dict[str, bool] = {}
        chunks = 0
        try:
            chunks = get_vector_store(cfg).count()
            checks["vector_store"] = chunks > 0
        except Exception as exc:
            logger.error("vector store health check failed", extra={"error": str(exc)})
            checks["vector_store"] = False

        try:
            cfg.require_llm_credentials()
            checks["llm_credentials"] = True
        except ConfigurationError:
            checks["llm_credentials"] = cfg.llm_backend == "ollama"

        status: Literal["ok", "degraded"] = "ok" if all(checks.values()) else "degraded"
        return HealthResponse(
            status=status,
            version=cfg.app_version,
            environment=cfg.environment,
            llm_backend=cfg.llm_backend,
            vector_store_chunks=chunks,
            checks=checks,
        )

    @app.get("/ready", tags=["ops"])
    def ready(cfg: Settings = Depends(get_cfg)) -> dict[str, Any]:
        if get_vector_store(cfg).count() == 0:
            raise HTTPException(status_code=503, detail="index not ready")
        return {"status": "ready"}

    @app.post("/v1/chat", response_model=ChatResponse, tags=["chat"])
    def chat(req: ChatRequest, rag: RagService = Depends(get_service)) -> ChatResponse:
        answer = rag.answer(req.question, include_sources=req.include_sources)
        return ChatResponse(
            question=req.question,
            answer=answer.text,
            sources=answer.sources,
            grounded=answer.grounded,
            latency_ms=round(answer.latency_ms, 1),
            request_id=request_id_var.get(),
        )

    # Backwards-compatible alias (older clients/notebooks used /chat).
    @app.post("/chat", response_model=ChatResponse, include_in_schema=False)
    def chat_legacy(req: ChatRequest, rag: RagService = Depends(get_service)) -> ChatResponse:
        return chat(req, rag)

    @app.post("/whatsapp/webhook", response_class=PlainTextResponse, tags=["whatsapp"])
    async def whatsapp_webhook(
        request: Request,
        From: str = Form(...),
        Body: str = Form(...),
        x_twilio_signature: str | None = Header(default=None, alias="X-Twilio-Signature"),
        cfg: Settings = Depends(get_cfg),
        rag: RagService = Depends(get_service),
    ) -> PlainTextResponse:
        if cfg.twilio_validate_signature:
            token = cfg.twilio_auth_token.get_secret_value() if cfg.twilio_auth_token else ""
            if not token:
                raise ConfigurationError(
                    "TWILIO_AUTH_TOKEN required when signature validation is enabled"
                )
            form = await request.form()
            base_url = str(request.url)
            if cfg.is_production:
                # Behind a proxy the public URL differs from the internal one.
                forwarded = request.headers.get("X-Forwarded-Proto")
                host = request.headers.get("X-Forwarded-Host") or request.headers.get("Host")
                if forwarded and host:
                    base_url = f"{forwarded}://{host}{request.url.path}"
            validate_twilio_signature(
                auth_token=token,
                url=base_url,
                params={k: str(v) for k, v in form.items()},
                signature=x_twilio_signature,
            )

        reply = build_reply(rag, Body)
        return PlainTextResponse(twiml_response(reply), media_type="application/xml")

    return app

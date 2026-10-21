"""Cross-cutting middleware: request correlation, timing and rate limiting."""

from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from threading import Lock
from typing import ClassVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from ..logging_config import request_id_var

logger = logging.getLogger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a request id to every request, log latency, echo headers back."""

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
        logger.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": round(elapsed_ms, 1),
            },
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window per-client limiter.

    Deliberately in-process: one instance is enough for a single-container
    deployment, and it keeps the service dependency-free. Swap for Redis if
    you scale horizontally.
    """

    EXEMPT_PATHS: ClassVar[set[str]] = {"/health", "/ready", "/docs", "/openapi.json", "/redoc"}

    def __init__(self, app, requests: int = 30, window_s: int = 60) -> None:  # type: ignore[no-untyped-def]
        super().__init__(app)
        self.requests = requests
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = {}
        self._lock = Lock()

    def _client_key(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _allow(self, key: str) -> tuple[bool, float]:
        now = time.monotonic()
        cutoff = now - self.window_s
        with self._lock:
            bucket = self._hits.setdefault(key, deque())
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self.requests:
                retry_after = max(0.0, self.window_s - (now - bucket[0]))
                return False, retry_after
            bucket.append(now)
            return True, 0.0

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        allowed, retry_after = self._allow(self._client_key(request))
        if not allowed:
            logger.warning(
                "rate limited",
                extra={"client": self._client_key(request), "path": request.url.path},
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limited",
                    "detail": "Too many requests. Please retry shortly.",
                    "request_id": request_id_var.get(),
                },
                headers={"Retry-After": str(int(retry_after) + 1)},
            )
        return await call_next(request)

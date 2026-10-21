"""RAG answering service.

Explicit, auditable pipeline (retrieve → build grounded prompt → invoke LLM)
with retries, timeouts and refusal detection. LangChain objects are used for
the retriever/prompt/LLM, but orchestration stays in plain Python so failure
modes are visible and testable.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .schemas import Source

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_core.documents import Document

logger = logging.getLogger(__name__)

REFUSAL = "मुझे इस बारे में जानकारी नहीं है।"
SNIPPET_CHARS = 320


@dataclass
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)
    grounded: bool = True
    latency_ms: float = 0.0


class LLMError(RuntimeError):
    """Raised when the LLM cannot produce a response after retries."""


class FatalLLMError(LLMError):
    """Configuration/installation problem: retrying cannot help."""


def _backoff(attempt: int, base: float = 0.5, cap: float = 8.0) -> float:
    return min(cap, base * (2 ** attempt)) * (0.5 + random.random() / 2)


def invoke_with_retries(fn: Callable[[], Any], retries: int, operation: str = "llm.invoke") -> Any:
    """Retry transient failures with exponential backoff + jitter.

    Configuration errors are permanent: re-raise immediately instead of
    burning the retry budget (and the caller's latency budget).
    """
    from .config import ConfigurationError

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except (ConfigurationError, FatalLLMError):
            raise
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            delay = _backoff(attempt)
            logger.warning(
                "transient failure, retrying",
                extra={
                    "operation": operation,
                    "attempt": attempt + 1,
                    "delay_s": round(delay, 2),
                    "error": str(exc),
                },
            )
            time.sleep(delay)
    raise LLMError(f"{operation} failed after {retries + 1} attempts: {last_error}") from last_error


class RagService:
    def __init__(
        self,
        settings: Any,
        retriever: Any | None = None,
        llm: Any | None = None,
    ) -> None:
        self.settings = settings
        self._retriever = retriever
        self._llm = llm

    # -- lazy dependencies ------------------------------------------------
    @property
    def retriever(self) -> Any:
        if self._retriever is None:
            from .retrieval import get_retriever

            self._retriever = get_retriever(self.settings)
        return self._retriever

    @property
    def llm(self) -> Any:
        if self._llm is None:
            self._llm = build_llm(self.settings)
        return self._llm

    # -- pipeline ---------------------------------------------------------
    def retrieve(self, question: str) -> list[Document]:
        return list(self.retriever.invoke(question))

    def build_context(self, documents: Sequence[Document]) -> str:
        parts = []
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[स्रोत: {source}]\n{doc.page_content.strip()}")
        return "\n\n---\n\n".join(parts)

    def to_sources(self, documents: Sequence[Document], include_sources: bool) -> list[Source]:
        if not include_sources:
            return []
        return [
            Source(
                source=str(doc.metadata.get("source", "unknown")),
                snippet=doc.page_content.strip()[:SNIPPET_CHARS],
            )
            for doc in documents
        ]

    def answer(self, question: str, include_sources: bool = True) -> Answer:
        started = time.perf_counter()
        documents = self.retrieve(question)

        # No context → refuse without spending an LLM call.
        if not documents:
            return Answer(
                text=REFUSAL,
                sources=[],
                grounded=False,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        messages = self._build_messages(question, self.build_context(documents))

        def _call() -> Any:
            return self.llm.invoke(messages)

        response = invoke_with_retries(_call, retries=self.settings.llm_max_retries)
        text = str(getattr(response, "content", response)).strip() or REFUSAL
        grounded = not _looks_like_refusal(text)

        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "answered question",
            extra={
                "latency_ms": round(elapsed_ms, 1),
                "context_chunks": len(documents),
                "grounded": grounded,
                "answer_chars": len(text),
            },
        )
        return Answer(
            text=text,
            sources=self.to_sources(documents, include_sources),
            grounded=grounded,
            latency_ms=elapsed_ms,
        )

    def stream(self, question: str) -> Iterator[str]:
        documents = self.retrieve(question)
        if not documents:
            yield REFUSAL
            return
        messages = self._build_messages(question, self.build_context(documents))
        for chunk in self.llm.stream(messages):
            content = getattr(chunk, "content", "")
            if content:
                yield str(content)

    # -- prompt -----------------------------------------------------------
    def _build_messages(self, question: str, context: str) -> list[Any]:
        from langchain_core.messages import HumanMessage, SystemMessage

        return [
            SystemMessage(content=self.settings.system_prompt),
            HumanMessage(content=f"संदर्भ:\n{context}\n\nप्रश्न: {question}"),
        ]


def _looks_like_refusal(text: str) -> bool:
    normalised = text.strip()
    return normalised == REFUSAL or "जानकारी नहीं है" in normalised


def build_llm(settings: Any) -> Any:
    """Construct the configured chat model (Groq hosted or local Ollama)."""
    if settings.llm_backend == "groq":
        settings.require_llm_credentials()
        from langchain_groq import ChatGroq

        return ChatGroq(
            api_key=settings.groq_api_key.get_secret_value(),
            model=settings.groq_model,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_s,
            max_retries=0,  # retries handled by invoke_with_retries
        )

    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise FatalLLMError(
            "LLM_BACKEND=ollama requires the optional dependency: pip install '.[ollama]'"
        ) from exc

    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=settings.llm_temperature,
    )


def get_rag_service(settings: Any) -> RagService:
    return RagService(settings)

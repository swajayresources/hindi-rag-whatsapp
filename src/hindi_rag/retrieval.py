"""Vector store and retrieval.

A thin, typed wrapper around Chroma + FastEmbed so the rest of the codebase
never touches LangChain/Chroma internals (the previous version called the
private ``_collection`` attribute, which breaks across releases).
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def build_embeddings(model_name: str) -> Any:
    """FastEmbed (ONNX) embeddings — no torch, no Ollama required."""
    from langchain_community.embeddings import FastEmbedEmbeddings

    return FastEmbedEmbeddings(model_name=model_name)


class VectorStore:
    """Owns the persistent Chroma collection and exposes a small public API."""

    def __init__(self, embeddings: Any, persist_dir: str | Path, collection_name: str) -> None:
        from langchain_chroma import Chroma

        self._persist_dir = str(persist_dir)
        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
        self._store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=self._persist_dir,
        )
        self._collection_name = collection_name

    # -- public API ------------------------------------------------------
    def count(self) -> int:
        """Number of indexed chunks (public API, no private attributes)."""
        return len(self._store.get(include=[])["ids"])

    def add_documents(self, documents: Sequence[Document]) -> list[str]:
        if not documents:
            return []
        return self._store.add_documents(list(documents))

    def reset(self) -> None:
        """Drop and recreate the collection (used by explicit rebuilds)."""
        import chromadb

        client = chromadb.PersistentClient(path=self._persist_dir)
        with contextlib.suppress(Exception):
            client.delete_collection(self._collection_name)
        from langchain_chroma import Chroma

        self._store = Chroma(
            collection_name=self._collection_name,
            embedding_function=self._store._embedding_function,
            persist_directory=self._persist_dir,
        )

    def retriever(self, k: int, fetch_k: int, lambda_mult: float) -> Any:
        """MMR retriever — reduces near-duplicate chunks in the context."""
        return self._store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
        )

    def similarity_search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """Plain cosine search returning scores (used for retrieval metrics)."""
        return self._store.similarity_search_with_relevance_scores(query, k=k)


class VectorStoreProvider:
    """Process-wide lazy singleton (embedding models are expensive to load)."""

    _lock = threading.Lock()
    _instance: VectorStore | None = None
    _key: tuple[str, str, str] | None = None

    @classmethod
    def get(cls, settings: Any) -> VectorStore:
        key = (
            settings.embedding_model,
            str(settings.chroma_persist_dir),
            settings.collection_name,
        )
        with cls._lock:
            if cls._instance is None or cls._key != key:
                logger.info(
                    "initialising vector store",
                    extra={"persist_dir": key[1], "collection": key[2], "embedding_model": key[0]},
                )
                cls._instance = VectorStore(
                    build_embeddings(settings.embedding_model), key[1], key[2]
                )
                cls._key = key
            return cls._instance

    @classmethod
    def reset_singleton(cls) -> None:
        """Test hook."""
        with cls._lock:
            cls._instance = None
            cls._key = None


def get_vector_store(settings: Any) -> VectorStore:
    return VectorStoreProvider.get(settings)


def get_retriever(settings: Any) -> Any:
    store = get_vector_store(settings)
    return store.retriever(
        k=settings.retrieval_k,
        fetch_k=settings.retrieval_fetch_k,
        lambda_mult=settings.mmr_lambda,
    )

"""Document ingestion: load → Hindi-aware chunking → vector store.

Kept as pure functions where possible so chunking behaviour is unit-testable
without loading embeddings or a vector database.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Devanagari uses "।" (danda) as the sentence terminator; splitting on it
# before falling back to whitespace keeps Hindi sentences intact.
HINDI_SEPARATORS = ["।\n", "। ", "।", "\n\n", "\n", ". ", " ", ""]

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


@dataclass(frozen=True)
class IngestReport:
    documents: int
    chunks: int
    skipped_existing: bool


def load_documents(corpus_dir: str | Path) -> list[Document]:
    """Load every supported file under ``corpus_dir`` into LangChain Documents."""
    from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader

    root = Path(corpus_dir)
    if not root.exists():
        raise FileNotFoundError(f"Corpus directory not found: {root}")

    documents: list[Document] = []

    for glob, loader_cls, kwargs in (
        ("**/*.txt", TextLoader, {"encoding": "utf-8", "autodetect_encoding": True}),
        ("**/*.md", TextLoader, {"encoding": "utf-8", "autodetect_encoding": True}),
    ):
        loader = DirectoryLoader(
            str(root),
            glob=glob,
            loader_cls=loader_cls,
            loader_kwargs=kwargs,
            silent_errors=True,
            show_progress=False,
        )
        documents.extend(loader.load())

    for pdf_path in sorted(root.rglob("*.pdf")):
        try:
            documents.extend(PyPDFLoader(str(pdf_path)).load())
        except Exception as exc:
            logger.warning(
                "skipping unreadable pdf",
                extra={"path": str(pdf_path), "error": str(exc)},
            )

    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        doc.metadata["source"] = str(Path(source).name)
        doc.metadata.setdefault("title", Path(source).stem)

    if not documents:
        raise ValueError(f"No supported documents ({sorted(SUPPORTED_SUFFIXES)}) found in {root}")

    logger.info("loaded documents", extra={"count": len(documents), "corpus_dir": str(root)})
    return documents


def chunk_documents(
    documents: Sequence[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Document]:
    """Split documents with Hindi-aware separators, preserving metadata."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=HINDI_SEPARATORS,
        length_function=len,
    )
    chunks = splitter.split_documents(list(documents))
    chunks = [c for c in chunks if c.page_content and c.page_content.strip()]

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index

    logger.info("chunked documents", extra={"documents": len(documents), "chunks": len(chunks)})
    return chunks


def load_and_chunk(
    corpus_dir: str | Path,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Document]:
    return chunk_documents(load_documents(corpus_dir), chunk_size, chunk_overlap)


def ingest_corpus(
    corpus_dir: str | Path,
    vector_store: Any,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    rebuild: bool = False,
) -> IngestReport:
    """Idempotent ingest: skip when the collection already has content."""
    existing = vector_store.count()
    if existing and not rebuild:
        logger.info("vector store already populated", extra={"chunks": existing})
        return IngestReport(documents=0, chunks=existing, skipped_existing=True)

    if rebuild and existing:
        logger.warning("rebuilding vector store", extra={"dropped_chunks": existing})
        vector_store.reset()

    chunks = load_and_chunk(corpus_dir, chunk_size, chunk_overlap)
    vector_store.add_documents(chunks)
    logger.info("ingest complete", extra={"chunks": len(chunks)})
    return IngestReport(
        documents=len({c.metadata.get("source") for c in chunks}),
        chunks=len(chunks),
        skipped_existing=False,
    )


def iter_sources(documents: Iterable[Document]) -> list[str]:
    """Unique, ordered source filenames — used by evaluation and reporting."""
    seen: dict[str, None] = {}
    for doc in documents:
        seen.setdefault(str(doc.metadata.get("source", "unknown")), None)
    return list(seen)

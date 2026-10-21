"""Hindi-aware chunking behaviour."""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_text_splitters")

from langchain_core.documents import Document

from hindi_rag.ingestion import chunk_documents


def _doc(text: str, source: str = "भारत.txt") -> Document:
    return Document(page_content=text, metadata={"source": source})


def test_chunks_respect_size() -> None:
    text = "यह एक वाक्य है। " * 300
    chunks = chunk_documents([_doc(text)], chunk_size=200, chunk_overlap=20)
    assert chunks
    assert all(len(c.page_content) <= 200 for c in chunks)


def test_metadata_preserved_and_indexed() -> None:
    chunks = chunk_documents([_doc("वाक्य। " * 100, "दिल्ली.txt")], 128, 16)
    assert all(c.metadata["source"] == "दिल्ली.txt" for c in chunks)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_overlap_must_be_smaller_than_chunk() -> None:
    with pytest.raises(ValueError):
        chunk_documents([_doc("पाठ")], chunk_size=100, chunk_overlap=100)


def test_empty_and_whitespace_chunks_dropped() -> None:
    chunks = chunk_documents([_doc("   \n\n  ")], 128, 16)
    assert chunks == []


def test_devanagari_danda_preserved() -> None:
    text = "पहला वाक्य। दूसरा वाक्य। तीसरा वाक्य।"
    chunks = chunk_documents([_doc(text)], chunk_size=24, chunk_overlap=4)
    joined = " ".join(c.page_content for c in chunks)
    assert "।" in joined

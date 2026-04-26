"""
Data ingestion pipeline.
Loads Hindi text documents, chunks them, embeds with multilingual-e5-base.
Adapted from AI4Bharat/IndicLLMSuite data patterns + World Bank WhatsApp-RAG-Example.
"""
import os
import hashlib
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain_core.documents import Document

from .config import get_settings

cfg = get_settings()

# Separators that work well for Hindi (Devanagari uses । as sentence-end)
HINDI_SEPARATORS = ["।\n", "।", "\n\n", "\n", " ", ""]


def load_documents(corpus_dir: str) -> List[Document]:
    path = Path(corpus_dir)
    docs: List[Document] = []

    txt_loader = DirectoryLoader(
        str(path), glob="**/*.txt", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}, silent_errors=True
    )
    docs.extend(txt_loader.load())

    for pdf_path in path.rglob("*.pdf"):
        docs.extend(PyPDFLoader(str(pdf_path)).load())

    for doc in docs:
        doc.metadata["source_hash"] = hashlib.md5(
            doc.page_content.encode()
        ).hexdigest()[:8]

    return docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=HINDI_SEPARATORS,
    )
    return splitter.split_documents(docs)


def load_and_chunk(corpus_dir: str = "./data/corpus") -> List[Document]:
    docs = load_documents(corpus_dir)
    chunks = chunk_documents(docs)
    print(f"Loaded {len(docs)} documents → {len(chunks)} chunks")
    return chunks

"""
Vector store — ChromaDB with FastEmbed embeddings (ONNX, no Ollama/torch needed).
Uses multilingual-e5-small which handles Hindi well and runs fully locally via ONNX.
"""
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from .config import get_settings

cfg = get_settings()

_embeddings: Optional[FastEmbedEmbeddings] = None
_vectorstore: Optional[Chroma] = None


def get_embeddings() -> FastEmbedEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = FastEmbedEmbeddings(model_name=cfg.embedding_model)
    return _embeddings


def get_vectorstore(persist_dir: str = None) -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory=persist_dir or cfg.chroma_persist_dir,
            embedding_function=get_embeddings(),
            collection_name="hindi_rag_v2",
        )
    return _vectorstore


def ingest_chunks(chunks: List[Document], persist_dir: str = None) -> Chroma:
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=persist_dir or cfg.chroma_persist_dir,
        collection_name="hindi_rag_v2",
    )
    global _vectorstore
    _vectorstore = vs
    print(f"Ingested {len(chunks)} chunks into ChromaDB at {persist_dir or cfg.chroma_persist_dir}")
    return vs


def get_retriever() -> VectorStoreRetriever:
    return get_vectorstore().as_retriever(
        search_type="mmr",          # maximal marginal relevance — less repetitive results
        search_kwargs={"k": cfg.retrieval_k, "fetch_k": cfg.retrieval_k * 3},
    )

"""
LLM + RAG chain.
Supports Ollama (Qwen3-4B local) and Groq API as fallback.
Uses LangChain LCEL for composable, streaming-ready chain.
"""
from typing import AsyncIterator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from .config import get_settings
from .retrieval import get_retriever

cfg = get_settings()

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", cfg.system_prompt),
    ("human", "संदर्भ:\n{context}\n\nप्रश्न: {question}"),
])


def _get_llm():
    if cfg.llm_backend == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=cfg.groq_api_key,
            model=cfg.groq_model,
            temperature=0.3,
        )
    # default: Ollama local
    from langchain_ollama import ChatOllama
    return ChatOllama(
        base_url=cfg.ollama_base_url,
        model=cfg.ollama_model,
        temperature=0.3,
    )


def _format_docs(docs) -> str:
    return "\n\n---\n\n".join(
        f"[स्रोत: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in docs
    )


def build_rag_chain():
    retriever = get_retriever()
    llm = _get_llm()

    chain = (
        RunnableParallel(
            context=(retriever | _format_docs),
            question=RunnablePassthrough(),
        )
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


def answer(question: str) -> str:
    chain = build_rag_chain()
    return chain.invoke(question)


async def answer_stream(question: str) -> AsyncIterator[str]:
    chain = build_rag_chain()
    async for chunk in chain.astream(question):
        yield chunk

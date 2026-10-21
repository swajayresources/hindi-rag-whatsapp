"""Pydantic request/response contracts (the public API surface)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, examples=["भारत की राजधानी क्या है?"])
    include_sources: bool = Field(
        True, description="Return retrieved source snippets with the answer."
    )

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question cannot be empty or whitespace")
        return v


class Source(BaseModel):
    source: str = Field(..., description="Document path the chunk came from")
    snippet: str = Field(..., description="Retrieved chunk text (truncated for transport)")
    score: float | None = Field(None, description="Similarity score when available")


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source] = Field(default_factory=list)
    grounded: bool = Field(..., description="False when the model reported insufficient context")
    latency_ms: float
    request_id: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    environment: str
    llm_backend: str
    vector_store_chunks: int
    checks: dict[str, bool]


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: str

"""End-to-end smoke test against the real index (no LLM required)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from hindi_rag.api.app import create_app  # noqa: E402
from hindi_rag.config import Settings  # noqa: E402

settings = Settings(
    _env_file=None,
    llm_backend="ollama",
    twilio_validate_signature=False,
    corpus_dir=ROOT / "data" / "corpus",
)
app = create_app(settings)

with TestClient(app) as client:
    health = client.get("/health")
    print("health:", health.status_code, health.json())

    ready = client.get("/ready")
    print("ready:", ready.status_code, ready.json())

    # Chat must fail cleanly (503) because no LLM credentials are configured,
    # not blow up with a stack trace.
    chat = client.post("/v1/chat", json={"question": "भारत की राजधानी क्या है?"})
    print("chat:", chat.status_code, chat.json())

    docs = client.get("/openapi.json").json()
    print("routes:", sorted(docs["paths"].keys()))

"""API contract tests (fake RAG service, no vector DB, no LLM)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from hindi_rag.api.app import create_app


@pytest.fixture()
def client(settings, fake_service):
    app = create_app(settings, service=fake_service, bootstrap_index=False)
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_degraded_without_vector_store(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert "checks" in body
    assert body["llm_backend"] == "ollama"


def test_chat_returns_answer_and_sources(client) -> None:
    response = client.post("/v1/chat", json={"question": "भारत की राजधानी क्या है?"})
    assert response.status_code == 200
    body = response.json()
    assert "नई दिल्ली" in body["answer"]
    assert body["grounded"] is True
    assert body["sources"]
    assert body["request_id"]


def test_legacy_chat_alias_still_works(client) -> None:
    response = client.post("/chat", json={"question": "प्रश्न"})
    assert response.status_code == 200


def test_empty_question_is_rejected(client) -> None:
    response = client.post("/v1/chat", json={"question": "   "})
    assert response.status_code == 422


def test_question_too_long_is_rejected(client) -> None:
    response = client.post("/v1/chat", json={"question": "क" * 5000})
    assert response.status_code == 422


def test_request_id_is_echoed(client) -> None:
    response = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert response.headers["X-Request-ID"] == "abc123"
    assert "X-Response-Time-Ms" in response.headers


def test_whatsapp_webhook_returns_twiml(client) -> None:
    response = client.post(
        "/whatsapp/webhook",
        data={"From": "whatsapp:+10000000000", "Body": "भारत की राजधानी क्या है?"},
    )
    assert response.status_code == 200
    assert "Response" in response.text
    assert "नई दिल्ली" in response.text


def test_whatsapp_webhook_empty_body(client) -> None:
    response = client.post(
        "/whatsapp/webhook",
        data={"From": "whatsapp:+10000000000", "Body": "  "},
    )
    assert response.status_code == 200
    assert "कृपया" in response.text


def test_whatsapp_signature_enforced_when_enabled(settings, fake_service) -> None:
    settings.twilio_validate_signature = True
    settings.twilio_auth_token = SecretStr("test-token")
    app = create_app(settings, service=fake_service, bootstrap_index=False)
    with TestClient(app) as client:
        response = client.post(
            "/whatsapp/webhook",
            data={"From": "whatsapp:+100", "Body": "प्रश्न"},
        )
    assert response.status_code == 403
    assert response.json()["error"] == "invalid_signature"


def test_openapi_schema_is_available(client) -> None:
    schema = client.get("/openapi.json").json()
    assert "/v1/chat" in schema["paths"]
    assert "/health" in schema["paths"]

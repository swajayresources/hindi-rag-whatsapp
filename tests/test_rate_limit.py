"""Rate limiting and request-context middleware."""

from __future__ import annotations

from fastapi.testclient import TestClient

from hindi_rag.api.app import create_app


def _client(settings, service, requests: int):
    settings.rate_limit_requests = requests
    settings.rate_limit_window_s = 60
    app = create_app(settings, service=service, bootstrap_index=False)
    return TestClient(app)


def test_rate_limit_returns_429_after_budget(settings, fake_service) -> None:
    with _client(settings, fake_service, requests=2) as client:
        assert client.post("/v1/chat", json={"question": "एक"}).status_code == 200
        assert client.post("/v1/chat", json={"question": "दो"}).status_code == 200
        blocked = client.post("/v1/chat", json={"question": "तीन"})
    assert blocked.status_code == 429
    assert blocked.json()["error"] == "rate_limited"
    assert "Retry-After" in blocked.headers


def test_health_is_exempt_from_rate_limit(settings, fake_service) -> None:
    with _client(settings, fake_service, requests=1) as client:
        client.post("/v1/chat", json={"question": "एक"})
        assert client.post("/v1/chat", json={"question": "दो"}).status_code == 429
        assert client.get("/health").status_code == 200

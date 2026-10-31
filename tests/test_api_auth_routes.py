"""HTTP-level authentication tests for the session and chat routes."""
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import chat, sessions
from app.core.exceptions import register_exception_handlers


class FakeTenantRepository:
    def __init__(self, tenant_id="tenant-A"):
        self.tenant_id = tenant_id
        self.lookups = []

    def resolve_api_key(self, token_hash):
        self.lookups.append(token_hash)
        return self.tenant_id if token_hash == "hashed-valid-key" else None


class FakeSessionService:
    def __init__(self):
        self.calls = []

    def create_session(self, tenant_id, customer_id):
        self.calls.append((tenant_id, str(customer_id)))
        return SimpleNamespace(
            session_id="session-1",
            tenant_id=tenant_id,
            customer_id=str(customer_id),
            session_api_token="session-secret",
            customer_token="customer-secret",
        )


class FakeChatService:
    def __init__(self):
        self.ask_calls = []
        self.history_calls = []

    async def ask(self, session_id, token, query, lang):
        self.ask_calls.append((session_id, token, query, lang))
        return SimpleNamespace(answer="answer", sources=[], tenant_id="tenant-A")

    def record_exchange(self, *args):
        pass

    def get_history(self, session_id, token):
        self.history_calls.append((session_id, token))
        return {"history": [], "customer_id": "customer-1"}


class FakeQAService:
    def __init__(self):
        self.calls = []

    async def get_questions(self, session_id, token, lang):
        self.calls.append((session_id, token, lang))
        return {
            "session_id": session_id,
            "tenant_id": "tenant-A",
            "language": lang,
            "questions": [],
        }


def make_client():
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(sessions.router)
    app.include_router(chat.router)
    app.state.database_backend = "postgres"
    app.state.tenant_repo = FakeTenantRepository()
    app.state.session_service = FakeSessionService()
    app.state.chat_service = FakeChatService()
    app.state.qa_service = FakeQAService()
    return TestClient(app), app


def test_postgres_session_creation_requires_tenant_api_key():
    client, app = make_client()

    response = client.post(
        "/create-session",
        json={"tenant_id": "tenant-A", "customer_id": "customer-1"},
    )

    assert response.status_code == 401
    assert response.json()["error"] == "TENANT_API_KEY_REQUIRED"
    assert app.state.session_service.calls == []


def test_postgres_session_creation_rejects_invalid_tenant_api_key():
    client, app = make_client()

    response = client.post(
        "/create-session",
        headers={"X-Tenant-API-Key": "invalid-key"},
        json={"tenant_id": "tenant-A", "customer_id": "customer-1"},
    )

    assert response.status_code == 401
    assert response.json()["error"] == "INVALID_TENANT_API_KEY"
    assert app.state.session_service.calls == []


def test_postgres_session_creation_rejects_tenant_id_spoofing():
    client, app = make_client()

    response = client.post(
        "/create-session",
        headers={"X-Tenant-API-Key": "valid-key"},
        json={"tenant_id": "tenant-B", "customer_id": "customer-1"},
    )

    assert response.status_code == 403
    assert response.json()["error"] == "TENANT_ID_MISMATCH"
    assert app.state.session_service.calls == []


def test_postgres_session_creation_uses_tenant_resolved_from_api_key(monkeypatch):
    from app.api.routes import sessions as sessions_route

    monkeypatch.setattr(sessions_route, "hash_token", lambda value: f"hashed-{value}")
    client, app = make_client()

    response = client.post(
        "/create-session",
        headers={"X-Tenant-API-Key": "valid-key"},
        json={"tenant_id": "tenant-A", "customer_id": "customer-1"},
    )

    assert response.status_code == 200
    assert response.json()["tenant_id"] == "tenant-A"
    assert app.state.tenant_repo.lookups == ["hashed-valid-key"]
    assert app.state.session_service.calls == [("tenant-A", "customer-1")]


@pytest.mark.parametrize("path,method", [
    ("/ask?session_id=session-1", "post"),
    ("/get-questions?session_id=session-1", "get"),
    ("/chat-history?session_id=session-1", "get"),
])
def test_chat_endpoints_require_bearer_authentication(path, method):
    client, app = make_client()

    response = getattr(client, method)(
        path,
        json={"query": "hello"} if method == "post" else None,
    )

    assert response.status_code == 403
    assert app.state.chat_service.ask_calls == []
    assert app.state.chat_service.history_calls == []
    assert app.state.qa_service.calls == []


def test_chat_endpoints_forward_bearer_token_to_services():
    client, app = make_client()
    headers = {"Authorization": "Bearer session-secret"}

    ask = client.post(
        "/ask?session_id=session-1",
        headers=headers,
        json={"query": "hello"},
    )
    questions = client.get("/get-questions?session_id=session-1", headers=headers)
    history = client.get("/chat-history?session_id=session-1", headers=headers)

    assert ask.status_code == 200
    assert questions.status_code == 200
    assert history.status_code == 200
    assert app.state.chat_service.ask_calls == [
        ("session-1", "session-secret", "hello", "en")
    ]
    assert app.state.qa_service.calls == [("session-1", "session-secret", "en")]
    assert app.state.chat_service.history_calls == [("session-1", "session-secret")]

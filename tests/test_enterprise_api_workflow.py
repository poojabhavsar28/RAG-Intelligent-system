from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import chat, documents, health, sessions
from app.core.exceptions import register_exception_handlers
from app.core.security import hash_token


TENANT_KEY = "workflow-tenant-key"
SESSION_TOKEN = "workflow-session-token"
CUSTOMER_TOKEN = "workflow-customer-token"
SESSION_ID = "workflow-session-id"


class FakeTenantRepository:
    def resolve_api_key(self, token_hash):
        return "tenant-workflow" if token_hash == hash_token(TENANT_KEY) else None


class FakeSessionService:
    def create_session(self, tenant_id, customer_id):
        return SimpleNamespace(
            session_id=SESSION_ID,
            tenant_id=tenant_id,
            customer_id=str(customer_id),
            session_api_token=SESSION_TOKEN,
            customer_token=CUSTOMER_TOKEN,
        )


class FakeDocumentIngestionService:
    def __init__(self):
        self.documents = {}

    def ingest_text(self, tenant_id, filename, content, language):
        document_id = f"{tenant_id}-document-1"
        self.documents[document_id] = content
        return SimpleNamespace(
            document_id=document_id,
            tenant_id=tenant_id,
            filename=filename,
            chunk_count=1,
        )


class FakeChatService:
    def __init__(self, documents):
        self.documents = documents
        self.history = []

    async def ask(self, session_id, token, query, lang):
        assert session_id == SESSION_ID
        assert token == SESSION_TOKEN
        assert lang == "en"
        document_text = " ".join(self.documents.values())
        answer = document_text if "refund" in query.lower() else "I don't know."
        return SimpleNamespace(
            answer=answer,
            sources=[{"content": document_text}],
            tenant_id="tenant-workflow",
        )

    def record_exchange(self, session_id, tenant_id, question, answer):
        self.history.extend([
            {
                "role": "user",
                "message": question,
                "sender": "customer",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": session_id,
            },
            {
                "role": "assistant",
                "message": answer,
                "sender": "assistant",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": session_id,
            },
        ])

    def get_history(self, session_id, token):
        assert session_id == SESSION_ID
        assert token == CUSTOMER_TOKEN
        return {"history": self.history, "customer_id": "customer-workflow"}


class FakeQAService:
    async def get_questions(self, session_id, token, lang):
        assert session_id == SESSION_ID
        assert token == SESSION_TOKEN
        return {
            "session_id": session_id,
            "tenant_id": "tenant-workflow",
            "language": lang,
            "questions": [],
        }


class FakeConnection:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def execute(self, statement):
        assert str(statement) == "SELECT 1"


class FakePostgresEngine:
    def connect(self):
        return FakeConnection()


def test_upload_session_ask_and_chat_history_workflow():
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(sessions.router)
    app.include_router(chat.router)
    app.include_router(documents.router)
    app.include_router(health.router)

    documents_service = FakeDocumentIngestionService()
    app.state.database_backend = "postgres"
    app.state.postgres_engine = FakePostgresEngine()
    app.state.ai_assistant = SimpleNamespace(initialized_global_models_flag=True)
    app.state.tenant_repo = FakeTenantRepository()
    app.state.document_ingestion_service = documents_service
    app.state.session_service = FakeSessionService()
    app.state.chat_service = FakeChatService(documents_service.documents)
    app.state.qa_service = FakeQAService()
    client = TestClient(app)

    health_response = client.get("/health")
    assert health_response.status_code == 200
    assert health_response.json() == {
        "status": "healthy",
        "database": "ok",
        "ai_assistant": "ok",
    }

    upload = client.post(
        "/documents/upload",
        headers={"X-Tenant-API-Key": TENANT_KEY},
        files={"file": ("refund-policy.txt", b"Refunds are accepted within 30 days.", "text/plain")},
        data={"language": "en"},
    )
    assert upload.status_code == 201
    assert upload.json()["chunks"] == 1

    session = client.post(
        "/create-session",
        headers={"X-Tenant-API-Key": TENANT_KEY},
        json={"tenant_id": "tenant-workflow", "customer_id": "customer-workflow"},
    )
    assert session.status_code == 200
    assert session.json()["session_id"] == SESSION_ID

    session_headers = {"Authorization": f"Bearer {SESSION_TOKEN}"}
    questions = client.get(
        f"/get-questions?session_id={SESSION_ID}&lang=en",
        headers=session_headers,
    )
    assert questions.status_code == 200
    assert questions.json()["tenant_id"] == "tenant-workflow"

    answer = client.post(
        f"/ask?session_id={SESSION_ID}&lang=en",
        headers=session_headers,
        json={"query": "What is the refund policy?"},
    )
    assert answer.status_code == 200
    assert answer.json()["answer"] == "Refunds are accepted within 30 days."

    history = client.get(
        f"/chat-history?session_id={SESSION_ID}",
        headers={"Authorization": f"Bearer {CUSTOMER_TOKEN}"},
    )
    assert history.status_code == 200
    assert [message["role"] for message in history.json()["history"]] == ["user", "assistant"]
    assert history.json()["history"][1]["message"] == answer.json()["answer"]


def test_health_reports_ai_initialization_failure():
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(health.router)
    app.state.database_backend = "postgres"
    app.state.postgres_engine = FakePostgresEngine()
    app.state.ai_assistant = SimpleNamespace(
        initialized_global_models_flag=False,
        initialization_error="Ollama model unavailable",
    )

    response = TestClient(app).get("/health")

    assert response.status_code == 503

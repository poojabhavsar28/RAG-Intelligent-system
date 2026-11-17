import asyncio
from types import SimpleNamespace

import pytest

from app.core.exceptions import UnauthorizedError
from app.core.security import hash_token
from app.services.qa_service import QAService


class FakeSessionRepository:
    def __init__(self, session):
        self.session = session
        self.lookups = []

    def get_by_id_and_token(self, session_id, token_hash):
        self.lookups.append((session_id, token_hash))
        if token_hash != hash_token("valid-session-token"):
            return None
        return self.session


class FakeRagEngine:
    def __init__(self):
        self.calls = []

    async def get_questions(self, tenant_id, language):
        self.calls.append((tenant_id, language))
        return [{"question": "Where is support?", "created_at": None}]


def test_get_questions_requires_valid_session_token():
    sessions = FakeSessionRepository(SimpleNamespace(tenant_id="tenant-A"))
    rag = FakeRagEngine()
    service = QAService(sessions, rag)

    with pytest.raises(UnauthorizedError):
        asyncio.run(service.get_questions("session-1", "invalid-token", "en"))

    assert rag.calls == []


def test_get_questions_uses_tenant_from_authenticated_session():
    sessions = FakeSessionRepository(SimpleNamespace(tenant_id="tenant-A"))
    rag = FakeRagEngine()
    service = QAService(sessions, rag)

    result = asyncio.run(
        service.get_questions("session-1", "valid-session-token", "en")
    )

    assert sessions.lookups == [("session-1", hash_token("valid-session-token"))]
    assert rag.calls == [("tenant-A", "en")]
    assert result["tenant_id"] == "tenant-A"
    assert result["questions"] == [{"question": "Where is support?", "created_at": None}]

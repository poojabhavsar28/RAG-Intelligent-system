import asyncio
from types import SimpleNamespace

from app.core.security import hash_token
from app.services.chat_service import ChatService


class FakeSessionRepository:
    def get_by_id_and_token(self, session_id, token_hash):
        assert session_id == "session-1"
        assert token_hash == hash_token("session-secret")
        return SimpleNamespace(tenant_id="tenant-opaque-42", customer_id="customer-7")


class FakeRagEngine:
    async def process_query(self, **kwargs):
        assert kwargs["tenant_id"] == "tenant-opaque-42"
        return {"answer": "Answer", "sources": ["source-1"]}


def test_ask_result_contains_authenticated_tenant_for_history_write():
    service = ChatService(
        session_repo=FakeSessionRepository(),
        token_repo=object(),
        chat_repo=object(),
        rag_engine=FakeRagEngine(),
    )

    result = asyncio.run(
        service.ask("session-1", "session-secret", "question", "en")
    )

    assert result.answer == "Answer"
    assert result.sources == ["source-1"]
    assert result.tenant_id == "tenant-opaque-42"

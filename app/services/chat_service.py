"""
ChatService.

Replaces the logic previously inline in main.py's ask_question() and
get_chat_history() endpoint functions.

Bug fix folded in here: the original get_chat_history() took
`credentials: HTTPAuthorizationCredentials = Depends(security_scheme)` as a
*required* parameter (HTTPBearer() defaults to auto_error=True), but then
guarded the token-validation block with `if credentials:` — which is always
true for a required dependency. The check could never actually skip
validation; it read as if a token were optional when it wasn't. This
service makes that explicit: token validation always runs, no dead
conditional.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.core.exceptions import ForbiddenError, UnauthorizedError
from app.core.security import hash_token
from app.domain.rag_engine import RAGEngine
from app.repositories.chat_repository import ChatRepository
from app.repositories.session_repository import SessionRepository
from app.repositories.token_repository import TokenRepository


@dataclass
class AskResult:
    answer: str
    sources: List[Any]


class ChatService:
    def __init__(
        self,
        session_repo: SessionRepository,
        token_repo: TokenRepository,
        chat_repo: ChatRepository,
        rag_engine: RAGEngine,
    ):
        self._session_repo = session_repo
        self._token_repo = token_repo
        self._chat_repo = chat_repo
        self._rag_engine = rag_engine

    async def ask(self, session_id: str, session_api_token: str, query: str, lang: str) -> AskResult:
        session = self._session_repo.get_by_id_and_token(session_id, hash_token(session_api_token))
        if not session:
            raise UnauthorizedError("Invalid session or token", error_code="INVALID_SESSION")

        result = await self._rag_engine.process_query(
            query=query,
            tenant_id=session.tenant_id,
            customer_id=session.customer_id,
            lang=lang,
            session_id=session_id,
        )
        return AskResult(answer=result["answer"], sources=result.get("sources", []))

    def record_exchange(self, session_id: str, tenant_id: str, question: str, answer: str) -> None:
        """Called as a background task from the API route — the client gets
        their answer immediately and doesn't wait on this write."""
        self._chat_repo.append_exchange(session_id, tenant_id, question, answer)

    def get_history(self, session_id: str, customer_token: str) -> Dict[str, Any]:
        session = self._session_repo.get_by_id(session_id)
        if not session:
            raise UnauthorizedError("Session not found", error_code="SESSION_NOT_FOUND")

        token_record = self._token_repo.get_valid(hash_token(customer_token))
        if not token_record:
            raise UnauthorizedError("Invalid or expired customer token", error_code="INVALID_TOKEN")
        if token_record.tenant_id != session.tenant_id or str(token_record.customer_id) != str(session.customer_id):
            raise ForbiddenError("Tenant/Customer mismatch", error_code="TENANT_CUSTOMER_MISMATCH")

        messages = self._chat_repo.get_history(session_id)
        history = [
            {
                "role": m.role,
                "message": m.message,
                "sender": m.sender,
                "timestamp": m.timestamp.isoformat(),
                "session_id": session_id,
            }
            for m in messages
        ]
        return {"history": history, "customer_id": str(session.customer_id)}

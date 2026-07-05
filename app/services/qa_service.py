from typing import Any, Dict

from app.core.exceptions import NotFoundError
from app.domain.rag_engine import RAGEngine
from app.repositories.session_repository import SessionRepository


class QAService:
    def __init__(self, session_repo: SessionRepository, rag_engine: RAGEngine):
        self._session_repo = session_repo
        self._rag_engine = rag_engine

    async def get_questions(self, session_id: str, lang: str) -> Dict[str, Any]:
        session = self._session_repo.get_by_id(session_id)
        if not session:
            raise NotFoundError("Session not found", error_code="SESSION_NOT_FOUND")

        questions = await self._rag_engine.get_questions(session.tenant_id, lang)
        formatted = [
            {
                "question": q["question"],
                "created_at": q["created_at"].isoformat() if q.get("created_at") else None,
            }
            for q in questions
        ]
        return {
            "session_id": session_id,
            "tenant_id": session.tenant_id,
            "language": lang,
            "questions": formatted,
        }

from typing import Any, Dict, List

from pydantic import BaseModel


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    sources: List[Any] = []


class ChatMessageResponse(BaseModel):
    role: str
    message: str
    sender: str
    timestamp: str
    session_id: str


class ChatHistoryResponse(BaseModel):
    history: List[Dict[str, Any]]
    customer_id: str


class QuestionItem(BaseModel):
    question: str
    created_at: str | None = None


class QuestionsResponse(BaseModel):
    session_id: str
    tenant_id: str
    language: str
    questions: List[QuestionItem]

"""
Dependency providers for FastAPI's Depends() system.

Why this replaces the old approach: main.py used module-level globals
(`db_pool = None`, `ai_assistant = None`, reassigned inside the lifespan
function). That works, but it's untestable — there's no way to substitute a
fake database or fake AI engine for a unit test without monkeypatching
module globals, and every function that needs them has to `global` import
from main.

Instead, the app factory (app/main.py) builds each service once at startup
and stores them on `app.state`. These functions just retrieve them by name
for injection into route handlers. Swapping a real dependency for a test
double in a unit test is then just `app.state.chat_service = FakeChatService()`.
"""
from fastapi import Request

from app.services.chat_service import ChatService
from app.services.qa_service import QAService
from app.services.session_service import SessionService


def get_session_service(request: Request) -> SessionService:
    return request.app.state.session_service


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


def get_qa_service(request: Request) -> QAService:
    return request.app.state.qa_service

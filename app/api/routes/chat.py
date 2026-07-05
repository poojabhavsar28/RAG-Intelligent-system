from fastapi import APIRouter, BackgroundTasks, Depends, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.api.deps import get_chat_service, get_qa_service
from app.schemas.chat import AskRequest, AskResponse, ChatHistoryResponse, QuestionsResponse
from app.schemas.common import ErrorResponse
from app.services.chat_service import ChatService
from app.services.qa_service import QAService

router = APIRouter(tags=["chat"])
security_scheme = HTTPBearer()


@router.get(
    "/get-questions",
    response_model=QuestionsResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Retrieve predefined questions",
)
async def get_questions(
    session_id: str = Query(...),
    lang: str = Query("en"),
    qa_service: QAService = Depends(get_qa_service),
):
    result = await qa_service.get_questions(session_id, lang)
    return QuestionsResponse(**result)


@router.post(
    "/ask",
    response_model=AskResponse,
    responses={401: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Ask a question",
)
async def ask_question(
    request: AskRequest,
    background_tasks: BackgroundTasks,
    session_id: str = Query(...),
    lang: str = Query("en"),
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
    chat_service: ChatService = Depends(get_chat_service),
):
    result = await chat_service.ask(session_id, credentials.credentials, request.query, lang)

    # NOTE: the original code used asyncio.create_task() directly, which is
    # fire-and-forget with no framework awareness — if the request handler
    # returns and the app is mid-shutdown, that task can be silently
    # cancelled. FastAPI's BackgroundTasks ties the task's lifetime to the
    # response cycle properly.
    background_tasks.add_task(
        chat_service.record_exchange, session_id, result.tenant_id, request.query, result.answer
    )
    return AskResponse(answer=result.answer, sources=result.sources)


@router.get(
    "/chat-history",
    response_model=ChatHistoryResponse,
    responses={401: {"model": ErrorResponse}, 403: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Retrieve chat history by session_id",
)
async def get_chat_history(
    session_id: str = Query(...),
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
    chat_service: ChatService = Depends(get_chat_service),
):
    result = chat_service.get_history(session_id, credentials.credentials)
    return ChatHistoryResponse(**result)

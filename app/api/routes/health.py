from fastapi import APIRouter, Request
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.core.exceptions import DependencyUnavailableError
from app.schemas.common import ErrorResponse, HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": ErrorResponse}},
    summary="Health check endpoint",
)
async def health_check(request: Request):
    database_status = "unavailable"
    try:
        if getattr(request.app.state, "database_backend", "mysql") == "postgres":
            engine = getattr(request.app.state, "postgres_engine", None)
            if engine is None:
                raise DependencyUnavailableError("PostgreSQL engine is not initialized")
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        else:
            pool = getattr(request.app.state, "mysql_pool", None)
            if pool is None:
                raise DependencyUnavailableError("MySQL pool is not initialized")
            with pool.connection():
                pass
        database_status = "ok"
    except (DependencyUnavailableError, SQLAlchemyError):
        database_status = "unavailable"

    ai_assistant = getattr(request.app.state, "ai_assistant", None)
    if ai_assistant and getattr(ai_assistant, "initialization_error", None):
        ai_status = "unavailable"
    elif ai_assistant and getattr(ai_assistant, "initialized_global_models_flag", False):
        ai_status = "ok"
    else:
        ai_status = "initializing"

    if database_status != "ok":
        raise DependencyUnavailableError("Service unhealthy", error_code="SERVICE_UNHEALTHY")
    if ai_status != "ok":
        raise DependencyUnavailableError("AI models are not ready", error_code="AI_UNAVAILABLE")

    return HealthResponse(status="healthy", database=database_status, ai_assistant=ai_status)

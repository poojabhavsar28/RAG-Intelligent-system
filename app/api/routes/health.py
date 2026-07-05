from fastapi import APIRouter, Request

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
    pool = request.app.state.mysql_pool
    database_status = "unavailable"
    try:
        with pool.connection():
            database_status = "ok"
    except DependencyUnavailableError:
        database_status = "unavailable"

    ai_assistant = getattr(request.app.state, "ai_assistant", None)
    ai_status = "ok" if ai_assistant and getattr(ai_assistant, "initialized_global_models_flag", False) else "initializing"

    if database_status != "ok":
        raise DependencyUnavailableError("Service unhealthy", error_code="SERVICE_UNHEALTHY")

    return HealthResponse(status="healthy", database=database_status, ai_assistant=ai_status)

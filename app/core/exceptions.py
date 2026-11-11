"""
Application-level exceptions.

Why: the original endpoints caught bare `Exception` and returned
`detail=str(e)` directly to the client (see main.py's ask_question,
create_session, etc). That leaks internal error strings (stack traces,
SQL fragments, file paths) to API consumers, which is both an information
disclosure risk and unhelpful (the client can't branch on a string).

These typed exceptions carry a stable `error_code` the client can safely
switch on, and a `detail` that is safe to expose. Anything unexpected is
caught by the generic handler and returned as a fixed-shape 500 without
leaking internals; the real exception still gets logged with a stack trace
server-side.
"""
from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.logging_config import logging

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Base class for all handled application errors."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code: str = "INTERNAL_ERROR"

    def __init__(self, detail: str, error_code: str | None = None, status_code: int | None = None):
        self.detail = detail
        if error_code:
            self.error_code = error_code
        if status_code:
            self.status_code = status_code
        super().__init__(detail)


class NotFoundError(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    error_code = "NOT_FOUND"


class UnauthorizedError(AppError):
    status_code = status.HTTP_401_UNAUTHORIZED
    error_code = "UNAUTHORIZED"


class ForbiddenError(AppError):
    status_code = status.HTTP_403_FORBIDDEN
    error_code = "FORBIDDEN"


class ValidationFailedError(AppError):
    status_code = status.HTTP_400_BAD_REQUEST
    error_code = "VALIDATION_FAILED"


class DependencyUnavailableError(AppError):
    """Raised when a downstream dependency (DB, vector store, model) is down."""
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_code = "DEPENDENCY_UNAVAILABLE"


def _error_body(status_code: int, detail: str, error_code: str) -> dict:
    return {"status_code": status_code, "detail": detail, "error": error_code}


def register_exception_handlers(app) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        logger.warning(f"{exc.error_code}: {exc.detail}", extra={"path": request.url.path})
        return JSONResponse(status_code=exc.status_code, content=_error_body(exc.status_code, exc.detail, exc.error_code))

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.info(f"Validation error on {request.url.path}: {exc.errors()}")
        return JSONResponse(
            status_code=422,
            content=_error_body(422, "Request validation failed", "VALIDATION_FAILED"),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        # Full detail goes to logs (with stack trace); the client gets a
        # generic, safe message. This is the difference between "helpful for
        # debugging" and "leaking internals to the internet."
        logger.exception(f"Unhandled exception on {request.url.path}")
        return JSONResponse(
            status_code=500,
            content=_error_body(500, "An internal error occurred", "INTERNAL_SERVER_ERROR"),
        )
